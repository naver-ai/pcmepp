""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
Original code: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from huggingface_hub import PyTorchModelHubMixin
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    unc_layers: int = 2
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.0  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = "avg"  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = "linear"  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.0  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    unc_layers: int = 2
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = "mlp"
    pooler_type: str = "mean_pooler"
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = torch.bfloat16
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
    embed_dim: int, vision_cfg: CLIPVisionCfg, quick_gelu: bool = False, cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        input_patchnorm=vision_cfg.input_patchnorm,
        global_average_pool=vision_cfg.global_average_pool,
        attentional_pool=vision_cfg.attentional_pool,
        n_queries=vision_cfg.n_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
        unc_layers=vision_cfg.unc_layers,
    )

    return visual


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    uncertainty_text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.unc_layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text, uncertainty_text, text_cfg.layers


class HfPCMEPPModel(nn.Module, PyTorchModelHubMixin):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text, uncertainty_text, text_depth = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.uncertainty_transformer = uncertainty_text.transformer
        self.unc_layers = text_cfg["unc_layers"]
        self.text_depth = text_depth
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.uncertainty_text_projection = uncertainty_text.text_projection
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        shift = torch.ones(1)
        negative_scale = torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return {
            'mean': F.normalize(features['mean'], dim=-1) if normalize else features,
            'std': features['std']
        }

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for idx in range(0, self.text_depth - self.unc_layers):
            x = self.transformer.resblocks._modules[str(idx)](x, attn_mask=self.attn_mask)
        mean_x = x
        std_x = self.uncertainty_transformer(x, attn_mask=self.attn_mask)
        for idx in range(self.text_depth - self.unc_layers, self.text_depth):
            mean_x = self.transformer.resblocks._modules[str(idx)](mean_x, attn_mask=self.attn_mask)
        x = mean_x.permute(1, 0, 2)  # LND -> NLD
        std_x = std_x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        std_x = std_x[torch.arange(std_x.shape[0]), text.argmax(dim=-1)] @ self.uncertainty_text_projection
        return {
            'mean': F.normalize(x, dim=-1) if normalize else x,
            'std': std_x
        }

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(images, normalize=True) if images is not None else {}
        text_features = self.encode_text(texts, normalize=True) if texts is not None else {}
        return {
            "image_features": image_features.get('mean'),
            "image_stds": image_features.get('std'),
            "text_features": text_features.get('mean'),
            "text_stds": text_features.get('std'),
            "logit_scale": self.logit_scale.exp(),
            "shift": self.shift,
            "negative_scale": self.negative_scale,
        }


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.to(dtype)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(dtype)

        if isinstance(layer, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(layer, (HfPCMEPPModel, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(layer, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(layer, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(layer, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat
