""" Text modules
Only ProbEncoderTextClip and EncoderTextClip are verified.

Original and reference code: https://github.com/woodfrog/vse_infty/blob/master/lib/encoders.py
"""
import torch
import torch.nn as nn

from transformers import BertModel

from pcmepp.modules.aggr.gpo import GPO

from pcmepp.modules import clip

from pcmepp.modules.func import l2norm


class EncoderTextBert(nn.Module):
    """ Language Model with BERT (from VSE infty)
    original code: https://github.com/woodfrog/vse_infty/blob/master/lib/encoders.py
    """
    def __init__(self, embed_size, no_txtnorm=False, **kwargs):
        super(EncoderTextBert, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.backbone = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)
        self.gpool = GPO(32, 32)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.backbone(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)

        pooled_features, _ = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        return {'mean': pooled_features}


class ProbEncoderTextBert(nn.Module):
    """ Language Model with BERT (from VSE infty)
    original code: https://github.com/woodfrog/vse_infty/blob/master/lib/encoders.py
    """
    def __init__(self, embed_size, no_txtnorm=False, **kwargs):
        super(ProbEncoderTextBert, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.backbone = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)
        self.gpool = GPO(32, 32)

        self.std_linear = nn.Linear(768, embed_size)
        self.std_gpool = GPO(32, 32)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.backbone(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)

        pooled_features, _ = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        std_cap_emb = self.std_linear(bert_emb)
        std_pooled_features, _ = self.std_gpool(std_cap_emb, cap_len.to(cap_emb.device))

        return {'mean': pooled_features, 'std': std_pooled_features}


class CLIPTextBackbone(nn.Module):
    """CLIP Text backbone for EncoderTextClip
    """
    def __init__(self, clip_model, n_unc_layers=1, **kwargs):
        super(CLIPTextBackbone, self).__init__()
        model, _ = clip.load(
            clip_model, device='cuda',
            n_unc_layers=n_unc_layers,
            rand_init=kwargs.get('rand_init'))
        self.transformer = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.vocab_size = model.vocab_size

    def forward(self, x):
        """Handles variable size captions
        """
        # [batch_size, n_ctx, d_model]
        x = self.token_embedding(x)
        x = x + self.positional_embedding

        # NLD -> LND
        x = x.permute(1, 0, 2)
        transformer_out = self.transformer(x)

        emb = {}
        # LND -> NLD
        for key, _emb in transformer_out.items():
            cur_emb = _emb.permute(1, 0, 2)
            if key == 'mean':
                cur_emb = self.ln_final(cur_emb)
            emb[key] = cur_emb
        return emb


class EncoderTextClip(nn.Module):
    """ Language Model with CLIP text encoder (deterministic model)
    """
    def __init__(self, embed_size, clip_model, no_txtnorm=False, n_unc_layers=None, **kwargs):
        super(EncoderTextClip, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.backbone = CLIPTextBackbone(clip_model, n_unc_layers=n_unc_layers, **kwargs)

        # XXX CLIP configuration is hard coded here.
        if 'ViT-L' in clip_model:
            d = 768
        else:
            d = 512
        self.linear = nn.Linear(d, embed_size)
        if kwargs.get('no_gpo'):
            self.gpool = None
        else:
            self.gpool = GPO(32, 32)

    def reinit(self, n_reinit_layers):
        for n in range(n_reinit_layers):
            self.backbone.transformer.resblocks[11-n].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            # TODO configurable?
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_mean(self, x, lengths):
        emb = self.backbone(x)
        mean_emb = emb['mean']

        # Embed word ids to vectors
        cap_len = lengths
        cap_emb = self.linear(mean_emb)
        if self.gpool is None:
            # return eot token
            # https://github.com/openai/CLIP/blob/main/clip/model.py#L352-L354
            pooled_features = cap_emb[torch.arange(len(lengths)), lengths.long() - 1, :]
        else:
            pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)
        return pooled_features, emb

    def forward(self, x, lengths):
        mean_feature, _ = self.forward_mean(x, lengths)
        out_feats = {}
        out_feats['mean'] = mean_feature
        return out_feats

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, fixed_blocks=None):
        if not fixed_blocks:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # freeze all params first, then adjust the base parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            # unfreeze probabilistic blocks
            for param in self.backbone.transformer.prob_resblocks.parameters():
                param.requires_grad = True
            for n in range(12 - fixed_blocks):
                for param in self.backbone.transformer.resblocks[11-n].parameters():
                    param.requires_grad = True


class ProbEncoderTextClip(EncoderTextClip):
    """ Language Model with CLIP text encoder (stochastic model)
    """
    def __init__(self, embed_size, clip_model, no_txtnorm=False,
                 n_unc_layers=2, sigma_ln_init=0.01, sigma_ln_init_bias=0, bias_init=0, **kwargs):
        super(ProbEncoderTextClip, self).__init__(embed_size, clip_model, no_txtnorm, n_unc_layers, **kwargs)

        if 'ViT-L' in clip_model:
            d = 768
        else:
            d = 512
        self.std_linear = nn.Linear(d, embed_size)
        if self.gpool is not None:
            self.std_gpool = GPO(32, 32)
        else:
            self.std_gpool = None
        if sigma_ln_init is not None:
            self.text_std_ln = nn.LayerNorm(embed_size)
            nn.init.constant_(self.text_std_ln.weight, sigma_ln_init)
            nn.init.constant_(self.text_std_ln.bias, sigma_ln_init_bias)
        else:
            self.text_std_ln = nn.Identity()
            # As there is no layer norm, set the bias of the linear layer to -4 to prevent too large std
            nn.init.constant_(self.std_linear.bias, bias_init)

    def forward(self, x, lengths):
        mean_feature, emb = self.forward_mean(x, lengths)
        out_feats = {}
        out_feats['mean'] = mean_feature

        std_cap_emb = self.std_linear(emb['std'])
        if self.gpool is None:
            # return eot token
            # https://github.com/openai/CLIP/blob/main/clip/model.py#L352-L354
            pooled_features = std_cap_emb[torch.arange(len(lengths)), lengths.long() - 1, :]
        else:
            pooled_features, _ = self.std_gpool(std_cap_emb, lengths)
        out_feats['std'] = self.text_std_ln(pooled_features)

        return out_feats

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.transformer.prob_resblocks.parameters():
            param.requires_grad = True
