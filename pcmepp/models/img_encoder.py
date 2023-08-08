""" Image modules
Only ProbEncoderClipImageFull and EncoderClipImageFull are verified.

Original and reference code: https://github.com/woodfrog/vse_infty/blob/master/lib/encoders.py
"""
import torch
import torch.nn as nn
import numpy as np

from pcmepp.modules.aggr.gpo import GPO, AvgPool
from pcmepp.modules.mlp import MLP

from pcmepp.modules import clip

from pcmepp.modules.func import l2norm


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, no_sigma_ln=False, bias_init=0, **kwargs):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        if kwargs.get('no_gpo'):
            self.gpool = AvgPool()
        else:
            self.gpool = GPO(32, 32)
        self.init_weights(no_sigma_ln, bias_init)
        # FC => 512, 1024

    def init_weights(self, no_sigma_ln, bias_init):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        if no_sigma_ln:
            # As there is no layer norm, set the bias of the linear layer to -4 to prevent too large std
            # nn.init.constant_(self.fc.bias, -4)
            self.fc.bias.data.fill_(bias_init)
        else:
            self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for the embedding transformation
            features = self.mlp(images) + features

        features, _ = self.gpool(features, image_lengths)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


class ProbEncoderImageAggrBUTD(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, bias_init=0, **kwargs):
        super(ProbEncoderImageAggrBUTD, self).__init__()
        self.mean_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm, bias_init, **kwargs)
        self.std_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm, bias_init, **kwargs)

    def forward(self, images, image_lengths):
        mean = self.mean_encoder(images, image_lengths)
        std = self.std_encoder(images, image_lengths)
        return {'mean': mean, 'std': std}


class EncoderImageAggrBUTD(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, bias_init=0, **kwargs):
        super(EncoderImageAggrBUTD, self).__init__()
        self.mean_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm, bias_init, **kwargs)

    def forward(self, images, image_lengths):
        return {'mean': self.mean_encoder(images, image_lengths)}


class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, size_augment=0.2, **kwargs):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm, **kwargs)
        self.backbone_freezed = False
        self.size_augment = size_augment

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > self.size_augment:
                    feat_i = base_features[i][np.where(rand_list_1[i] > self.size_augment * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.image_encoder(base_features, feat_lengths)

        return {'mean': features}

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()


class ProbEncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, size_augment=0.2, **kwargs):
        super(ProbEncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm, **kwargs)
        self.std_image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=no_imgnorm, **kwargs)
        self.backbone_freezed = False
        self.size_augment = size_augment

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > self.size_augment:
                    feat_i = base_features[i][np.where(rand_list_1[i] > self.size_augment * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.image_encoder(base_features, feat_lengths)
        std_features = self.std_image_encoder(base_features, feat_lengths)

        return {'mean': features, 'std': std_features}

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()


# Visual Model with CLIP visual encoder
class EncoderClipImageFull(nn.Module):
    def __init__(self, clip_model, img_dim, embed_size, no_imgnorm=False, size_augment=0.2, n_unc_layers=None, **kwargs):
        super(EncoderClipImageFull, self).__init__()
        model, _ = clip.load(
            clip_model, device='cuda',
            n_unc_layers=n_unc_layers,
            rand_init=kwargs.get('rand_init'))
        self.backbone = model.visual
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type='backbone', no_imgnorm=no_imgnorm, **kwargs)
        self.backbone_freezed = False
        self.size_augment = size_augment

    def forward_base(self, images, is_prob=False):
        """Extract image feature vectors."""
        base_features = self.backbone(images)
        # [batch_size, 50, 768]
        # 50 = 1 (cls token) + 7 * 7 => (B/16) 197 = 1 (cls token) + 14 * 14

        feat_lengths = torch.zeros(base_features['mean'].size(0)).to(base_features['mean'].device)
        feat_lengths[:] = base_features['mean'].size(1)

        if self.training:
            n_feats = base_features['mean'].size(0)
            n_tokens = base_features['mean'].size(1)
            emb_dim = base_features['mean'].size(2)

            new_mean = []
            new_std = []
            feat_lengths = []
            # TODO configurable
            # TODO functionize... remove ugly std check
            for i in range(n_feats):
                prob = np.random.rand()
                if prob > self.size_augment:
                    # 80% of features randomly drop at most 20% tokens
                    # (following the original implementation of VSE infty)
                    tokenwise_prob = np.random.rand(n_tokens)
                    selected = np.where(tokenwise_prob > self.size_augment * prob)[0]
                    if len(selected) == n_tokens:
                        new_mean.append(base_features['mean'][i])
                        if is_prob:
                            new_std.append(base_features['std'][i])
                        feat_lengths.append(n_tokens)
                    else:
                        selected = np.array(list(range(n_tokens)))[selected]
                        new_mean.append(
                            torch.cat([
                                base_features['mean'][i][selected],
                                torch.zeros(n_tokens - len(selected), emb_dim).to(base_features['mean'].device),
                            ], dim=0)
                        )
                        if is_prob:
                            new_std.append(
                                torch.cat([
                                    base_features['std'][i][selected],
                                    torch.zeros(n_tokens - len(selected), emb_dim).to(base_features['mean'].device),
                                ], dim=0)
                            )
                        feat_lengths.append(len(selected))
                else:
                    new_mean.append(base_features['mean'][i])
                    if is_prob:
                        new_std.append(base_features['std'][i])
                    feat_lengths.append(n_tokens)

            base_features['mean'] = torch.stack(new_mean, dim=0)
            if is_prob:
                base_features['std'] = torch.stack(new_std, dim=0)
            feat_lengths = torch.tensor(feat_lengths).to(base_features['mean'].device)

        return base_features, feat_lengths

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

    def forward(self, images):
        base_features, feat_lengths = self.forward_base(images)

        features = {}
        features['mean'] = self.image_encoder(base_features['mean'], feat_lengths)
        return features

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
            for n in range(12 - fixed_blocks):
                for param in self.backbone.transformer.resblocks[11-n].parameters():
                    param.requires_grad = True


class ProbEncoderClipImageFull(EncoderClipImageFull):
    def __init__(self, clip_model, img_dim, embed_size, no_imgnorm=False,
                 sigma_ln_init=0.01, sigma_ln_init_bias=0, n_unc_layers=2, size_augment=0.2, **kwargs):
        super(ProbEncoderClipImageFull, self).__init__(
            clip_model, img_dim, embed_size, no_imgnorm=no_imgnorm, size_augment=size_augment, n_unc_layers=n_unc_layers, **kwargs)
        self.image_std_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type='backbone', no_imgnorm=True, no_sigma_ln=sigma_ln_init is None, **kwargs)
        if sigma_ln_init is not None:
            self.image_std_ln = nn.LayerNorm(embed_size)
            nn.init.constant_(self.image_std_ln.weight, sigma_ln_init)
            nn.init.constant_(self.image_std_ln.bias, sigma_ln_init_bias)
        else:
            self.image_std_ln = nn.Identity()
            # As there is no layer norm, set the bias of the linear layer to -10 to prevent too large std
            # nn.init.constant_(self.std_linear.bias, -10)

    def forward(self, images):
        base_features, feat_lengths = self.forward_base(images, is_prob=True)

        features = {}
        features['mean'] = self.image_encoder(base_features['mean'], feat_lengths)
        std_feat = self.image_std_encoder(base_features['std'], feat_lengths)
        std_feat = self.image_std_ln(std_feat)
        features['std'] = std_feat

        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.uncertainty_transformer.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self, fixed_blocks=None):
        if not fixed_blocks:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # freeze all params first, then adjust the base parameters
            self.freeze_backbone()
            for n in range(12 - fixed_blocks):
                for param in self.backbone.transformer.resblocks[11-n].parameters():
                    param.requires_grad = True
