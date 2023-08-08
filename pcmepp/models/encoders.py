""" Image and text encoders for PCME++

Reference code: https://github.com/woodfrog/vse_infty/blob/master/lib/encoders.py
"""
from pcmepp.models.img_encoder import EncoderClipImageFull, ProbEncoderClipImageFull
from pcmepp.models.img_encoder import EncoderImageAggrBUTD, ProbEncoderImageAggrBUTD
from pcmepp.models.img_encoder import EncoderImageFull, ProbEncoderImageFull
from pcmepp.models.txt_encoder import EncoderTextClip, ProbEncoderTextClip
from pcmepp.models.txt_encoder import EncoderTextBert, ProbEncoderTextBert
from pcmepp.modules.resnet import ResnetFeatureExtractor


def get_text_encoder(embed_size, backbone_source=None, is_probabilistic_model=True, no_txtnorm=False, **kwargs):
    if backbone_source and 'clip' in backbone_source:
        if is_probabilistic_model:
            return ProbEncoderTextClip(embed_size, clip_model=backbone_source.split('_')[1], no_txtnorm=no_txtnorm, **kwargs)
        else:
            return EncoderTextClip(embed_size, clip_model=backbone_source.split('_')[1], no_txtnorm=no_txtnorm, **kwargs)
    else:
        if is_probabilistic_model:
            return ProbEncoderTextBert(embed_size, no_txtnorm=no_txtnorm)
        else:
            return EncoderTextBert(embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(img_dim, embed_size, size_augment=0.2, is_probabilistic_model=True,
                      precomp_enc_type='basic', backbone_source=None, backbone_path=None,
                      no_imgnorm=False, **kwargs):
    if backbone_source and 'clip' in backbone_source:
        if precomp_enc_type != 'backbone':
            raise ValueError('CLIP backbone only supports `precomp_enc_type` = backbone')
        if is_probabilistic_model:
            return ProbEncoderClipImageFull(
                clip_model=backbone_source.split('_')[1],
                img_dim=img_dim,
                embed_size=embed_size,
                size_augment=size_augment,
                no_imgnorm=no_imgnorm, **kwargs)
        else:
            return EncoderClipImageFull(
                clip_model=backbone_source.split('_')[1],
                img_dim=img_dim,
                embed_size=embed_size,
                size_augment=size_augment,
                no_imgnorm=no_imgnorm,
                **kwargs)

    raise NotImplementedError('Other options are not implemented or verified yet')

    if precomp_enc_type == 'basic':
        if backbone_source:
            raise ValueError('if you use `precomp_enc_type` = basic, then it does not support backbone')

        if is_probabilistic_model:
            return ProbEncoderImageAggrBUTD(img_dim, embed_size, precomp_enc_type, no_imgnorm, **kwargs)
        else:
            return EncoderImageAggrBUTD(img_dim, embed_size, precomp_enc_type, no_imgnorm, **kwargs)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        if is_probabilistic_model:
            return ProbEncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm)
        else:
            return EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))
