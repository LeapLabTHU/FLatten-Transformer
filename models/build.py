# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .flatten_swin import FLattenSwinTransformer
from .flatten_pvt import flatten_pvt_tiny, flatten_pvt_small, flatten_pvt_medium, flatten_pvt_large
from .flatten_pvt_v2 import flatten_pvt_v2_b0, flatten_pvt_v2_b1, flatten_pvt_v2_b2, flatten_pvt_v2_b3, \
    flatten_pvt_v2_b4, flatten_pvt_v2_b5
from .flatten_cswin import FLatten_CSWin_64_24181_tiny_224, FLatten_CSWin_64_24322_small_224, \
    FLatten_CSWin_96_36292_base_224, FLatten_CSWin_96_36292_base_384


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type in ['flatten_swin']:
        model = eval('FLattenSwinTransformer' + '(img_size=config.DATA.IMG_SIZE,'
                                                'patch_size=config.MODEL.SWIN.PATCH_SIZE,'
                                                'in_chans=config.MODEL.SWIN.IN_CHANS,'
                                                'num_classes=config.MODEL.NUM_CLASSES,'
                                                'embed_dim=config.MODEL.SWIN.EMBED_DIM,'
                                                'depths=config.MODEL.SWIN.DEPTHS,'
                                                'num_heads=config.MODEL.SWIN.NUM_HEADS,'
                                                'window_size=config.MODEL.SWIN.WINDOW_SIZE,'
                                                'mlp_ratio=config.MODEL.SWIN.MLP_RATIO,'
                                                'qkv_bias=config.MODEL.SWIN.QKV_BIAS,'
                                                'qk_scale=config.MODEL.SWIN.QK_SCALE,'
                                                'drop_rate=config.MODEL.DROP_RATE,'
                                                'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                                'ape=config.MODEL.SWIN.APE,'
                                                'patch_norm=config.MODEL.SWIN.PATCH_NORM,'
                                                'use_checkpoint=config.TRAIN.USE_CHECKPOINT,'
                                                'focusing_factor=config.MODEL.LA.FOCUSING_FACTOR,'
                                                'kernel_size=config.MODEL.LA.KERNEL_SIZE,'
                                                'attn_type=config.MODEL.LA.ATTN_TYPE)')

    elif model_type in ['FLatten_CSWin_64_24181_tiny_224', 'FLatten_CSWin_64_24322_small_224',
                        'FLatten_CSWin_96_36292_base_224', 'FLatten_CSWin_96_36292_base_384']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'in_chans=config.MODEL.SWIN.IN_CHANS,'
                                  'num_classes=config.MODEL.NUM_CLASSES,'
                                  'drop_rate=config.MODEL.DROP_RATE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'focusing_factor=config.MODEL.LA.FOCUSING_FACTOR,'
                                  'kernel_size=config.MODEL.LA.KERNEL_SIZE,'
                                  'attn_type=config.MODEL.LA.ATTN_TYPE,'
                                  'la_split_size=config.MODEL.LA.CSWIN_LA_SPLIT_SIZE)')

    elif model_type in ['flatten_pvt_tiny', 'flatten_pvt_small', 'flatten_pvt_medium', 'flatten_pvt_large',
                        'flatten_pvt_v2_b0', 'flatten_pvt_v2_b1', 'flatten_pvt_v2_b2',
                        'flatten_pvt_v2_b3', 'flatten_pvt_v2_b4', 'flatten_pvt_v2_b5']:
        model = eval(model_type + '(drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'focusing_factor=config.MODEL.LA.FOCUSING_FACTOR,'
                                  'kernel_size=config.MODEL.LA.KERNEL_SIZE,'
                                  'attn_type=config.MODEL.LA.ATTN_TYPE,'
                                  'la_sr_ratios=str(config.MODEL.LA.PVT_LA_SR_RATIOS))')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
