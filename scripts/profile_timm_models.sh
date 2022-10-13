#!/bin/bash

profile_timm() {
    python3 scripts/profile_timm.py --model $1 --size $2
    python3 scripts/profile_timm.py --model $1 --use-onnx --size $2
}

profile_timm beit_base_patch16_224 224
profile_timm botnet26t_256 256
profile_timm gernet_s 224
profile_timm cait_xxs24_224 224
profile_timm coat_tiny 224
profile_timm convit_tiny 224
profile_timm convmixer_768_32 224
profile_timm convnext_tiny 288
profile_timm crossvit_15_240 240
profile_timm cspresnet50 256
profile_timm deit_tiny_patch16_224 224
profile_timm densenet121 224
profile_timm dla34 224
profile_timm dpn68 224
profile_timm edgenext_xx_small 288
profile_timm efficientformer_l1 224
profile_timm gcvit_xxtiny 224
profile_timm ghostnet_050 224
profile_timm gluon_resnet18_v1b 224
profile_timm gluon_xception65 299
profile_timm hardcorenas_a 224
profile_timm hrnet_w18_small 224
profile_timm inception_resnet_v2 299
profile_timm inception_v3 299
profile_timm inception_v4 299
profile_timm levit_128s 224
profile_timm maxvit_tiny_224 224
profile_timm coatnet_0_224 224
profile_timm mixer_s32_224 224
profile_timm mobilenetv3_small_050 224
profile_timm mobilevit_xs 256
profile_timm mvitv2_tiny 224
profile_timm nasnetalarge 331
profile_timm nest_tiny 224
profile_timm dm_nfnet_f0 256
profile_timm pit_ti_224 224
profile_timm pnasnet5large 331
profile_timm poolformer_s12 224
profile_timm pvt_v2_b0 224
profile_timm regnetx_040 224
profile_timm res2net50_26w_4s 224
profile_timm resnest14d 224
profile_timm resnet10t 224
profile_timm resnetv2_50 224
profile_timm rexnet_100 224
profile_timm selecsls42 224
profile_timm legacy_senet154 224
profile_timm sequencer2d_s 224
profile_timm skresnet18 224
profile_timm swin_tiny_patch4_window7_224 224
profile_timm swinv2_tiny_window8_256 256
profile_timm swinv2_cr_tiny_224 224
profile_timm tnt_s_patch16_224 224
profile_timm tresnet_m 224
profile_timm twins_pcpvt_small 224
profile_timm vgg11 224
profile_timm visformer_tiny 224
profile_timm volo_d1_224 224
profile_timm vovnet39a 224
profile_timm xception 299
profile_timm xception41 299
profile_timm xcit_nano_12_p16_224 224