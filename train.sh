#!/bin/bash

# python train.py --dataset regdb --trial 1 --gpu 0 --w_hc 0.5 --per_img 8 --lr 1e-2 --epochs 1000
python train.py --dataset sysu --trial 1 --gpu 0 --w_hc 0.5 --per_img 8 --lr 1e-4 --epochs 1000 --resume "sysu_id_bn_relu_lr_1.0e-03_dim_512_whc_0.5_thd_0_pimg_8_ds_l2_md_all_RGA_att4_epoch_160.t"

# python train.py --dataset regdb --trial 1 --gpu 0 --w_hc 0.5 --per_img 8 --lr 1e-3 --resume "regdb_id_bn_relu_lr_1.0e-03_dim_512_whc_0.5_thd_0_pimg_8_ds_l2_md_all_trial_1_RGA_att34_best.t"
