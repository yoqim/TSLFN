#!/bin/bash

# python test.py --dataset regdb --trial 1 --gpu 1 --low-dim 512 --visualization False --resume 'regdb_id_bn_relu_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_l2_md_all_trial_1_best.t' --w_hc 0.5

# python test.py --dataset sysu --gpu 1 --low-dim 512 --visualization True --resume 'sysu_id_bn_relu_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_l2_md_all_best.t' --w_hc 0.5 --mode all --gall-mode single

python test.py --dataset regdb --trial 1 --gpu 1 --low-dim 512 --visualization False --resume 'regdb_id_bn_relu_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_l2_md_all_trial_1_RGA_att4_best.t' --w_hc 0.5