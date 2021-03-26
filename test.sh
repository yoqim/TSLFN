#!/bin/bash

###### regdb ###### 
# python test.py --dataset regdb --trial 1 --gpu 1 --vis --resume 'regdb_id_bn_relu_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_l2_md_all_trial_1_best.t'

# python test.py --dataset regdb --trial 1 --gpu 1 --resume 'regdb_id_bn_relu_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_l2_md_all_trial_1_RGA_att4_best.t'



###### sysu ###### 
# python test.py --dataset sysu --gpu 1 --resume 'sysu_lr_1.0e-02_md_all_sharenet3_mulcla4_graph_best.t' --mode all --gall-mode single 

python test.py --dataset sysu --gpu 1 --vis --resume 'sysu_lr_1.0e-02_md_all_sharenet3_mulcla4_graph_best.t' --mode all --gall-mode single 
