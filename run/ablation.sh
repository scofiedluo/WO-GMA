#!/bin/bash

cd ..

# nohup python main_E2E.py --vertices_fusion_out_dim 1080 --experiment 'ablation_BN_WO_st_cas_topk_v2_wolocal--20' --window_size 20 --window_stride 20 --base-lr 0.00005 > ablation_BN_WO_st_cas_topk_v2_wolocal.log 2>&1 &

# nohup python main_E2E.py --batch-size 24 --vertices_fusion_out_dim 1080 --experiment 'ablation_BN_WO_st_cas_topk_v2_batch24_wolocal--20' --window_size 20 --window_stride 20 --base-lr 0.00005 > ablation_BN_WO_st_cas_topk_v2_batch24_wolocal.log 2>&1 &

# nohup python main_E2E.py --experiment 'ablation_BN_WO_st_cas_topk_v2_batch24_wolong--20' --window_size 20 --window_stride 20 --base-lr 0.00005 > ablation_BN_WO_st_cas_topk_v2_batch24_wolong.log 2>&1 &

# without persudo gen branch
# nohup python main_E2E.py --experiment 'ablation_BN_WO_st_cas_topk_v2_mil2_wopseudo--20' --window_size 20 --window_stride 20 --base-lr 0.00005 > ablation_BN_WO_st_cas_topk_v2_mil2_wopseudo.log 2>&1 &

# without local
# nohup python main_E2E.py --vertices_fusion_out_dim 1080 --experiment 'ablation_BN_WO_st_cas_topk_v2_mil2_wolocal--20' --window_size 20 --window_stride 20 --base-lr 0.00005 > ablation_BN_WO_st_cas_topk_v2_mil2_wolocal.log 2>&1 &

# without longrang
nohup python main_E2E.py --experiment 'ablation_BN_WO_st_cas_topk_v2_mil2_wolong--20' --window_size 20 --window_stride 20 --base-lr 0.00005 > ablation_BN_WO_st_cas_topk_v2_mil2_wolong.log 2>&1 &