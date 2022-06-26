#!/bin/bash

cd ..

nohup python main_E2E.py --experiment 'crossval_BN_WO_st_cas_topk_v2_mil2_fold4--20' \
                         --window_size 20 --window_stride 20 --base-lr 0.00005 > crossval_BN_WO_st_cas_topk_v2_mil2_fold4.log 2>&1 &