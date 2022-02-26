#!/bin/bash

cd ..
nohup python inference.py --work-dir 'inference_results/' --checkpoint 'training_results/formal_train_v8.13_1--20/best_model.pt' --experiment 'inference_formal_train_v8.13_1--20_val' --window_size 20 --window_stride 20 --OAR_thresh 0.5 --OAR_precent 0.05 > inference.log 2>&1 &