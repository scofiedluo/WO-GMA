#!/bin/bash

cd ..
nohup python inference.py --device 0 --work-dir 'inference_results/' --checkpoint 'training_results/test_train4/best_45_model.pt' --experiment 'test_train4' --window_size 20 --window_stride 20 --OAR_thresh 0.5 --OAR_precent 0.05 > inference.log 2>&1 &