#!/bin/bash

cd ..
nohup python inference.py --device 0 --work-dir 'inference_results/' \
                          --checkpoint 'training_results/train_dev/best_61_model.pt' \
                          --experiment 'test_dev' --window_size 20 --window_stride 20 > inference.log 2>&1 &