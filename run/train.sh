#!/bin/bash

cd ..

nohup python main_E2E.py --experiment 'formal_train_v8.13_1_cpu--20' --window_size 20 --window_stride 20 --base-lr 0.00005 > train_cpu.log 2>&1 &