#!/bin/bash

cd ..

nohup python main.py --experiment 'train_dev' --device 2 \
                     --window_size 20 --window_stride 20 --base-lr 0.00005 > train_dev.log 2>&1 &
