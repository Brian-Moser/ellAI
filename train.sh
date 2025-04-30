#!/bin/bash

python main.py --dataset_path datasets/leaves_healthy \
               --mode train \
               --model mobilenet_v2 \
               --use_finetuning \
               --output_path .