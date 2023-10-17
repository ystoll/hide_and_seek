#!/bin/bash

# This script allow to reproduce the results we presented in the README.md file.


# Mono level training

# On level 4
python3.10 train.py --model_folder best_model_level_4 \
        --observation LongViewObservation \
        --timesteps 500000 \
        --train_map random \
        --levels 4 \
        --view_size 5 \
        --learning_rate 0.001 \
        --learning_starts 50000 \
        --exploration 0.01 \
        --gamma 0.80 \
        --exploration_fraction 0.5 \
        --log_interval 4 \
        --train_freq 4 \
        --unit step


# On level 3, 4 and 5
# python train.py --model_folder best_model_level_3_4_5 \
#         --observation LongViewObservation \
#         --timesteps 250000 \
#         --train_map random \
#         --levels 3 4 5 \
#         --view_size 5 \
#         --learning_rate 0.001 \
#         --learning_starts 50000 \
#         --exploration 0.01 \
#         --gamma 0.80 \
#         --exploration_fraction 0.7 \
#         --log_interval 4 \
#         --train_freq 4 \
#         --unit step
