#!/bin/bash

# Start a training on a grid of models with given parameters


# Training test (for testig purposes, quick but the agents are useless).
python train_grid.py --grid_name test \
                         --timesteps 500 \
                         --train_map random \
                         --levels 1 2 5 \
                         --train_freqs 4  \
                         --units step \
                         --view_sizes 5 7 \
                         --learning_rates 0.0001 \
                         --explorations 0.1 \
                         --gammas 0.99 \
                         --exploration_fractions 0.1


# python evaluate_models.py models/test \
#            --eval_map random \
#            --nb_episodes 100 \
#            --levels 1 1 1 1


# Training grid levels 3 4 5
# python train_grid.py --grid_name levels345 \
#                          --timesteps 250000 \
#                          --train_map random \
#                          --levels 3 4 5 \
#                          --train_freqs 4 \
#                          --units step \
#                          --view_sizes 5 7 \
#                          --learning_rates 0.001 0.0001 \
#                          --explorations  0.01 0.001\
#                          --gammas 0.80 0.90 0.99 \
#                          --exploration_fractions 0.3 0.5 0.7


# python evaluate_models.py models/levels345/ \
#            --eval_map random \
#            --nb_episodes 1000 \
#            --levels 3 4 5


# Training grid level 4
# python train_grid.py --grid_name level4 \
#                          --timesteps 500000 \
#                          --save_interval 5000 \
#                          --train_map random \
#                          --level 4 \
#                          --train_freqs 4 \
#                          --units step \
#                          --view_sizes 5 7 \
#                          --learning_rates 0.001 0.0001 \
#                          --explorations 0.1 0.05 0.01\
#                          --gammas 0.80 0.90 0.99 \
#                          --exploration_fractions 0.1 0.3 0.5

# python evaluate_models.py models/level4/ \
#            --eval_map random \
#            --nb_episodes 1000 \


