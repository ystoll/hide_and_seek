#!/bin/bash


# Evaluating the best level 4 model on 1000 episodes of maps of level 4
python evaluate.py models/best_models/level4/DQN_TS_map_random_freq_4_step_level_4_lr_0.001_expl_0.01_explfrac_0.5_gamma_0.8_LongViewObservation_view_size_5/500000.zip \
           --eval_map random \
           --nb_episodes 1000 \
           --levels 3 5

# Evaluating the best level 4 model on 1000 episodes of statement map.
# python evaluate.py models/best_models/level4/DQN_TS_map_random_freq_4_step_level_4_lr_0.001_expl_0.01_explfrac_0.5_gamma_0.8_LongViewObservation_view_size_5/500000.zip \
#            --eval_map statement \
#            --nb_episodes 1000