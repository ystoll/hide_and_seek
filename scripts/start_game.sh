#!/bin/bash

# See the AI play.

# Mono level agent

# Start the best level 4 model we got on random map, playing 50 episodes on levels 4
python start_game.py models/best_models/level4/DQN_TS_map_random_freq_4_step_level_4_lr_0.001_expl_0.01_explfrac_0.5_gamma_0.8_LongViewObservation_view_size_5/500000.zip\
                        --game_map random \
                        --fps 5 \
                        --nb_episodes 20 \
                        --levels 4


# Start the best level 4 model we got on random map, playing 20 episodes on levels 3 and 4
# python start_game.py models/best_models/level4/DQN_TS_map_random_freq_4_step_level_4_lr_0.001_expl_0.01_explfrac_0.5_gamma_0.8_LongViewObservation_view_size_5/500000.zip\
#                         --game_map random \
#                         --fps 5 \
#                         --nb_episodes 20 \
#                         --levels 3 4


# Start the best level 4 model we got on statement map, playing 50 episodes
# python start_game.py models/best_models/level4/DQN_TS_map_random_freq_4_step_level_4_lr_0.001_expl_0.01_explfrac_0.5_gamma_0.8_LongViewObservation_view_size_5/500000.zip \
#                         --game_map statement \
#                         --fps 5 \
#                         --nb_episodes 50

# Multi levels agent
# Start the best level 3/4/5 model we got on random map, playing 20 episodes on levels 3 4 5
# python start_game.py models/best_models/levels345/DQN_TS_map_random_freq_4_step_level_3_4_5_lr_0.001_expl_0.01_explfrac_0.7_gamma_0.8_LongViewObservation_view_size_5/3_4_5__250000.zip \
#                         --game_map random \
#                         --fps 5 \
#                         --nb_episodes 20 \
#                         --levels 3 4 5