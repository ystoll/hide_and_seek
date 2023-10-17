"""
This file is used to train a RL agent to play the game of Hide and Seek using
the stable-baselines3 library.
It can be run from the command line with different parameters in order to
try different models.
"""

import argparse
import os
import pickle
import re

from stable_baselines3 import DQN
from typing import List

from envrl.hide_and_seek_env import HideAndSeekEnv
from envrl.observation_type import (
    BasicObservation,
    ImmediateSuroundingsObservation,
    LongViewObservation,
)


def train(
        model_folder,
        observation: str = "LongViewObservation",
        timesteps: int = 500_000,
        train_map: str = "statement",
        levels: List[int] = [-1],
        train_freq: int = 4,
        unit: str = "step",
        view_size: int = 5,
        learning_rate: float = 0.001,
        learning_starts: int = 50000,
        exploration: float = 0.05,
        exploration_fraction: float = 0.1,
        gamma: float = 0.99,
        log_interval: int = 4,
        progress_bar: bool = True
) -> None:
    """Train a agent using a Deep Q Network (DQN) algorithm from the stable baseline 3 library.
    The agent is trained "timesteps" time steps per level.

    Parameters
    ----------
    model_folder : "str"
        The trained agent is saved under models/model_folder
    observation : str, optional
        Observation class provided to the agent (see envrl/observation_type.py), by default "LongViewObservation"
    timesteps : int, optional
        Number of time steps to train the agent on, by default 500_000
    train_map : str, optional
        Map used to train each agent, by default "statement"
    levels : List[int], optional
        List of levels to train the agent on. Each agent is trained "timesteps" time steps per level. By default [4]
    train_freq : int, optional
        Update the model every train_freq "unit". Unit is set in units argument. by default 4
    unit : str, optional
        Either "step" or "episode". Unit of train_freq, by default "step"
    view_size : int, optional
        Value of the view_size for the LongViewObservation observation class (see envrl/observation_type.py), by default 5
    learning_rate : float, optional
        Learning rate to train the agent on, by default 0.001
    learning_starts : int, optional
        Set the warmup phase i.e the agent does not learn before learning_starts time steps are done, by default 50000
    exploration : float, optional
        Final value of random action probability to train the agent on, by default 0.05
    exploration_fraction : float, optional
        The fraction of the entire  training period over which the exploration rate is reduced, by default 0.1
    gamma : float, optional
        The discount factors to train the agent on. Determine how much importance is given to future rewards when calculating policy. By default 0.99
    log_interval : int, optional
        The number of episodes before logging, by default 4
    progress_bar : bool, optional
        Wheither or not to display a progress bar during training, by default True

    Raises
    ------
    ValueError
        Raised if train_map is not random and levels are set
    """
    if (train_map != "random" and levels != -1):
        raise ValueError("Level must be set only if game_map='random'")
    if train_map != "random":
        levels = [-1]  # set dummy levels if not used.

    selected_model = "DQN"  # fixed, the only one useful in our case
    observation_type = {
        "BasicObservation": BasicObservation(),
        "ImmediateSuroundingsObservation": ImmediateSuroundingsObservation(),
        "LongViewObservation": LongViewObservation(view_size),
    }[observation]

    # model_name = f"{selected_model}_{timer_id}_{str(observation_type)}"
    if train_map == "random":
        levels_spanned = "_".join(str(elem) for elem in levels)
        model_name = f"{selected_model}_TS_map_{train_map}_freq_{train_freq}_{unit}_level_{levels_spanned}_lr_{learning_rate}_expl_{exploration}_explfrac_{exploration_fraction}_gamma_{gamma}_{str(observation_type)}"
    else:
        model_name = f"{selected_model}_TS_map_{train_map}_freq_{train_freq}_{unit}_lr_{learning_rate}_expl_{exploration}_explfrac_{exploration_fraction}_gamma_{gamma}_{str(observation_type)}"
    model_name = re.sub(r"[(=]", "_", model_name)  # replace (, ) and = signs by underscores to avoid escaping them
    model_name = model_name.replace(")", "")
    models_dir = f"models/{model_folder}/{model_name}"
    log_dir = f"logs/{model_folder}"

    # Create models and logs directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Write information about the model in a file
    with open(f"{models_dir}/model_info.txt", "w") as f:
        f.write(f"Model name: {model_name}\n")
        f.write(f"Observation type: {str(observation_type)}\n")
        f.write(f"Map trained on: {train_map}\n")
        f.write(f"Number of timesteps: {timesteps}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Learning starts: {learning_starts}\n")
        f.write(f"Exploration: {exploration}\n")
        f.write(f"Log interval: {log_interval}\n")
        f.write(f"Progress bar: {progress_bar}\n")

    # Save the observation type class in a file
    with open(f"{models_dir}/observation_type.pkl", "wb") as f:
        pickle.dump(observation_type, f, pickle.HIGHEST_PROTOCOL)

    # create environment in "rgb_array" mode to not have a display
    env = HideAndSeekEnv(render_mode="rgb_array",
                         observation_type=observation_type,
                         map_name=train_map,
                         level=levels[0])
    env.reset()

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        batch_size=256,
        tensorboard_log=log_dir,
        learning_rate=learning_rate,
        learning_starts=learning_starts,
        exploration_final_eps=exploration,
        train_freq=tuple((train_freq, unit)),
        exploration_fraction=exploration_fraction,
        gamma=gamma

    )
    # Set the seed of the pseudo-random generators (python, numpy, pytorch, gym, action_space)
    model.set_random_seed(seed=2015)

    print(f"Training {model_name} with parameters:")
    print(f"- Observation type: {str(observation_type)}")
    print(f"- Map trained on: {train_map}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Learning starts: {learning_starts}")
    print(f"- Exploration: {exploration}")

# Important note: we tried different learning loops. It turns out that for our needs, at least for timesteps = 5.10^5 it was as efficient to train the agent in one go,
# than gradually as was done in the original package. Most probably this is due to the default buffer size of the DQN model being 10^6.
    if train_map == "random":
        # If random map, the agent is trained over the different levels.
        levels_learned = ""
        for level in levels:
            env.level = level  # we set the current level of the env.
            env.reset()
            print(f"Training Level: {level}")
            try:
                levels_learned = levels_learned + str(level) + "_"
                model.learn(
                    total_timesteps=timesteps,
                    reset_num_timesteps=False,
                    tb_log_name=model_name,
                    progress_bar=True,
                    log_interval=log_interval
                )
                model.save(f"{models_dir}/{levels_learned}_{timesteps}")

            except Exception as err:  # Very nasty ! But here for rapidity we just ignore failed training.
                log_failed = open(f"models/{model_folder}/log_train_failed.txt", "a")
                log_failed.write(f"model: {model_name}. Error: {err}")
                log_failed.close()
                pass

        env.close()
    else:
        try:
            model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=False,
                tb_log_name=model_name,
                progress_bar=True,
                log_interval=log_interval
            )
            model.save(f"{models_dir}/{timesteps}")

        except Exception as err:  # Very nasty ! But here for rapidity we just ignore failed training.
            log_failed = open(f"models/{model_folder}/log_train_failed.txt", "a")
            log_failed.write(f"model: {model_name}. Error: {err}")
            log_failed.close()
            pass

        env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_folder",
        type=str,
        default="default",
        help=("The trained agent will be saved in models/model_folder directory."),
    )
    parser.add_argument(
        "--observation",
        type=str,
        default="LongViewObservation",
        choices=[
            "BasicObservation",
            "ImmediateSuroundingsObservation",
            "LongViewObservation",
        ],
        help=(
            "BasicObservation, ImmediateSuroundingsObservation or LongViewObservation."
            + " Default: LongViewObservation. Observation type to use for training."
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help=("Number of timesteps to train in total. Default: 500 000."),
    )
    parser.add_argument(
        "--train_map",
        type=str,
        default="statement",
        help=(
            f"statement, custom, few_walls or random. Default: statement."
            + " Map to use for training."
        ),
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=-1,
        help=("Map levels on which the agent will be trained. Default: 4."),
    )
    parser.add_argument(
        "--train_freq",
        type=int,
        default=4,
        help=(
            "Update the model every train_freq unit (see --unit option)."
        ),
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="step",
        help=(
            "Unit, either 'step' or 'episode' to update the model on."
        ),
    )
    parser.add_argument(
        "--view_size",
        type=int,
        default=5,
        help=(
            "View size for LongViewObservation. Used if --observation is set to LongViewObservation."
            + " Ignored otherwise. Default is 5."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help=("Learning rate. Default: 0.001."),
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=50000,
        help=("Learning starts. Default: 50000."),
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=0.05,
        help=("Exploration. Default: 0.05.")
    )

    parser.add_argument(
        "--exploration_fraction",
        type=float,
        default=0.1,
        help=(
            "Exploration fraction during training. Default: 0.1"
        ),
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help=(
            "Gamma (discount factor) during training. Default: 0.99"
        ),
    )

    parser.add_argument(
        "--log_interval", type=int, default=4, help=("Log interval. Default: 4.")
    )
    parser.add_argument(
        "--progress_bar",
        action="store_true",
        default=True,
        help=("Display a progress bar during training."),
    )
    args = parser.parse_args()

    train(**vars(args))
