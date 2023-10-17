import argparse
import os
import pickle


from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from typing import List

from envrl.hide_and_seek_env import HideAndSeekEnv
from math import sqrt


def evaluate_agent(model_path,
                   eval_map: str = "statement",
                   nb_episodes: int = 1000,
                   levels: List[int] = [-1]) -> None:
    """Evaluate a trained agent n_episodes per level on eval_map.

    Parameters
    ----------
    model_path : str
        path of model to be evaluated (zip file),
    eval_map : str, optional
        Map on which the agent will be evaluated, by default "statement"
    nb_episodes : int, optional
        Number of episodes over which each level is evaluated, by default 1000
    levels : List[int], optional
        Levels on which the agent is evaluated. Each agent is evaluated nb_episodes episodes per level. By default [4]

    Returns
    -------
    total_mean_rewards_levels : float
        Average of the per level mean reward
    total_std_reward_levels : float
        Roots of the quadratic sum of the per level mean's standard deviation.
    mean_rewards_levels : List[float]
        List of the per level mean reward
    std_reward_levels : List[float]
        List of the per level means's standard deviation
    Raises
    ------
    ValueError
        Raised if eval_map is not "random" and levels are set.
    """
    levels_on = True
    if (eval_map != "random" and levels != -1):
        raise ValueError("Levels must be set only if game_map='random'")
    if eval_map != "random":
        levels_on = False

    # Get infos from model
    # The observation type is needed to load the environment
    model_directory = os.path.dirname(model_path)
    observation_type = None
    with open(os.path.join(model_directory, "observation_type.pkl"), "rb") as obs:
        observation_type = pickle.load(obs)

    mean_rewards_levels = []
    std_reward_levels = []

    if levels_on:
        # Evaluating the agent on the differents levels.
        for i_level, level in enumerate(levels):

            eval_env = Monitor(
                HideAndSeekEnv(render_mode="rgb_array",
                               observation_type=observation_type,
                               map_name=eval_map,
                               level=level)
            )
            eval_env.reset()

            model = DQN.load(model_path, env=eval_env)
            # Setting the seed of models.
            # Results are now reproducible across runs.
            model.set_random_seed(seed=2015)

            if i_level == 0:
                print(f"Evaluating model {model_path} on map {eval_map}")
                print(f"- Learning rate: {model.learning_rate}")
                print(f"- Learning starts: {model.learning_starts}")
                print(f"- Exploration final eps: {model.exploration_final_eps}")
                print(f"- Exploration decay: {model.exploration_fraction}")
                print(f"- Gamma: {model.gamma}")

            print(f"Evaluating {nb_episodes} episodes on level {level}.")

            mean_reward_level, std_reward_level = evaluate_policy(model,
                                                                  eval_env,
                                                                  n_eval_episodes=nb_episodes)

            mean_rewards_levels.append(mean_reward_level)
            std_reward_levels.append(std_reward_level)
            print(f"Level: {level} ---> mean_reward:{mean_reward_level:.2f} +/- {std_reward_level:.2f}")

        total_mean_rewards_levels = sum(mean_rewards_levels) / len(mean_rewards_levels)
        # We want to average out the different standard deviations, computed on each level.
        # Works only because we are evaluating each level on the same number of episodes.
        total_std_reward_levels = sqrt(sum(std * std for std in std_reward_levels)) / len(std_reward_levels)

        print("#" * 95)
        print("#### ALL LEVELS:")
        print(f"#### mean_reward: {total_mean_rewards_levels:.2f} +/- {total_std_reward_levels:.2f}")
        print("#" * 95)

        return total_mean_rewards_levels, total_std_reward_levels, mean_rewards_levels, std_reward_levels

    else:

        eval_env = Monitor(
            HideAndSeekEnv(render_mode="rgb_array",
                           observation_type=observation_type,
                           map_name=eval_map,
                           level=-1)
        )
        eval_env.reset()

        model = DQN.load(model_path, env=eval_env)
        # Setting the seed of models.
        # Results are now reproducible across runs.
        model.set_random_seed(seed=2015)

        print(f"Evaluating model {model_path} on map {eval_map}")
        print(f"- Learning rate: {model.learning_rate}")
        print(f"- Learning starts: {model.learning_starts}")
        print(f"- Exploration final eps: {model.exploration_final_eps}")
        print(f"- Exploration decay: {model.exploration_fraction}")
        print(f"- Gamma: {model.gamma}")

        mean_reward, std_reward = evaluate_policy(model,
                                                  eval_env,
                                                  n_eval_episodes=nb_episodes)

        print("#" * 95)
        print(f"#### mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print("#" * 95)

        return mean_reward, std_reward, None, None




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help=("Path to the model to be evaluated (zip file).")
    )
    parser.add_argument(
        "--eval_map",
        type=str,
        default="statement",
        choices=["statement", "custom", "few_walls", "random"],
        help=(
            f"Map on which the agent will be evaluated. Default: statement."
        ),
    )
    parser.add_argument(
        "--nb_episodes",
        type=int,
        default=1000,
        help=("Number of episodes over which each level is evaluated. Default: 1000."),
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=-1,
        help=("Levels on which the agent is evaluated. Default: [-1]."),
    )

    args = parser.parse_args()

    evaluate_agent(**vars(args))
