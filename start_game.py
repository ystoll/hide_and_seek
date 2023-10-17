import argparse
import os
import pickle

from stable_baselines3 import DQN
from typing import List

from envrl.hide_and_seek_env import HideAndSeekEnv

def start_game(model: str,
               game_map: str = "statement",
               fps: int = 5,
               nb_episodes: int = 20,
               levels: List[int] = [-1]
               ) -> None:
    """Starts a hide and seek game using the model "model",
    playing "nb_episodes" episodes per level on map "game_map".

    Parameters
    ----------
    model : str
        Path to the model to load, this should be a "zip" file.
    game_map : str, optional
        Map to use to start the game, by default "statement"
    fps : int, optional
        Number of frame per second, by default 5
    nb_episodes : int, optional
        Number of episodes to be played per level, by default 20
    levels : _type_, by deault [-1]
        List of levels to be played during the game, nb_eisodes will be played on each level. By default [-1]

    Raises
    ------
    ValueError
        Raised if game_map is not "random" and levels are specified.
    """
    if (game_map != "random" and levels != -1):
        raise ValueError("Levels must be set only if game_map='random'")
    if game_map != "random":
        levels = [-1, -1, -1, -1, -1]  # set dummy levels if not used.
    # Get infos from model
    # The observation type is needed to load the environment
    model_directory = os.path.dirname(model)
    print(model_directory)
    observation_type = None
    with open(os.path.join(model_directory, "observation_type.pkl"), "rb") as obs:
        observation_type = pickle.load(obs)

    env = HideAndSeekEnv(
        render_mode="human",
        fps=fps,
        observation_type=observation_type,
        map_name=game_map,
        level=levels[0]
    )
    env.reset()

    model = DQN.load(model, env=env)

    # Run nb_episodes episodes per level.

    if all(1 <= x <= 5 for x in levels):
        for level in levels:
            for ep in range(nb_episodes):
                env.level = level
                print(f"Level: {level}. Episode: {ep}")
                obs, _ = env.reset()
                print("obs: ", obs)

                done = False

                while not done:
                    action, _ = model.predict(obs)
                    print("action: ", action)
                    obs, _, done, _, _ = env.step(action)
    else:
        level = -1
        for ep in range(nb_episodes):

            print(f"Episode: {ep}")
            obs, _ = env.reset()
            print("obs: ", obs)

            done = False

            while not done:
                action, _ = model.predict(obs)
                print("action: ", action)
                obs, _, done, _, _ = env.step(action)

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help=("Model to load. A saved .zip file from train.py.")
    )
    parser.add_argument(
        "--game_map",
        type=str,
        default="statement",
        help=(
            f"statement, few_walls, custom or random. Default: statement."
            + " Map to use for the game."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help=("Replay speed (frames per second). Default: 5."),
    )
    parser.add_argument(
        "--nb_episodes",
        type=int,
        default=20,
        help=("Number of episodes to play per level. Default: 20."),
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=-1,
        help=("Map levels on which the game will be played. Default: -1."),
    )
    args = parser.parse_args()

    start_game(**vars(args))
