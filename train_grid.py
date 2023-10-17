"""
Run multiple training sessions of predeterminated parameters.
"""

import argparse

from typing import List
from train import train


def train_grid(
    grid_name,
    timesteps: int = 500_000,
    train_map: str = "statement",
    levels: List[int] = [4],
    train_freqs: List[int] = [4],
    units: List[str] = ["step"],
    view_sizes: List[int] = [5, 7],
    learning_rates: List[float] = [0.01, 0.001, 0.0001],
    explorations: List[float] = [0.1, 0.05, 0.01],
    gammas: List[float] = [0.80, 0.90, 0.99],
    exploration_fractions: List[float] = [0.1, 0.3, 0.5],
) -> None:
    """Train agents over a grid of parameters set by train_freqs, units, view_sizes, learning_rates, explorations, gammas, and explorations_fractions.

    Parameters
    ----------
    grid_name : str,
        The results of the grid search are saved under models/grid_name.
    timesteps : int, optional
        Number of timesteps to train each agent on. By default 500_000
    train_map : str, optional
        Map used to train each agent, by default "statement"
    level : List[int], optional
        Level on which the agents are trained. Each agent is trained "timesteps" time steps per level.
        If only one level is set, the agent is trained during "timesteps" time steps in total. By default 4
    train_freqs : List[float], optional
      Update the model every train_freq "unit". Unit is set in units argument. by default 4
    units : List[str], optional
        Either "step" or "episode". Unit of the train_freqs, by default "step"
    view_sizes : List[int], optional
        View sizes for the LongViewObservation to train the agents on, by default [5, 7]
    learning_rates : List[float], optional
        Learning rates to train the agents on, by default [0.01, 0.001, 0.0001]
    explorations : List[float], optional
        Final value of random action probability to train the agents on, by default [0.1, 0.05, 0.01]
    gammas : List[float], optional
        The discount factors to train the agent on. Determine how much importance is given to future rewards when calculating policy. By default [0.80, 0.90, 0.99]
    exploration_fractions : List[float], optional
        The fractions of the entire  training period over which the exploration rate is reduced to train the agent on, by default [0.1, 0.3, 0.5]
    """
    # Note:
    # We drop BasicObservation and ImmediateSuroundingsObservation
    # as the results were not promising during the exploratory phase.

    # Basic grid search.
    # A more refined approach such as a Bayesian search should be considered in the future.
    nb_models_to_train = (
        len(units) * len(train_freqs) * len(view_sizes) * len(learning_rates) * len(explorations) * len(gammas) * len(exploration_fractions)
    )
    print(f"BEGINNING OF GRID SEARCH. TRAINING {nb_models_to_train} AGENTS.")
    nb_models = 0
    for unit in units:
        for train_freq in train_freqs:
            for view_size in view_sizes:
                for learning_rate in learning_rates:
                    for exploration in explorations:
                        for gamma in gammas:
                            for exploration_fraction in exploration_fractions:
                                train(
                                    model_folder=grid_name,
                                    observation="LongViewObservation",
                                    timesteps=timesteps,
                                    train_map=train_map,
                                    levels=levels,
                                    train_freq=train_freq,
                                    unit=unit,
                                    view_size=view_size,
                                    learning_rate=learning_rate,
                                    learning_starts=50000,
                                    exploration=exploration,
                                    exploration_fraction=exploration_fraction,
                                    gamma=gamma,
                                    log_interval=4,
                                    progress_bar=True,
                                )
                                nb_models = nb_models + 1
                                print(f"------------------- ({nb_models}/{nb_models_to_train} done)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid_name", type=str, default="None", help=(f"The trained agents will be saved in models/grid_name folder."),
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000, help=(f"Number of timesteps to use during training for each agent. Default: 500_000."),
    )
    parser.add_argument(
        "--train_map", type=str, default="random", help=("Map on which the grid will be trained. Default: random"),
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[4],
        help=("Map level on which the agents of the grid will be trained. Default: None."),
    )
    parser.add_argument(
        "--train_freqs",
        nargs="+",
        type=int,
        default=4,
        help=(f"List of training freqs to test during training. Default: [2, 4, 7]"),
    )
    parser.add_argument(
        "--view_sizes",
        nargs="+",
        type=int,
        default=[5, 7],
        help=(f"List of view_size to test during training. Default: [3, 5, 7, 9]"),
    )
    parser.add_argument(
        "--learning_rates",
        nargs="+",
        type=float,
        default=[0.01, 0.001, 0.0001],
        help=(f"List of learning rates to test during training. Default: [0.01, 0.001, 0.0001]"),
    )
    parser.add_argument(
        "--explorations",
        nargs="+",
        type=float,
        default=[0.1, 0.05, 0.01],
        help=(f"List of final values of random action probability to test during training. Default: [0.1, 0.05, 0.01]"),
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.80, 0.90, 0.99],
        help=(f"List of gamma paramaters (discount factors) to test during training. Default: [0.80, 0.90, 0.99]"),
    )
    parser.add_argument(
        "--exploration_fractions",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5],
        help=(f"List of the fractions of the entire training period over which the exploration rate is reduced to test during training. Default: [0.1, 0.3, 0.5]"),
    )

    parser.add_argument(
        "--units",
        nargs="+",
        type=str,
        choices=["step", "episode"],
        default="step",
        help=(f"List of units, 'step' or 'episode',  to test during training. Default: [step]")
    ),
    args = parser.parse_args()

    train_grid(**vars(args))
