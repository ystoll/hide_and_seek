"""
Run multiple evaluation sessions of all trained agents.
"""
import argparse
import glob
import os

from typing import List
from evaluate import evaluate_agent

def evaluate_models(agents_folder: str,
                    eval_map: str = "statement",
                    nb_episodes: int = 1000,
                    levels: List[int] = [-1]) -> None:
    """Evaluate a batch of agents stored in the repository "agents_folder"

    Parameters
    ----------
    agents_folder : str
        Folder that contains the trained agents,
    eval_map : str, optional
        Map used for evaluation, by default "statement"
    nb_episodes : int, optional
        Number of episodes to play per level, by default 1000
    levels : List[int], optional
        Levels on which the agents are evaluated. Each agent is evaluated "nb_episodes" episodes per level. By default [-1]

    Returns
    -------
    results : List[str, tuple[float, float]]
        Results of the evaluation.
        Each element of the list contains the path of the model,
        and a tuple with the total mean reward and standard deviation
        evaluated on all levels for a given model ("model_path", (total_mean_rewards, total_std_rewards)).
        The element of the list are stored in descending order following total_mean_rewards.

    Raises
    ------
    ValueError
        Raised if eval_map is not random and levels are set
    """
    if (eval_map != "random" and levels != -1):
        raise ValueError("Levels must be set only if game_map='random'")
    if eval_map != "random":
        levels = [-1, -1, -1, -1, -1]  # set dummy levels if not used.

    # list all folders
    all_models = glob.glob(os.path.join(agents_folder, "*/"))

    results = []
    num_models = len(all_models)

    for i_model, model in enumerate(all_models):
        print(f"Evaluating model {i_model + 1}/{num_models}")
        # Load the last trained model (last timestep saved, so the last .zip file)
        zip_files_list = glob.glob(os.path.join(model, "*.zip"))
        zip_files_list.sort(key=os.path.getctime, reverse=True)

        assert len(zip_files_list) > 0, f"No model  found in the {model} folder."

        last_model = zip_files_list[0]
        mean_rewards, std_reward, _, _ = evaluate_agent(last_model, eval_map, nb_episodes=nb_episodes, levels=levels)
        results.append(tuple((last_model, mean_rewards, std_reward)))
        print("-------------------")

    results.sort(key=lambda elem: elem[1], reverse=True)  # sort the results by mean_rewards.

    # Print the Best model
    best_model = results[0][0]
    best_model_reward = results[0][1]
    best_model_std_reward = results[0][2]
    print("#" * 95)
    print("#### Best model:")
    print(f"#### model: {best_model}")
    print(f"#### mean_reward: {best_model_reward:.2f} +/- {best_model_std_reward:.2f}")
    print("#" * 95)
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "agents_folder", type=str, help=("Folder containing the trained agents.")
    )
    parser.add_argument(
        "--eval_map",
        type=str,
        default="statement",
        help=(
            f"statement, custom, few_walls or random. Default: statement."
            + " Map to use for evaluation."
        ),
    )

    parser.add_argument(
        "--nb_episodes",
        type=int,
        default=1000,
        help=(
            f"Number of episodes upon which each model is evaluated per level. default: 1000"
        ),
    )

    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=-1,
        help=("Levels on which the agent is evaluated. Default: -1."),
    )

    args = parser.parse_args()

    evaluate_models(**vars(args))
