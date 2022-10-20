from pathlib import Path
from typing import Set, Union

import numpy as np
import tikzplotlib
import torch
from matplotlib import pyplot as plt

from evaluate.evaluate_model import compute_action_corridor
from utils import InjectMode, EvaluationUtils


def get_distance(
    checkpoint_path: Union[str, Path],
    het: bool,
    n_episodes: int,
    agents_to_inject: Set = None,
    inject_mode: InjectMode = None,
    noise_delta: float = None,
):
    def update_config(config):
        config["env_config"]["scenario_config"]["obs_noise"] = 0
        return config

    config, trainer, env = EvaluationUtils.get_config_trainer_and_env_from_checkpoint(
        checkpoint_path, config_update_fn=update_config
    )

    inject = agents_to_inject is not None and len(agents_to_inject) > 0

    def action_callback(observation):
        model = torch.load(
            f"/Users/Matteo/Downloads/{'het_' if het else ''}a_range_1_u_range_0_5_[3_2_0_002]_0_05_dt_0_1_friction_0_dt_delay_option_0.pt"
        )
        model.eval()

        action = compute_action_corridor(
            pos0_x=observation[0][0],
            pos0_y=observation[0][1],
            vel0_x=observation[0][2],
            vel0_y=observation[0][3],
            pos1_x=observation[1][0],
            pos1_y=observation[1][1],
            vel1_x=observation[1][2],
            vel1_y=observation[1][3],
            model=model,
            u_range=0.5,
        )
        return action

    rewards, _, obs, actions = EvaluationUtils.rollout_episodes(
        n_episodes=n_episodes,
        render=False,
        get_obs=True,
        get_actions=True,
        trainer=None,
        action_callback=action_callback,
        env=env,
        inject=inject,
        inject_mode=inject_mode,
        agents_to_inject=agents_to_inject,
        noise_delta=noise_delta,
    )

    obs = torch.tensor(np.array(obs))
    obs = obs.view(obs.shape[0], -1, env.env.n_agents, 6)
    pos_x = 0
    x_positions = obs[..., pos_x]

    x_positions_agent_0 = x_positions[..., 0]
    x_positions_agent_1 = x_positions[..., 1]

    def success_from_x_position(x1, x2):
        distance_from_goal1 = torch.minimum(x1 - 2, torch.zeros_like(x1)).abs()
        distance_from_goal2 = torch.maximum(2 + x2, torch.zeros_like(x2)).abs()
        distance = distance_from_goal1 + distance_from_goal2  # 0 to 8
        distance /= -8
        distance += 1
        return distance

    distance = success_from_x_position(x_positions_agent_0, x_positions_agent_1)

    # Cut after success
    argmax = torch.argmax(distance, dim=1)
    max_argmax = int(torch.max(argmax).item())
    distance = distance[..., :max_argmax]

    return distance


def plot_distances(het_distance, homo_distance):
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{libertine}\n\\usepackage[libertine]{newtxmath}",
        "font.family": "Linux Libertine",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 19,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 16,
        "legend.title_fontsize": 7,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }

    plt.rcParams.update(tex_fonts)

    fig, ax = plt.subplots(figsize=(5, 5))

    CB_color_cycle = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]

    for i, episode_obs in enumerate(het_distance):
        ax.plot(
            np.linspace(0, len(episode_obs) * 0.05, len(episode_obs)),
            episode_obs,
            label="HetGPPO" if i == 0 else None,
            color=CB_color_cycle[0],
        )
    for i, episode_obs in enumerate(homo_distance):
        ax.plot(
            np.linspace(0, len(episode_obs) * 0.05, len(episode_obs)),
            episode_obs,
            label="GPPO" if i == 0 else None,
            color=CB_color_cycle[1],
        )

    ax.grid()
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Task completion")
    ax.legend()

    tikzplotlib.save(
        f"trial.tex",
        textsize=18,
    )
    plt.savefig(f"trial.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":

    checkpoint_path = "/Users/Matteo/Downloads/MultiPPOTrainer_give_way_deploy_72a9b_00000_0_2022-10-18_11-13-01/checkpoint_001294/checkpoint-1294"
    n_episodes = 10

    het_distance = get_distance(checkpoint_path, n_episodes=n_episodes, het=True)
    homo_distance = get_distance(checkpoint_path, n_episodes=n_episodes, het=False)
    plot_distances(het_distance=het_distance, homo_distance=homo_distance)
