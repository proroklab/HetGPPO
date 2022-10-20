from pathlib import Path
from typing import Set, Union

import numpy as np
import torch
from matplotlib import pyplot as plt

from evaluate.evaluate_model import clamp_with_norm
from utils import InjectMode, EvaluationUtils


def plot_trajectory(
    checkpoint_path: Union[str, Path],
    n_episodes: int = 1,
    agents_to_inject: Set = None,
    inject_mode: InjectMode = None,
    noise_delta: float = None,
):

    config, trainer, env = EvaluationUtils.get_config_trainer_and_env_from_checkpoint(
        checkpoint_path
    )

    inject = agents_to_inject is not None and len(agents_to_inject) > 0

    rewards, _, obs, actions = EvaluationUtils.rollout_episodes(
        n_episodes=n_episodes,
        render=False,
        get_obs=True,
        get_actions=True,
        trainer=trainer,
        env=env,
        inject=inject,
        inject_mode=inject_mode,
        agents_to_inject=agents_to_inject,
        noise_delta=noise_delta,
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # Get just one episode
    best_episode_index = rewards.index(max(rewards))
    obs = obs[best_episode_index]
    actions = actions[best_episode_index]

    obs = torch.tensor(np.array(obs))
    pos_x = 0
    pos_y = 1
    vel_x = 2
    vel_y = 3

    obs_to_plot = pos_x

    actions = torch.tensor(np.array(actions)).view(
        -1, len(env.env.agents), env.action_space[0].shape[0]
    )[:, 0]
    actions = clamp_with_norm(actions, 1)

    ax.plot(np.arange(len(obs)), obs[..., pos_x], label="Pos x")
    ax.plot(np.arange(len(obs)), actions[:, 0], label="Vel x ref")
    ax.plot(np.arange(len(obs)), obs[..., vel_x], label="Vel x")

    ax.plot(np.arange(len(obs)), obs[..., pos_y], label="Pos y")
    ax.plot(np.arange(len(obs)), actions[:, 1], label="Vel y ref")
    ax.plot(np.arange(len(obs)), obs[..., vel_y], label="Vel y")

    # ax.quiver(
    #     obs_0[:, 0],
    #     obs_1[:, 0],
    #     actions[:, 0, 0],
    #     actions[:, 0, 1],
    #     scale=50,
    #     angles="xy",
    #     color="r",
    #     label="Action rollout",
    # )
    # ax.legend(loc="upper right")

    plt.xlabel("Timestep", fontsize=14)
    plt.legend()
    (
        model_title,
        model_name,
        arch_title,
        arch_name,
    ) = EvaluationUtils.get_model_name(config)

    inject_title, inject_name = (
        EvaluationUtils.get_inject_name(
            agents_to_inject=agents_to_inject,
            noise_delta=noise_delta,
            inject_mode=inject_mode,
        )
        if inject
        else ("", "")
    )

    fig.suptitle(model_title + " " + arch_title, fontsize=16)
    plt.show()
    # save_folder = PathUtils.result_dir / f"{n_floors} floor/trajectory graphs"
    # name = f"trajectory_{model_name}_{arch_name}" + (
    #     "_" + inject_name if inject else ""
    # )
    # plt.savefig(str(save_folder / f"{name}.pdf"))


if __name__ == "__main__":

    checkpoint_path = "/Users/Matteo/Downloads/MultiPPOTrainer_het_goal_de3e6_00000_0_2022-10-05_18-19-28/checkpoint_000500/checkpoint-500"

    plot_trajectory(
        checkpoint_path,
        n_episodes=1,
        agents_to_inject={0},
        inject_mode=InjectMode.OBS_NOISE,
        noise_delta=0.01,
    )
