#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from pathlib import Path
from typing import Union, Set

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import flatten
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian, TorchCategorical

from rllib_differentiable_comms.multi_action_dist import (
    TorchBetaMulti,
)
from utils import EvaluationUtils, InjectMode


def get_action_distributions_gippo(
    obs_space_0, obs_space_1, obs_index_0, obs_index_1, env, model
):
    action_distributions = np.empty(
        (obs_space_0.shape[0], obs_space_1.shape[0], len(env.env.agents)),
        dtype=object,
    )
    k = 0
    for i in obs_space_0:
        f = 0
        for l in obs_space_1:
            observation = SampleBatch()
            observation[SampleBatch.OBS] = torch.zeros(
                (1, len(env.env.agents), env.env.observation_space[0].shape[0])
            )
            observation[SampleBatch.OBS][:, 0, obs_index_0] = i
            observation[SampleBatch.OBS][:, 1, obs_index_1] = l
            observation[SampleBatch.OBS] = flatten(
                observation[SampleBatch.OBS], framework="torch"
            )
            # print(f"\nObs: {observation[SampleBatch.OBS].shape}\n")
            logits = model(observation)[0].detach()
            # print(f"Logits shape: {logits.shape}\n Logits: {logits}")
            j = 0
            for agent_index in [0, 1]:
                if agent_index == 1:
                    temp = k
                    k = f
                    f = temp

                agent_action_space = env.action_space[agent_index]
                if isinstance(agent_action_space, gym.spaces.box.Box):
                    assert len(agent_action_space.shape) == 1
                    inputs = logits[:, j : (j + 2 * agent_action_space.shape[0])]
                    j += 2 * agent_action_space.shape[0]
                    if model.use_beta:
                        action_distributions[k, f, agent_index] = TorchBetaMulti(
                            inputs,
                            model,
                            agent_action_space.low,
                            agent_action_space.high,
                        )
                    else:
                        action_distributions[k, f, agent_index] = TorchDiagGaussian(
                            inputs, model
                        )
                elif isinstance(agent_action_space, gym.spaces.discrete.Discrete):
                    inputs = logits[:, j : (j + agent_action_space.n)]
                    # print(f"Inputs: {inputs}")
                    j += agent_action_space.n
                    action_distributions[k, f, agent_index] = TorchCategorical(
                        inputs, model
                    )
                if agent_index == 1:
                    temp = k
                    k = f
                    f = temp
            f += 1
        print(
            f"Computing action distributions: {int((k / (obs_space_0.shape[0] - 1)) * 100)}%",
            end="\r",
        )
        k += 1
    return action_distributions


def get_actions_mean_and_variance(
    action_distributions, obs_space_0, obs_space_1, env, n_obs_samples
):
    agent_means = torch.zeros(
        (
            obs_space_0.shape[0],
            obs_space_1.shape[0],
            len(env.env.agents),
            env.env.action_space[0].shape[0],
        )
    )
    agent_means_vectors = torch.zeros(
        (
            obs_space_0.shape[0],
            obs_space_1.shape[0],
            len(env.env.agents),
            env.env.action_space[0].shape[0],
        )
    )
    agent_variances = torch.zeros(
        (
            obs_space_0.shape[0],
            obs_space_1.shape[0],
            len(env.env.agents),
            env.env.action_space[0].shape[0],
        )
    )

    for agent_index in range(len(env.env.agents)):
        agent_dists = action_distributions[:, :, agent_index]
        if isinstance(agent_dists[0, 0], TorchBetaMulti):
            for i in range(n_obs_samples):
                for j in range(n_obs_samples):
                    dist = agent_dists[i, j]
                    agent_means[i, j, agent_index] = dist._squash(dist.dist.mean)

                    if agent_index == 1:
                        agent_means_vectors[j, i, agent_index] = dist._squash(
                            dist.dist.mean
                        )
                    else:
                        agent_means_vectors[i, j, agent_index] = dist._squash(
                            dist.dist.mean
                        )

        elif isinstance(agent_dists[0][0], TorchCategorical):
            classes = torch.tensor(
                [-env.env.agents[0].u_range, 0.0, env.env.agents[0].u_range]
            )
            for i in range(n_obs_samples):
                for j in range(n_obs_samples):
                    dist = agent_dists[i, j]
                    agent_means[i, j, agent_index] = torch.sum(
                        dist.dist.probs * classes
                    )
                    if agent_index == 1:
                        agent_means_vectors[j, i, agent_index] = torch.sum(
                            dist.dist.probs * classes
                        )
                    else:
                        agent_means_vectors[i, j, agent_index] = torch.sum(
                            dist.dist.probs * classes
                        )

        elif isinstance(agent_dists[0][0], TorchDiagGaussian):
            for i in range(n_obs_samples):
                for j in range(n_obs_samples):
                    dist = agent_dists[i, j]
                    agent_means[i, j, agent_index] = dist.dist.mean
                    agent_variances[i, j, agent_index] = dist.dist.variance
                    if agent_index == 1:
                        agent_means_vectors[j, i, agent_index] = dist.dist.mean
                    else:
                        agent_means_vectors[i, j, agent_index] = dist.dist.mean
        else:
            assert False
    return agent_means, agent_variances, agent_means_vectors


def plot_vector_field(
    ax,
    is_first,
    agent_means_vectors,
    n_obs_samples,
    env,
    obs_space_0_mesh,
    obs_space_1_mesh,
    obs_range_1,
    obs_range_2,
    plot_trajectory,
    trainer,
    model_title,
    env_title,
    title,
    action_index,
    obs_index_0: int,
    obs_index_1: int,
    agents_to_inject: Set = None,
    inject_mode: InjectMode = None,
    noise_delta: float = None,
):
    vectors = agent_means_vectors[..., action_index].view(
        n_obs_samples * n_obs_samples, len(env.env.agents)
    )

    ax.grid()
    ax.quiver(
        obs_space_0_mesh,
        obs_space_1_mesh,
        vectors[:, 0],
        vectors[:, 1],
        scale=30,
        angles="xy",
        alpha=0.7,
        label="Action policy" if is_first else "",
    )

    inject = agents_to_inject is not None and len(agents_to_inject) > 0

    if plot_trajectory:
        rewards, _, obs, actions = EvaluationUtils.rollout_episodes(
            n_episodes=1,
            render=False,
            get_obs=True,
            get_actions=True,
            trainer=trainer,
            env=env,
            inject=inject,
            agents_to_inject=agents_to_inject,
            inject_mode=inject_mode,
            noise_delta=noise_delta,
        )

        assert obs_index_0 == obs_index_1

        obs = torch.tensor(np.array(obs[0]))
        obs = obs[..., obs_index_0]
        actions = (
            torch.tensor(np.array(actions))
            .view(-1, len(env.env.agents), env.action_space[0].shape[0])[
                ..., action_index
            ]
            .view(-1, len(env.env.agents))
        )
        ax.quiver(
            obs[:, 0],
            obs[:, 1],
            actions[:, 0],
            actions[:, 1],
            scale=20,
            angles="xy",
            color="#e41a1c",
            label="Action rollout" if is_first else "",
        )

    inject_title, inject_name = (
        EvaluationUtils.get_inject_name(
            agents_to_inject=agents_to_inject,
            noise_delta=noise_delta,
            inject_mode=inject_mode,
        )
        if inject
        else ("", "")
    )
    # fig.suptitle(model_title + " " + env_title, fontsize=16)
    ax.set_title(title, y=-0.25)
    ax.set_xlim([-obs_range_1, obs_range_1])
    ax.set_ylim([-obs_range_2, obs_range_2])
    ax.set_xlabel("Agent 1: $\mathbf{v}_1$")
    if is_first:
        ax.legend(fancybox=True, shadow=True)
    # ax.set_ylabel("Agent 1 velocity")
    # ax.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.14),
    #     fancybox=True,
    #     shadow=True,
    #     ncol=2,
    # )

    # save_folder = PathUtils.result_dir / env_title / model_title
    # name = f"circulation" + ("_" + inject_name if inject else "")
    # plt.savefig(str(save_folder / f"{name}.pdf"), bbox_inches="tight", pad_inches=0)
    # plt.show()


def plot_policy_2d(
    agent_means,
    agent_variances,
    env,
    obs_space_0_mesh,
    obs_space_1_mesh,
    model_title,
    env_title,
    action_index,
):

    fig = plt.figure(figsize=(17, 8))
    fig.suptitle(model_title + " " + env_title)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    colors = plt.rcParams["axes.prop_cycle"]()
    agent_index = 0
    for col in (ax1, ax2):
        c = next(colors)["color"]

        surf = col.plot_surface(
            obs_space_0_mesh,
            obs_space_1_mesh,
            agent_means[:, :, agent_index, action_index],
            cmap="viridis",
            rstride=10,
            cstride=10,
            edgecolor="none",
        )

        col.plot_surface(
            obs_space_0_mesh,
            obs_space_1_mesh,
            agent_means[:, :, agent_index, action_index]
            + agent_variances[:, :, agent_index, action_index],
            cmap="viridis",
            rstride=10,
            cstride=10,
            edgecolor="none",
            alpha=0.5,
        )
        col.plot_surface(
            obs_space_0_mesh,
            obs_space_1_mesh,
            agent_means[:, :, agent_index, action_index]
            - agent_variances[:, :, agent_index, action_index],
            cmap="viridis",
            rstride=10,
            cstride=10,
            edgecolor="none",
            alpha=0.5,
        )
        # col.contour(
        #     obs_space_0_mesh,
        #     obs_space_1_mesh,
        #     agent_means,
        #     100
        # )
        col.set_title(f"Agent {agent_index} mean")
        col.set_xlabel("Agent obs")
        col.set_ylabel("Neighbour obs")
        col.set_zlabel("Action")
        agent_index += 1

    cb = fig.colorbar(
        surf,
        ax=[ax1, ax2],
        shrink=0.5,
        extend="both",
        location="bottom",
        label=f"Action range: [{int(-env.env.agents[0].u_range)}, {int(env.env.agents[0].u_range)}]",
    )
    plt.show()


def plot_policy(
    checkpoint_path: Union[str, Path],
    ax,
    title,
    is_first: bool,
    obs_index_0: int,
    obs_index_1: int,
    obs_range_0: float,
    obs_range_1: float,
    n_obs_samples: int,
    plot_vectors: bool,
    action_index: int,
    plot_trajectory=False,
    agents_to_inject: Set = None,
    inject_mode: InjectMode = None,
    noise_delta: float = None,
):

    config, trainer, env = EvaluationUtils.get_config_trainer_and_env_from_checkpoint(
        checkpoint_path
    )

    policy = trainer.get_policy()
    model = policy.model

    is_gippo = config["model"]["custom_model_config"]["share_observations"]

    low_bound = -env.env.agents[0].u_range
    high_bound = -low_bound

    (
        model_title,
        model_name,
        env_title,
        env_name,
    ) = EvaluationUtils.get_model_name(config)

    if not is_gippo:
        x_size = 600
        obs_space = torch.linspace(0, 200, x_size, dtype=torch.float32)
        action_distributions = np.empty(
            (obs_space.shape[0], len(env.agents)), dtype=object
        )
        k = 0
        for i in obs_space:
            observation = SampleBatch()
            observation[SampleBatch.OBS] = {
                "agents": torch.full((len(env.agents), 1), i).to("cpu")
            }
            logits = model(observation)[0].detach()
            # print(f"Logits shape: {logits.shape}\n Logits: {logits}")
            j = 0
            for agent_index in range(len(env.agents)):
                agent_action_space = env.action_space[agent_index]
                if isinstance(agent_action_space, gym.spaces.box.Box):
                    assert len(agent_action_space.shape) == 1
                    inputs = logits[:, j : (j + 2 * agent_action_space.shape[0])]
                    j += 2 * agent_action_space.shape[0]
                    if model.use_beta:
                        action_distributions[k, agent_index] = TorchBetaMulti(
                            inputs,
                            model,
                            agent_action_space.low,
                            agent_action_space.high,
                        )
                    else:
                        action_distributions[k, agent_index] = TorchDiagGaussian(
                            inputs, model
                        )
                elif isinstance(agent_action_space, gym.spaces.discrete.Discrete):
                    inputs = logits[:, j : (j + agent_action_space.n)]
                    # print(f"Inputs: {inputs}")
                    j += agent_action_space.n
                    action_distributions[k, agent_index] = TorchCategorical(
                        inputs, model
                    )

            k += 1

        fig, axs = plt.subplots(
            nrows=config["env_config"]["n_floors"], ncols=2, figsize=(17, 8)
        )
        if n_floors == 1:
            axs = np.array([axs])
        fig.suptitle(model_title + " " + arch_title)
        colors = plt.rcParams["axes.prop_cycle"]()
        agent_index = 0
        for row in axs:
            for col in row:
                c = next(colors)["color"]
                # print(f"Action_distributions shape: {action_distributions.shape}")
                agent_dists = action_distributions[:, agent_index]

                if isinstance(agent_dists[0], TorchBetaMulti):
                    agent_means = torch.tensor(
                        [dist._squash(dist.dist.mean) for dist in agent_dists],
                        dtype=torch.float32,
                    )
                    agent_devs = torch.tensor(
                        [
                            dist._squash(torch.sqrt(dist.dist.variance)) - dist.low
                            for dist in agent_dists
                        ],
                        dtype=torch.float32,
                    )
                elif isinstance(agent_dists[0], TorchCategorical):
                    classes = torch.tensor([low_bound, 0.0, high_bound])
                    agent_means = torch.tensor(
                        [torch.sum(dist.dist.probs * classes) for dist in agent_dists],
                        dtype=torch.float32,
                    )
                    agent_devs = torch.tensor(
                        [
                            torch.sqrt(
                                torch.sum(
                                    (
                                        (classes - torch.sum(dist.dist.probs * classes))
                                        ** 2
                                    )
                                    * dist.dist.probs
                                )
                            )
                            for dist in agent_dists
                        ],
                        dtype=torch.float32,
                    )
                elif isinstance(agent_dists[0], TorchDiagGaussian):
                    agent_means = torch.tensor(
                        [dist.dist.mean for dist in agent_dists],
                        dtype=torch.float32,
                    )
                    agent_devs = torch.tensor(
                        [torch.sqrt(dist.dist.variance) for dist in agent_dists],
                        dtype=torch.float32,
                    )
                else:
                    assert False

                col.fill_between(
                    obs_space,
                    agent_means + agent_devs,
                    agent_means - agent_devs,
                    color=c,
                    alpha=0.6,
                )
                col.plot(obs_space, agent_means + agent_devs, color=c, alpha=0.2)
                col.plot(obs_space, agent_means - agent_devs, color=c, alpha=0.2)
                col.plot(
                    obs_space,
                    agent_means,
                    color=c,
                    label=f"Agent {agent_index} mean and std dev",
                )
                col.legend(loc="upper right")
                agent_index += 1

        # plt.setp(axs[-1, :], xlabel="Observation")
        # plt.setp(
        #     axs[:, 0],
        #     ylabel=f"Action range: [{int(low_bound)}, {int(high_bound)}]"
        #     if is_continuous
        #     else f"Available actions: {int(low_bound)}, 0, {int(high_bound)}",
        # )

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel("Observation", fontsize=14)
        plt.ylabel(
            f"Action (range {int(low_bound)}, {int(high_bound)})"
            if is_continuous
            else f"Action ({int(low_bound)}, 0, {int(high_bound)})",
            fontsize=14,
        )
        plt.show()

    else:

        obs_space_0 = torch.linspace(
            -obs_range_0, obs_range_0, n_obs_samples, dtype=torch.float32
        )
        obs_space_1 = torch.linspace(
            -obs_range_1, obs_range_1, n_obs_samples, dtype=torch.float32
        )
        obs_space_0_mesh, obs_space_1_mesh = torch.meshgrid(
            obs_space_0, obs_space_1, indexing="ij"
        )

        # Ob_space_0.shape[0] x Ob_space_1.shape[0] x n_agents
        action_distributions = get_action_distributions_gippo(
            obs_space_0, obs_space_1, obs_index_0, obs_index_1, env, model
        )

        (
            agent_means,
            agent_variances,
            agent_means_vectors,
        ) = get_actions_mean_and_variance(
            action_distributions, obs_space_0, obs_space_1, env, n_obs_samples
        )

        if plot_vectors:
            plot_vector_field(
                ax,
                is_first,
                agent_means_vectors,
                n_obs_samples,
                env,
                obs_space_0_mesh,
                obs_space_1_mesh,
                obs_range_0,
                obs_range_1,
                plot_trajectory,
                trainer,
                model_title,
                env_title,
                title,
                action_index,
                obs_index_0,
                obs_index_1,
                agents_to_inject,
                inject_mode,
                noise_delta,
            )
        else:
            plot_policy_2d(
                agent_means,
                agent_variances,
                env,
                obs_space_0_mesh,
                obs_space_1_mesh,
                model_title,
                env_title,
                action_index,
            )


def plot_het_test():

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{libertine}\n\\usepackage[libertine]{newtxmath}",
        "font.family": "Linux Libertine",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 15,
        "font.size": 13,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 13,
        "legend.title_fontsize": 7,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    plt.rcParams.update(tex_fonts)

    fig, axs = plt.subplots(1, 4, figsize=(16, 5.1))

    # HetGIPPO
    checkpoint_path = "/Users/Matteo/Downloads/het_test/hetgippo/MultiPPOTrainer_het_test_ecc9e_00000_0_2022-09-12_14-43-46/checkpoint_000093/checkpoint-93"
    plot_policy(
        checkpoint_path,
        ax=axs[0],
        title="\\textbf{(a) Heterogeneous}",
        is_first=True,
        obs_index_0=2,
        obs_index_1=2,
        obs_range_0=0.5,
        obs_range_1=0.5,
        n_obs_samples=15,
        plot_vectors=True,
        action_index=0,
        plot_trajectory=True,
        agents_to_inject={0, 1},
        inject_mode=InjectMode.OBS_NOISE,
        noise_delta=0.0,
    )
    # Gippo
    checkpoint_path = "/Users/Matteo/Downloads/het_test/gippo/MultiPPOTrainer_het_test_18f2c_00000_0_2022-09-12_14-45-00/checkpoint_000116/checkpoint-116"
    plot_policy(
        checkpoint_path,
        ax=axs[1],
        title="\\textbf{(b) Homogeneous}",
        is_first=False,
        obs_index_0=2,
        obs_index_1=2,
        obs_range_0=0.5,
        obs_range_1=0.5,
        n_obs_samples=15,
        plot_vectors=True,
        action_index=0,
        plot_trajectory=True,
        agents_to_inject={0, 1},
        inject_mode=InjectMode.OBS_NOISE,
        noise_delta=0.0,
    )
    # Hetgippo injected
    checkpoint_path = "/Users/Matteo/Downloads/het_test/hetgippo/MultiPPOTrainer_het_test_ecc9e_00000_0_2022-09-12_14-43-46/checkpoint_000093/checkpoint-93"
    plot_policy(
        checkpoint_path,
        ax=axs[2],
        title="\\textbf{(c) Heterogeneous with noise}",
        is_first=False,
        obs_index_0=2,
        obs_index_1=2,
        obs_range_0=0.5,
        obs_range_1=0.5,
        n_obs_samples=15,
        plot_vectors=True,
        action_index=0,
        plot_trajectory=True,
        agents_to_inject={0, 1},
        inject_mode=InjectMode.OBS_NOISE,
        noise_delta=0.3,
    )

    # Gippo injected
    checkpoint_path = "/Users/Matteo/Downloads/het_test/gippo/MultiPPOTrainer_het_test_18f2c_00000_0_2022-09-12_14-45-00/checkpoint_000116/checkpoint-116"
    plot_policy(
        checkpoint_path,
        ax=axs[3],
        title="\\textbf{(d) Homogenous with noise}",
        is_first=False,
        obs_index_0=2,
        obs_index_1=2,
        obs_range_0=0.5,
        obs_range_1=0.5,
        n_obs_samples=15,
        plot_vectors=True,
        action_index=0,
        plot_trajectory=True,
        agents_to_inject={0, 1},
        inject_mode=InjectMode.OBS_NOISE,
        noise_delta=0.3,
    )

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel("Agent 2: $\mathbf{v}_2$")

    # fig.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.08),
    #     fancybox=True,
    #     shadow=True,
    #     ncol=2,
    # )
    plt.tight_layout()

    # Separator
    # line = plt.Line2D(
    #     [0.514, 0.514], [0.1, 0.95], transform=fig.transFigure, color="black", alpha=0.6
    # )
    # fig.add_artist(line)

    # set column spanning title
    # the first two arguments to figtext are x and y coordinates in the figure system (0 to 1)
    # plt.figtext(0.3, 1, "Normal", va="center", ha="center", size=15)
    # plt.figtext(0.65, 1, "Noise", va="center", ha="center", size=15)

    plt.savefig(str(f"circulation_ensemble.pdf"), bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":

    plot_het_test()

    # Het test

    # HetGIPPO
    # checkpoint_path = "/Users/Matteo/Downloads/het_test/hetgippo/MultiPPOTrainer_het_test_ecc9e_00000_0_2022-09-12_14-43-46/checkpoint_000093/checkpoint-93"
    # Gippo
    # checkpoint_path = "/Users/Matteo/Downloads/het_test/gippo/MultiPPOTrainer_het_test_18f2c_00000_0_2022-09-12_14-45-00/checkpoint_000116/checkpoint-116"

    # Give way

    # # Gippo
    # checkpoint_path = "/Users/Matteo/Downloads/give_way/gippo/MultiPPOTrainer_give_way_0605e_00000_0_2022-09-12_19-30-49/checkpoint_000216/checkpoint-216"
    # # HetGippo
    # checkpoint_path = "/Users/Matteo/Downloads/give_way/hetgippo/MultiPPOTrainer_give_way_fecd2_00000_0_2022-09-12_19-30-37/checkpoint_000300/checkpoint-300"
