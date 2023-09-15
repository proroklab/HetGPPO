#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os
import pickle

from ray import tune

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import MultiCallbacks
from ray.tune.integration.wandb import WandbLoggerCallback

from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer
from utils import PathUtils, TrainingUtils

ON_MAC = False
save = PPOTrainer

train_batch_size = 60_000 if not ON_MAC else 200  # Jan 32768
num_workers = 1 if not ON_MAC else 0  # jan 4
num_envs_per_worker = 600 if not ON_MAC else 1  # Jan 32
rollout_fragment_length = (
    train_batch_size
    if ON_MAC
    else train_batch_size // (num_workers * num_envs_per_worker)
)
scenario_name = "sampling"
model_name = "MyFullyConnectedNetwork"
# model_name = "GPPO"


def train(
    share_observations,
    centralised_critic,
    restore,
    heterogeneous,
    max_episode_steps,
    use_mlp,
    aggr,
    topology_type,
    add_agent_index,
    continuous_actions,
    seed,
    notes,
    share_action_value,
):
    checkpoint_rel_path = ""
    checkpoint_path = PathUtils.scratch_dir / checkpoint_rel_path
    params_path = checkpoint_path.parent.parent / "params.pkl"

    if centralised_critic and not use_mlp:
        if share_observations:
            group_name = "GAPPO"
        else:
            group_name = "MAPPO"
    elif use_mlp:
        group_name = "CPPO"
    elif share_observations:
        group_name = "GPPO"
    else:
        group_name = "IPPO"

    group_name = f"{'Het' if heterogeneous else ''}{group_name}"

    if restore:
        with open(params_path, "rb") as f:
            config = pickle.load(f)

    trainer = MultiPPOTrainer
    trainer_name = "MultiPPOTrainer" if trainer is MultiPPOTrainer else "PPOTrainer"
    tune.run(
        trainer,
        name=group_name if model_name.startswith("GPPO") else model_name,
        callbacks=[
            WandbLoggerCallback(
                project=f"{scenario_name}{'_test' if ON_MAC else ''}",
                api_key_file=str(PathUtils.scratch_dir / "wandb_api_key_file"),
                group=group_name,
                notes=notes,
            )
        ],
        local_dir=str(PathUtils.scratch_dir / "ray_results" / scenario_name),
        stop={"training_iteration": 500},
        restore=str(checkpoint_path) if restore else None,
        config={
            "seed": seed,
            "framework": "torch",
            "env": scenario_name,
            "kl_coeff": 0,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,  # 0.3
            "vf_loss_coeff": 1,  # Jan 0.001
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,  # 0.01,
            "train_batch_size": train_batch_size,
            "rollout_fragment_length": rollout_fragment_length,
            "sgd_minibatch_size": 4096 if not ON_MAC else 100,  # jan 2048
            "num_sgd_iter": 45,  # Jan 30
            "num_gpus": 0.1 if not ON_MAC else 0,
            "num_gpus_per_worker": 0.1 if not ON_MAC else 0,
            "num_workers": num_workers,
            "num_envs_per_worker": num_envs_per_worker,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "grad_clip": 40,
            "batch_mode": "complete_episodes",
            "model": {
                "vf_share_layers": share_action_value,
                "custom_model": model_name,
                "custom_action_dist": (
                    "hom_multi_action" if trainer is MultiPPOTrainer else None
                ),
                "custom_model_config": {
                    "activation_fn": "tanh",
                    "share_observations": share_observations,
                    "gnn_type": "MatPosConv",
                    "centralised_critic": centralised_critic,
                    "heterogeneous": heterogeneous,
                    "use_beta": False,
                    "aggr": aggr,
                    "topology_type": topology_type,
                    "use_mlp": use_mlp,
                    "add_agent_index": add_agent_index,
                    "pos_start": 0,
                    "pos_dim": 2,
                    "vel_start": 2,
                    "vel_dim": 2,
                    "share_action_value": share_action_value,
                    "trainer": trainer_name,
                },
            },
            "env_config": {
                "device": "cuda" if not ON_MAC else "cpu",
                "num_envs": num_envs_per_worker,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_episode_steps,
                # Env specific
                "scenario_config": {
                    "n_agents": 3,
                    "shared_rew": False,
                    "n_gaussians": 3,
                },
            },
            "evaluation_interval": 30,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": False,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                # "explore": False,
                "env_config": {
                    "num_envs": 1,
                },
                "callbacks": MultiCallbacks(
                    [
                        TrainingUtils.RenderingCallbacks,
                        TrainingUtils.EvaluationCallbacks,
                    ]
                ),
            },
            "callbacks": MultiCallbacks(
                [
                    TrainingUtils.EvaluationCallbacks,
                ]
            ),
        }
        if not restore
        else config,
    )


if __name__ == "__main__":
    TrainingUtils.init_ray(scenario_name=scenario_name, local_mode=ON_MAC)
    for seed in [0,1,2]:
        train(
            seed=seed,
            restore=False,
            notes="",
            # Model important
            share_observations=False,
            heterogeneous=False,
            # Other model
            share_action_value=False,
            centralised_critic=False,
            use_mlp=False,
            add_agent_index=False,
            aggr="add",
            topology_type="full",
            # Env
            max_episode_steps=100,
            continuous_actions=True,
        )
