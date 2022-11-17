import os
import pickle

from ray import tune
from ray.rllib.algorithms.callbacks import MultiCallbacks, DefaultCallbacks
from ray.rllib.models import MODEL_DEFAULTS
from ray.tune.integration.wandb import WandbLoggerCallback

from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer
from utils import PathUtils, TrainingUtils

ON_MAC = False

train_batch_size = 60000 if not ON_MAC else 200  # Jan 32768
num_workers = 5 if not ON_MAC else 0  # jan 4
num_envs_per_worker = 40 if not ON_MAC else 1  # Jan 32
rollout_fragment_length = (
    train_batch_size
    if ON_MAC
    else train_batch_size // (num_workers * num_envs_per_worker)
)
scenario_name = "discovery"
model_name = "GIPPO"
n_targets = 7


class CurriculumReward(DefaultCallbacks):
    def on_train_result(self, trainer, result, **kwargs):
        def decrease_n_targets(env):
            env.scenario.n_targets -= 1

        # try:
        #     if (
        #         result["custom_metrics"]["agent 0/all_targets_covered_mean"]
        #         > n_targets / 5
        #     ):
        #         trainer.workers.foreach_worker(
        #             lambda ev: ev.foreach_env(lambda env: set_n_targets(env))
        #         )
        # except KeyError:
        #     pass

        if (result["training_iteration"] + 1) % 100 == 0:
            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(lambda env: decrease_n_targets(env))
            )
            trainer.evaluation_workers.foreach_worker(
                lambda ev: ev.foreach_env(lambda env: decrease_n_targets(env))
            )


def train(
    share_observations,
    centralised_critic,
    restore,
    heterogeneous,
    max_episode_steps,
    use_mlp,
    aggr,
    topology_type,
    comm_range,
    add_agent_index,
    continuous_actions,
    seed,
    notes,
):
    checkpoint_rel_path = "ray_results/joint/GIPPO/MultiPPOTrainer_joint_9b1c9_00000_0_2022-09-01_10-08-12/checkpoint_000099/checkpoint-99"
    checkpoint_path = PathUtils.scratch_dir / checkpoint_rel_path
    params_path = checkpoint_path.parent.parent / "params.pkl"

    fcnet_model_config = MODEL_DEFAULTS.copy()
    fcnet_model_config.update({"vf_share_layers": False})

    if centralised_critic and not use_mlp:
        if share_observations:
            group_name = "GAPPO"
        else:
            group_name = "MAPPO"
    elif use_mlp:
        group_name = "CPPO"
    elif share_observations:
        group_name = "GIPPO"
    else:
        group_name = "IPPO"

    group_name = f"{'Het' if heterogeneous else ''}{group_name}"

    if restore:
        with open(params_path, "rb") as f:
            config = pickle.load(f)

    tune.run(
        MultiPPOTrainer,
        name=group_name if model_name == "GIPPO" else model_name,
        callbacks=[
            WandbLoggerCallback(
                project=f"{scenario_name}{'_test' if ON_MAC else ''}",
                api_key_file=str(PathUtils.scratch_dir / "wandb_api_key_file"),
                group=group_name,
                notes=notes,
            )
        ],
        local_dir=str(PathUtils.scratch_dir / "ray_results" / scenario_name),
        stop={"training_iteration": 5000},
        restore=str(checkpoint_path) if restore else None,
        config={
            "seed": seed,
            "framework": "torch",
            "env": scenario_name,
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,  # 0.3
            "vf_loss_coeff": 1,  # Jan 0.001
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,  # 0.01,
            "train_batch_size": train_batch_size,
            "rollout_fragment_length": rollout_fragment_length,
            "sgd_minibatch_size": 4096 if not ON_MAC else 100,  # jan 2048
            "num_sgd_iter": 40,  # Jan 30
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": num_workers,
            "num_envs_per_worker": num_envs_per_worker,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "truncate_episodes",
            "model": {
                "custom_model": model_name,
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "activation_fn": "relu",
                    "share_observations": share_observations,
                    "gnn_type": "MatPosConv",
                    "centralised_critic": centralised_critic,
                    "heterogeneous": heterogeneous,
                    "use_beta": False,
                    "aggr": aggr,
                    "topology_type": topology_type,
                    "comm_radius": comm_range,
                    "use_mlp": use_mlp,
                    "add_agent_index": add_agent_index,
                    "pos_start": 0,
                    "pos_dim": 2,
                    "vel_start": 2,
                    "vel_dim": 2,
                    "share_action_value": True,
                }
                if model_name == "GIPPO"
                else fcnet_model_config,
            },
            "env_config": {
                "device": "cpu",
                "num_envs": num_envs_per_worker,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_episode_steps,
                # Env specific
                "scenario_config": {
                    "n_agents": 5,
                    "n_targets": n_targets,
                    "agents_per_target": 2,
                    "lidar_range": comm_range,
                    "covering_range": 0.25,
                    "agent_collision_penalty": 0,
                    "covering_rew_coeff": 1,
                    "targets_respawn": True,
                    "time_penalty": 0,
                    "shared_reward": False,
                },
            },
            "evaluation_interval": 1,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                },
                "callbacks": MultiCallbacks([TrainingUtils.RenderingCallbacks]),
            },
            "callbacks": MultiCallbacks(
                [TrainingUtils.EvaluationCallbacks, CurriculumReward]
            ),
        }
        if not restore
        else config,
    )


if __name__ == "__main__":
    TrainingUtils.init_ray(scenario_name=scenario_name, local_mode=ON_MAC)

    train(
        seed=2,
        restore=False,
        notes="curriculum every 100 -1 target",
        # Model important
        share_observations=True,
        heterogeneous=False,
        # Other model
        centralised_critic=False,
        use_mlp=False,
        add_agent_index=False,
        aggr="add",
        topology_type=None,
        comm_range=0.35,
        # Env
        max_episode_steps=300,
        continuous_actions=True,
    )
