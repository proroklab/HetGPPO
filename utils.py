#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import copy
import pickle
import platform
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, Callable
from typing import Union

import ray
import torch
import vmas
import wandb
from ray.rllib import RolloutWorker, BaseEnv, Policy, VectorEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from vmas import make_env
from vmas.simulator.environment import Environment

from evaluate.distance_metrics import *
from evaluate.evaluate_model import TorchDiagGaussian
from models.fcnet import MyFullyConnectedNetwork
from models.gppo import GPPO
from rllib_differentiable_comms.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)
from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer


class PathUtils:
    scratch_dir = (
        Path("/Users/Matteo/scratch/")
        if platform.system() == "Darwin"
        else Path("/local/scratch/mb2389/")
    )
    gppo_dir = Path(__file__).parent.resolve()
    result_dir = gppo_dir / "results"
    rollout_storage = result_dir / "rollout_storage"


class InjectMode(Enum):
    ACTION_NOISE = 1
    OBS_NOISE = 2

    def is_noise(self):
        if self is InjectMode.OBS_NOISE or self is InjectMode.ACTION_NOISE:
            return True
        return False

    def is_obs(self):
        if self is InjectMode.OBS_NOISE:
            return True
        return False

    def is_action(self):
        if self is InjectMode.ACTION_NOISE:
            return True
        return False


class TrainingUtils:
    @staticmethod
    def init_ray(scenario_name: str, local_mode: bool = False):
        if not ray.is_initialized():
            ray.init(
                _temp_dir=str(PathUtils.scratch_dir / "ray"),
                local_mode=local_mode,
            )
            print("Ray init!")
        register_env(scenario_name, lambda config: TrainingUtils.env_creator(config))
        ModelCatalog.register_custom_model("GPPO", GPPO)
        ModelCatalog.register_custom_model(
            "MyFullyConnectedNetwork", MyFullyConnectedNetwork
        )
        ModelCatalog.register_custom_action_dist(
            "hom_multi_action", TorchHomogeneousMultiActionDistribution
        )

    @staticmethod
    def env_creator(config: Dict):
        env = make_env(
            scenario_name=config["scenario_name"],
            num_envs=config["num_envs"],
            device=config["device"],
            continuous_actions=config["continuous_actions"],
            wrapper=vmas.Wrapper.RLLIB,
            max_steps=config["max_steps"],
            # Scenario specific
            **config["scenario_config"],
        )
        return env

    class EvaluationCallbacks(DefaultCallbacks):
        def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            episode: Episode,
            **kwargs,
        ):
            info = episode.last_info_for()
            for a_key in info.keys():
                for b_key in info[a_key]:
                    try:
                        episode.user_data[f"{a_key}/{b_key}"].append(info[a_key][b_key])
                    except KeyError:
                        episode.user_data[f"{a_key}/{b_key}"] = [info[a_key][b_key]]

        def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            **kwargs,
        ):
            info = episode.last_info_for()
            for a_key in info.keys():
                for b_key in info[a_key]:
                    metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                    episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()

    class RenderingCallbacks(DefaultCallbacks):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.frames = []

        def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: Episode,
            **kwargs,
        ) -> None:
            self.frames.append(base_env.vector_env.try_render_at(mode="rgb_array"))

        def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            **kwargs,
        ) -> None:
            vid = np.transpose(self.frames, (0, 3, 1, 2))
            episode.media["rendering"] = wandb.Video(
                vid, fps=1 / base_env.vector_env.env.world.dt, format="mp4"
            )
            self.frames = []

    class HeterogeneityMeasureCallbacks(DefaultCallbacks):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.policy = None
            self.all_obs = []

        def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: Episode,
            **kwargs,
        ) -> None:
            obs = episode.last_raw_obs_for()
            act = episode.last_action_for()
            info = episode.last_info_for()
            reward = episode.last_reward_for()
            for i, agent_obs in enumerate(obs):
                obs[i] = torch.tensor(obs[i]).unsqueeze(0)
            self.all_obs.append(obs)

        def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            **kwargs,
        ) -> None:
            self.env: Environment = base_env.vector_env.env
            self.n_agents = self.env.n_agents

            self.policy = policies["default_policy"]
            self.model = self.policy.model
            self.model_state_dict = self.model.state_dict()

            self.temp_model_i = copy.deepcopy(self.model)
            self.temp_model_j = copy.deepcopy(self.model)
            self.temp_model_i.eval()
            self.temp_model_j.eval()

            dists = torch.full(
                (
                    len(self.all_obs),
                    int((self.n_agents * (self.n_agents - 1)) / 2),
                    self.n_agents,
                    self.env.get_agent_action_size(self.env.agents[0]),
                ),
                -1.0,
                dtype=torch.float,
            )
            # num_obs,
            # number of unique pairs,
            # number of spots within an observation where I can evaluate the agents,
            # number of actions per agent

            all_measures = {
                "wasserstein": dists,
                "kl": dists.clone(),
                "kl_sym": dists.clone(),
                "hellinger": dists.clone(),
                "bhattacharyya": dists.clone(),
            }

            pair_index = 0
            to_check_count = 0
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if j <= i:
                        continue
                    # Line run for all pairs
                    for agent_index in range(self.n_agents):
                        self.temp_model_i.load_state_dict(self.model_state_dict)
                        self.temp_model_j.load_state_dict(self.model_state_dict)
                        for temp_layer_i, temp_layer_j, layer in zip(
                            self.temp_model_i.gnn.children(),
                            self.temp_model_j.gnn.children(),
                            self.model.gnn.children(),
                        ):
                            if len(list(temp_layer_j.children())) > 1:
                                self.load_agent_x_in_pos_y(
                                    temp_layer_i, layer, x=i, y=agent_index
                                )
                                self.load_agent_x_in_pos_y(
                                    temp_layer_j, layer, x=j, y=agent_index
                                )

                        for obs_index, obs in enumerate(self.all_obs):
                            if obs_index == 0 and agent_index == 1:
                                to_check = True
                                to_check_count += 1
                            else:
                                to_check = False
                            return_dict = self.compute_distance(
                                self.temp_model_i,
                                self.temp_model_j,
                                obs,
                                agent_index,
                                i,
                                j,
                                episode,
                                to_check,
                            )
                            for key, value in all_measures.items():
                                all_measures[key][
                                    obs_index, pair_index, agent_index
                                ] = return_dict[key]
                            assert to_check_count <= 1
                    pair_index += 1
            for key, value in all_measures.items():
                assert not (value < 0).any(), f"{key}_{value}"
                episode.custom_metrics[key] = value.mean().item()

        def load_agent_x_in_pos_y(self, temp_model, model, x, y):
            temp_model[y].load_state_dict(model[x].state_dict())
            return temp_model

        def compute_distance(
            self,
            temp_model_i,
            temp_model_j,
            obs,
            action_index,
            i,
            j,
            episode,
            to_check,
        ):

            input_dict = {"obs": obs}

            logits_i = temp_model_i(input_dict)[0].detach()[0]
            logits_j = temp_model_j(input_dict)[0].detach()[0]

            input_lens = [
                2 * self.env.get_agent_action_size(self.env.agents[0])
            ] * self.n_agents

            logits_i = logits_i.view(1, -1)
            logits_j = logits_j.view(1, -1)

            split_inputs_i = torch.split(logits_i, input_lens, dim=1)
            split_inputs_j = torch.split(logits_j, input_lens, dim=1)

            distr_i = TorchDiagGaussian(
                split_inputs_i[action_index], self.env.agents[0].u_range
            )
            distr_j = TorchDiagGaussian(
                split_inputs_j[action_index], self.env.agents[0].u_range
            )

            mean_i = distr_i.dist.mean
            mean_j = distr_j.dist.mean

            var_i = distr_i.dist.variance
            var_j = distr_i.dist.variance

            if to_check:
                self.checkoutput(distr_i, i, episode)
                self.checkoutput(distr_j, j, episode)

            wasserstein = []
            for k in range(self.env.get_agent_action_size(self.env.agents[0])):
                version1 = torch.tensor(
                    wasserstein_distance(
                        mean_i[..., k].numpy(),
                        var_i[..., k].unsqueeze(-1).numpy(),
                        mean_j[..., k].numpy(),
                        var_j[..., k].unsqueeze(-1).numpy(),
                    )
                )
                # version2 = torch.tensor(
                #     wasserstein_distance2(
                #         mean_i[..., k].numpy(),
                #         var_i[..., k].unsqueeze(-1).numpy(),
                #         mean_j[..., k].numpy(),
                #         var_j[..., k].unsqueeze(-1).numpy(),
                #     )
                # )
                # assert torch.isclose(
                #     version1, version2, atol=1e-3
                # ), f"{version1} != {version2}"
                wasserstein.append(version1)
            wasserstein = torch.stack(wasserstein)

            return_value = {
                "wasserstein": wasserstein,
            }

            for name, distance in zip(
                [
                    "kl",
                    "kl_sym",
                    "hellinger",
                    "bhattacharyya",
                ],
                [
                    kl_divergence,
                    kl_symmetric,
                    hellinger_distance,
                    bhattacharyya_distance,
                ],
            ):
                distances = []
                for k in range(self.env.get_agent_action_size(self.env.agents[0])):
                    distances.append(
                        torch.tensor(
                            distance(
                                mean_i[..., k].numpy(),
                                var_i[..., k].unsqueeze(-1).numpy(),
                                mean_j[..., k].numpy(),
                                var_j[..., k].unsqueeze(-1).numpy(),
                            )
                        )
                    )
                    assert (
                        distances[k] >= 0
                    ).all(), f"{name}, [{distances[k]} with mean_i {mean_i[..., k]} var_i {var_i[...,k]}, mean_j {mean_j[..., k]} var_j {var_j[...,k]}"
                return_value[name] = torch.stack(distances)

            return return_value

        def checkoutput(self, distr: TorchDiagGaussian, index, episode):
            for i in range(self.env.get_agent_action_size(self.env.agents[0])):
                episode.custom_metrics[
                    f"agent_{index}/sample_{'x' if i == 0 else 'y'}"
                ] = distr.sample()[..., i].item()


class EvaluationUtils:

    # Resilience injection utils
    @staticmethod
    def __inject_noise_in_action(
        agent_actions: tuple,
        agent_indices: Set[int],
        noise_delta: float,
        env: VectorEnv,
    ) -> Tuple:
        assert len(agent_indices) <= len(agent_actions)
        agent_actions_new = list(agent_actions)
        for agent_index in agent_indices:
            noise = np.random.uniform(
                -noise_delta, noise_delta, size=agent_actions_new[agent_index].shape
            )
            agent_actions_new[agent_index] += noise
            agent_actions_new[agent_index] = np.clip(
                agent_actions_new[agent_index],
                -env.env.agents[agent_index].u_range,
                env.env.agents[agent_index].u_range,
            )
        return tuple(agent_actions_new)

    @staticmethod
    def __inject_noise_in_observation(
        observations: tuple,
        agent_indices: Set[int],
        noise_delta: float,
    ) -> Tuple:
        assert len(agent_indices) <= len(observations)
        observations_new = list(observations)
        for agent_index in agent_indices:
            noise = np.random.uniform(
                -noise_delta, noise_delta, size=observations_new[agent_index].shape
            )
            observations_new[agent_index] += noise
        return tuple(observations_new)

    @staticmethod
    def get_inject_function(
        inject_mode: InjectMode,
        noise_delta: float,
        agents_to_inject: Set,
        env: VectorEnv,
    ):
        def inject_function(x):
            if inject_mode is InjectMode.ACTION_NOISE:
                return EvaluationUtils.__inject_noise_in_action(
                    x, agent_indices=agents_to_inject, noise_delta=noise_delta, env=env
                )
            elif inject_mode is InjectMode.OBS_NOISE:
                return EvaluationUtils.__inject_noise_in_observation(
                    x, noise_delta=noise_delta, agent_indices=agents_to_inject
                )
            else:
                assert False

        return inject_function

    @staticmethod
    def get_checkpoint_config(checkpoint_path: Union[str, Path]):
        params_path = Path(checkpoint_path).parent.parent / "params.pkl"
        with open(params_path, "rb") as f:
            config = pickle.load(f)
        return config

    @staticmethod
    def get_config_trainer_and_env_from_checkpoint(
        checkpoint_path: Union[str, Path],
        for_evaluation: bool = True,
        config_update_fn: Callable[[Dict], Dict] = None,
    ):
        config = EvaluationUtils.get_checkpoint_config(checkpoint_path)
        scenario_name = config["env"]
        TrainingUtils.init_ray(scenario_name=scenario_name)

        if for_evaluation:
            env_config = config["env_config"]
            env_config.update({"num_envs": 1})

            eval_config = {
                "in_evaluation": True,
                "num_workers": 0,
                "num_gpus": 0,
                "num_envs_per_worker": 1,
                "env_config": env_config,
                # "explore": False,
            }
            config.update(eval_config)

        if config_update_fn is not None:
            config = config_update_fn(config)

        print(f"\nConfig: {config}")

        trainer = MultiPPOTrainer(env=scenario_name, config=config)
        trainer.restore(str(checkpoint_path))
        env = TrainingUtils.env_creator(config["env_config"])
        env.seed(config["seed"])

        return config, trainer, env

    @staticmethod
    def rollout_episodes(
        n_episodes: int,
        render: bool,
        get_obs: bool,
        get_actions: bool,
        trainer: MultiPPOTrainer,
        action_callback,
        env: VectorEnv,
        inject: bool,
        inject_mode: InjectMode,
        agents_to_inject: Set,
        noise_delta: float,
    ):
        assert (trainer is None) != (action_callback is None)

        if trainer is not None:
            print(
                f"\nLoaded: {EvaluationUtils.get_model_name(trainer.config)[0]}, {EvaluationUtils.get_model_name(trainer.config)[2]}"
            )
        if inject:
            print(
                f"Injected: {EvaluationUtils.get_inject_name(inject_mode=inject_mode, agents_to_inject=agents_to_inject, noise_delta=noise_delta)[0]}"
            )
            inject_function = EvaluationUtils.get_inject_function(
                inject_mode,
                agents_to_inject=agents_to_inject,
                noise_delta=noise_delta,
                env=env,
            )

        # (
        #     rewards,
        #     best_gif,
        #     observations,
        #     actions,
        # ) = EvaluationUtils.__get_pickled_rollout(
        #     n_episodes,
        #     render,
        #     get_obs,
        #     get_actions,
        #     trainer,
        #     inject,
        #     inject_mode,
        #     agents_to_inject,
        #     noise_delta,
        # )
        #
        # if rewards is not None:
        #     print("Loaded from pickle!")
        #     return (
        #         rewards,
        #         best_gif if render else None,
        #         observations if get_obs else None,
        #         actions if get_actions else None,
        #     )

        best_reward = float("-inf")
        best_gif = None
        rewards = []
        observations = []
        actions = []
        for j in range(n_episodes):
            frame_list = []
            observations_this_episode = []
            actions_this_episode = []
            reward_sum = 0
            observation = env.vector_reset()[0]
            i = 0
            done = False
            if render:
                frame_list.append(env.try_render_at(mode="rgb_array"))
            while not done:
                i += 1
                if inject and inject_mode.is_obs():
                    observation = inject_function(observation)
                if get_obs:
                    observations_this_episode.append(observation)

                if trainer is not None:
                    action = trainer.compute_single_action(observation)
                else:
                    action = action_callback(observation)

                if inject and inject_mode.is_action():
                    action = inject_function(action)
                if get_actions:
                    actions_this_episode.append(action)
                obss, rews, ds, infos = env.vector_step([action])
                observation = obss[0]
                reward = rews[0]
                done = ds[0]
                info = infos[0]
                reward_sum += reward
                if render:
                    frame_list.append(env.try_render_at(mode="rgb_array"))
            print(f"Episode: {j + 1}, total reward: {reward_sum}")
            rewards.append(reward_sum)
            if reward_sum > best_reward:
                best_reward = reward_sum
                best_gif = frame_list.copy()
            if get_obs:
                observations.append(observations_this_episode)
            if get_actions:
                actions.append(actions_this_episode)
        print(
            f"Max reward: {np.max(rewards)}\nReward mean: {np.mean(rewards)}\nMin reward: {np.min(rewards)}"
        )

        best_gif = best_gif if render else None
        observations = observations if get_obs else None
        actions = actions if get_actions else None

        # EvaluationUtils.__store_pickled_rollout(
        #     rewards,
        #     best_gif,
        #     observations,
        #     actions,
        #     n_episodes,
        #     trainer,
        #     inject,
        #     inject_mode,
        #     agents_to_inject,
        #     noise_delta,
        # )

        return (
            rewards,
            best_gif,
            observations,
            actions,
        )

    @staticmethod
    def __store_pickled_rollout(
        rewards,
        best_gif,
        observations,
        actions,
        n_episodes,
        trainer: MultiPPOTrainer,
        inject: bool,
        inject_mode: InjectMode,
        agents_to_inject: Set,
        noise_delta: float,
    ):
        (
            _,
            model_name,
            _,
            env_name,
        ) = EvaluationUtils.get_model_name(trainer.config)

        _, inject_name = EvaluationUtils.get_inject_name(
            agents_to_inject=agents_to_inject,
            noise_delta=noise_delta,
            inject_mode=inject_mode,
        )

        name = f"{n_episodes}epis_{model_name}_{env_name}" + (
            "_" + inject_name if inject else ""
        )

        reward_file = PathUtils.rollout_storage / f"rew_{name}.pkl"
        best_gif_file = PathUtils.rollout_storage / f"gif_{name}.pkl"
        observations_file = PathUtils.rollout_storage / f"obs_{name}.pkl"
        actions_file = PathUtils.rollout_storage / f"acts_{name}.pkl"

        pickle.dump(rewards, open(reward_file, "wb"))
        if best_gif is not None:
            pickle.dump(best_gif, open(best_gif_file, "wb"))
        if observations is not None:
            pickle.dump(observations, open(observations_file, "wb"))
        if actions is not None:
            pickle.dump(actions, open(actions_file, "wb"))

    @staticmethod
    def __get_pickled_rollout(
        n_episodes: int,
        render: bool,
        get_obs: bool,
        get_actions: bool,
        trainer: MultiPPOTrainer,
        inject: bool,
        inject_mode: InjectMode,
        agents_to_inject: Set,
        noise_delta: float,
    ):

        (
            _,
            model_name,
            _,
            env_name,
        ) = EvaluationUtils.get_model_name(trainer.config)

        _, inject_name = EvaluationUtils.get_inject_name(
            agents_to_inject=agents_to_inject,
            noise_delta=noise_delta,
            inject_mode=inject_mode,
        )

        name = f"{n_episodes}epis_{model_name}_{env_name}" + (
            "_" + inject_name if inject else ""
        )

        rewards = None
        best_gif = (None,)
        observations = (None,)
        actions = (None,)

        reward_file = PathUtils.rollout_storage / f"rew_{name}.pkl"
        best_gif_file = PathUtils.rollout_storage / f"gif_{name}.pkl"
        observations_file = PathUtils.rollout_storage / f"obs_{name}.pkl"
        actions_file = PathUtils.rollout_storage / f"acts_{name}.pkl"

        if (
            (render and not best_gif_file.is_file())
            or (get_obs and not observations_file.is_file())
            or (get_actions and not actions_file.is_file())
        ):
            return rewards, best_gif, observations, actions

        if reward_file.is_file():
            rewards = pickle.load(open(reward_file, "rb"))
            if render:
                best_gif = pickle.load(open(best_gif_file, "rb"))
            if get_obs:
                observations = pickle.load(open(observations_file, "rb"))
            if get_actions:
                actions = pickle.load(open(actions_file, "rb"))

        return rewards, best_gif, observations, actions

    @staticmethod
    def get_model_name(config):

        # Model
        is_hetero = config["model"]["custom_model_config"]["heterogeneous"]
        is_gippo = config["model"]["custom_model_config"]["share_observations"]

        # Env
        env_config = config["env_config"]
        scenario_name = env_config["scenario_name"]
        scenario_config = env_config["scenario_config"]

        model_title = f"{'Het' if is_hetero else ''}{'GIPPO' if is_gippo else 'IPPO'}"
        model_name = model_title.lower().replace(" ", "_")

        env_title = scenario_name
        env_name = scenario_name.lower().replace(" ", "_")

        return model_title, model_name, env_title, env_name

    @staticmethod
    def get_inject_name(
        agents_to_inject: Set, inject_mode: InjectMode, noise_delta: float
    ):
        if agents_to_inject is not None and len(agents_to_inject) > 0:
            noise_title = (
                f"Agents injected: {agents_to_inject}, Inject mode: {inject_mode.name}"
                + (
                    " ($\\pm{}$ uniform noise)".format(noise_delta)
                    if inject_mode.is_noise()
                    else ""
                )
            )

            noise_name = (
                f"agents_injected_{agents_to_inject}_inject_mode_{inject_mode.name}"
                + (
                    "_{}_delta_noise".format(noise_delta)
                    if inject_mode.is_noise()
                    else ""
                )
            )

            return noise_title, noise_name
        return "", ""
