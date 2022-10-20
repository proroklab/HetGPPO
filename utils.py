import pickle
import platform
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, Callable
from typing import Union

import numpy as np
import ray
import torch
import vmas
import wandb
from models.gippo_small import GIPPOv2Small
from ray.rllib import RolloutWorker, BaseEnv, Policy, VectorEnv
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from vmas import make_env

from evaluate.evaluate_model import clamp_with_norm
from models.gippo import GIPPOv2
from rllib_differentiable_comms.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)
from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer


class TorchDiagGaussian:
    """Wrapper class for PyTorch Normal distribution."""

    def __init__(
        self,
        inputs,
        u_range,
    ):
        self.inputs = inputs
        mean, log_std = torch.chunk(self.inputs, 2, dim=1)
        self.dist = torch.distributions.normal.Normal(mean, torch.exp(log_std))
        self.u_range = u_range

    def sample(self):
        sample = self.dist.sample()
        return self._squash(sample)

    def deterministic_sample(self):
        sample = self.dist.mean
        return self._squash(sample)

    def _squash(self, sample):
        sample = -self.u_range + (sample + 1.0) * (2 * self.u_range) / 2.0
        return sample.clamp(-self.u_range, self.u_range)


def compute_action_corridor(
    pos0_x: float,
    pos0_y: float,
    vel0_x: float,
    vel0_y: float,
    pos1_x: float,
    pos1_y: float,
    vel1_x: float,
    vel1_y: float,
    model,
    u_range: float,
    n_agents: int = 2,
    obs_size: int = 4,
    num_actions_per_agent: int = 2,
    deterministic: bool = False,
    circular_action_constraint: bool = True,
):
    observation = torch.zeros((1, n_agents, obs_size))
    agent_input = [[vel0_x, vel0_y, pos0_x, pos0_y], [vel1_x, vel1_y, pos1_x, pos1_y]]
    for j in range(n_agents):
        for i, val in enumerate(agent_input[j]):
            observation[:, j, i] = float(val)
    pos = observation[..., 2:]
    vel = observation[..., :2]
    logits = model(observation, pos, vel)[0].detach()[0]

    input_lens = [2 * num_actions_per_agent] * n_agents

    logits = logits.view(1, -1)
    split_inputs = torch.split(logits, input_lens, dim=1)
    action_dists = []
    for agent_inputs in split_inputs:
        action_dists.append(TorchDiagGaussian(agent_inputs, u_range))
    actions = []
    for action_dist in action_dists:
        if deterministic:
            action = action_dist.deterministic_sample()[0]
        else:
            action = action_dist.sample()[0]
        if circular_action_constraint:
            action = clamp_with_norm(action, u_range)
        actions.append(action.tolist())
    return actions


class PathUtils:
    scratch_dir = (
        Path("/Users/Matteo/scratch/")
        if platform.system() == "Darwin"
        else Path("/local/scratch/mb2389/")
    )
    gippo_dir = Path(__file__).parent.resolve()
    result_dir = gippo_dir / "results"
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
        ModelCatalog.register_custom_model("GIPPO", GIPPOv2)
        ModelCatalog.register_custom_model("GIPPOSmall", GIPPOv2Small)
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
            episode.media["rendering"] = wandb.Video(vid, fps=30, format="mp4")
            self.frames = []


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
        env: VectorEnv,
        inject: bool,
        inject_mode: InjectMode,
        agents_to_inject: Set,
        noise_delta: float,
        het: bool,
    ):
        # print(
        #     f"\nLoaded: {EvaluationUtils.get_model_name(trainer.config)[0]}, {EvaluationUtils.get_model_name(trainer.config)[2]}"
        # )
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

        model = torch.load(
            f"/Users/Matteo/Downloads/{'het_' if het else ''}a_range_1_u_range_0_5_[3_2_0_002]_0_05_dt_0_1_friction_0_dt_delay_option_0.pt"
        )
        model.eval()

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
                # if 150 < i < 200:
                #     observation[0][2] = -observation[0][2]
                #     observation[1][2] = -observation[1][2]
                #     print("it happenmed")
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
                # print(action)
                # action = trainer.compute_single_action(observation)
                #
                # print(action)

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
