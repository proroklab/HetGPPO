#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from pathlib import Path
from typing import Union, Set

import cv2

from utils import EvaluationUtils, PathUtils, InjectMode


def render(
    checkpoint_path: Union[str, Path],
    n_episodes: int,
    agents_to_inject: Set = None,
    inject_mode: InjectMode = None,
    noise_delta: float = None,
):

    config, trainer, env = EvaluationUtils.get_config_trainer_and_env_from_checkpoint(
        checkpoint_path
    )

    inject = agents_to_inject is not None and len(agents_to_inject) > 0
    rewards, best_gif, _, _ = EvaluationUtils.rollout_episodes(
        n_episodes=n_episodes,
        render=True,
        get_obs=False,
        get_actions=False,
        trainer=trainer,
        env=env,
        inject=inject,
        inject_mode=inject_mode,
        noise_delta=noise_delta,
        agents_to_inject=agents_to_inject,
    )

    (
        model_title,
        model_name,
        env_title,
        env_name,
    ) = EvaluationUtils.get_model_name(config)

    inject_title, inject_name = EvaluationUtils.get_inject_name(
        agents_to_inject=agents_to_inject,
        noise_delta=noise_delta,
        inject_mode=inject_mode,
    )

    save_dir = PathUtils.result_dir / f"{env_title}/{model_title}/videos"
    name = f"{model_name}_{env_name}" + ("_" + inject_name if inject else "")
    video = cv2.VideoWriter(
        str(save_dir / f"{name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # FPS
        (best_gif[0].shape[1], best_gif[0].shape[0]),
    )
    for img in best_gif:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)
    video.release()


if __name__ == "__main__":
    render(
        "",
        n_episodes=1,
        agents_to_inject=None,
        inject_mode=InjectMode.OBS_NOISE,
        noise_delta=0.5,
    )
