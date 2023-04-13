#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import numpy as np
from matplotlib import pyplot as plt

from evaluate.evaluate_resiliance import evaluate_increasing_noise
from utils import InjectMode


def add_increasing_noise():
    noises = np.linspace(0, 2, 10)

    checkpoint_paths = [
        "/Users/Matteo/Downloads/multi_give_way/het/MultiPPOTrainer_multi_give_way_59e45_00000_0_2023-04-13_08-11-03/checkpoint_000173",
        "/Users/Matteo/Downloads/multi_give_way/homo/MultiPPOTrainer_multi_give_way_96e75_00000_0_2023-04-12_13-00-16/checkpoint_000412",
    ]

    fig, ax = plt.subplots(figsize=(4, 4))

    def inject_config_fn(config):
        # config["env_config"]["scenario_config"].update({"mass_position": -0.75})
        return config

    # evaluate_resilience(
    #     checkpoint_paths_mass_in_side,
    #     n_episodes_per_model=50,
    #     agents_to_inject={0, 1},
    #     inject_mode=InjectMode.SWITCH_AGENTS,
    #     noise_delta=0,
    #     inject_config_fn=inject_config_fn,
    #     ax=ax,
    #     plotting_mode=ResilencePlottinMode.PERFORMANCE_MAINTEINED,
    # )

    rewards = evaluate_increasing_noise(
        checkpoint_paths,
        n_episodes_per_model=50,
        agents_to_inject={0, 1, 2, 3},
        inject_mode=InjectMode.OBS_NOISE,
        noises=noises,
        ax=ax,
    )
    plt.savefig(f"asym_joint_resilience.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()


if __name__ == "__main__":
    add_increasing_noise()
