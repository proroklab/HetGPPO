#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import numpy as np
from matplotlib import pyplot as plt

from evaluate.evaluate_resiliance import (
    evaluate_resilience,
    ResilencePlottinMode,
)
from utils import InjectMode


def add_increasing_noise_to_noisless_asymmetric_joint():
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": "\\renewcommand{\\familydefault}{\\sfdefault}\n\\usepackage{helvet}",
        "font.family": "sans-serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 15,
        "font.size": 15,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 15,
        "legend.title_fontsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    plt.rcParams.update(tex_fonts)

    noises = np.linspace(0, 1, 10)

    checkpoint_paths_mass_in_side = [
        "/Users/Matteo/Downloads/joint_passage/HetGPPO/MultiPPOTrainer_joint_passage_bb7d6_00000_0_2023-02-27_11-13-16/checkpoint_000589",
        "/Users/Matteo/Downloads/joint_passage/GPPO/MultiPPOTrainer_joint_passage_3bdd9_00000_0_2023-02-24_10-01-40/checkpoint_000379",
    ]

    checkpoint_paths_mass_in_middle = [
        "/Users/Matteo/Downloads/joint_passage/HetGPPO/MultiPPOTrainer_joint_passage_db988_00000_0_2023-02-27_08-22-22/checkpoint_000589",
        "/Users/Matteo/Downloads/joint_passage/GPPO/MultiPPOTrainer_joint_passage_42a3f_00000_0_2023-02-27_06-59-21/checkpoint_000914",
    ]
    fig, ax = plt.subplots(figsize=(6, 6))

    def inject_config_fn(config):
        # config["env_config"]["scenario_config"].update({"mass_position": -0.75})
        return config

    evaluate_resilience(
        checkpoint_paths_mass_in_side,
        n_episodes_per_model=50,
        agents_to_inject={0, 1},
        inject_mode=InjectMode.SWITCH_AGENTS,
        noise_delta=0,
        inject_config_fn=inject_config_fn,
        ax=ax,
        plotting_mode=ResilencePlottinMode.PERFORMANCE_MAINTEINED,
    )

    # evaluate_increasing_noise(
    #     checkpoint_paths,
    #     n_episodes_per_model=1,
    #     agents_to_inject={0, 1},
    #     inject_mode=InjectMode.OBS_NOISE,
    #     noises=noises,
    #     ax=ax,
    # )
    plt.savefig(f"trial.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    add_increasing_noise_to_noisless_asymmetric_joint()
