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

    noises = np.linspace(0, 1, 50)

    checkpoint_paths = [
        "/Users/Matteo/Downloads/give_way/HetGPPO/MultiPPOTrainer_give_way_12608_00000_0_2023-03-14_14-11-10/checkpoint_001019",
        "/Users/Matteo/Downloads/give_way/GPPO/MultiPPOTrainer_give_way_0cecc_00000_0_2023-03-14_14-11-01/checkpoint_000158",
    ]

    fig, ax = plt.subplots(figsize=(6, 6))

    def inject_config_fn(config):
        # config["env_config"]["scenario_config"].update({"mass_position": -0.75})
        return config

    evaluate_resilience(
        checkpoint_paths,
        n_episodes_per_model=50,
        agents_to_inject={0, 1},
        inject_mode=InjectMode.SWITCH_AGENTS,
        noise_delta=0,
        inject_config_fn=inject_config_fn,
        ax=ax,
        plotting_mode=ResilencePlottinMode.VIOLIN,
    )

    # evaluate_increasing_noise(
    #     checkpoint_paths_mass_in_side,
    #     n_episodes_per_model=50,
    #     agents_to_inject={0, 1},
    #     inject_mode=InjectMode.OBS_NOISE,
    #     noises=noises,
    #     ax=ax,
    # )
    plt.savefig(f"trial.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    add_increasing_noise_to_noisless_asymmetric_joint()
