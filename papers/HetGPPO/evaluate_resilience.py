#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from matplotlib import pyplot as plt

from evaluate.evaluate_resiliance import evaluate_increasing_noise
from utils import InjectMode


def evaluate_het_test():
    # HetGIPPO
    checkpoint_paths.append(
        "/Users/Matteo/Downloads/het_test/hetgippo/MultiPPOTrainer_het_test_ecc9e_00000_0_2022-09-12_14-43-46/checkpoint_000093/checkpoint-93"
    )
    # Gippo
    checkpoint_paths.append(
        "/Users/Matteo/Downloads/het_test/gippo/MultiPPOTrainer_het_test_18f2c_00000_0_2022-09-12_14-45-00/checkpoint_000116/checkpoint-116"
    )

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{libertine}\n\\usepackage[libertine]{newtxmath}",
        "font.family": "Linux Libertine",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 19,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 16,
        "legend.title_fontsize": 7,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }

    plt.rcParams.update(tex_fonts)

    evaluate_increasing_noise(
        checkpoint_paths, 100, {0, 1}, InjectMode.OBS_NOISE, ResilencePlottinMode.VIOLIN
    )  # 50 datapoints


def evaluate_give_way():
    # Give way

    # HetGippo
    checkpoint_paths.append(
        "/Users/Matteo/Downloads/give_way/hetgippo/MultiPPOTrainer_give_way_553c5_00000_0_2022-09-12_23-00-37/checkpoint_000300/checkpoint-300"
    )
    # Gippo
    checkpoint_paths.append(
        "/Users/Matteo/Downloads/give_way/gippo/MultiPPOTrainer_give_way_5dee1_00000_0_2022-09-12_23-00-52/checkpoint_000300/checkpoint-300"
    )

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{libertine}\n\\usepackage[libertine]{newtxmath}",
        "font.family": "Linux Libertine",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 19,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 16,
        "legend.title_fontsize": 7,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }

    plt.rcParams.update(tex_fonts)

    evaluate_increasing_noise(
        checkpoint_paths, 100, {0, 1}, InjectMode.OBS_NOISE, ResilencePlottinMode.VIOLIN
    )  # 50 datapoints
