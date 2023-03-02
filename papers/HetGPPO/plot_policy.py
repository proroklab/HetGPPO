#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


from matplotlib import pyplot as plt

from evaluate.plot_policy import plot_policy
from utils import InjectMode


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
