#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from matplotlib import pyplot as plt

from evaluate.wandb_plot import get_wandb_panel


def plot_asym_joint(attribute, attribute_name, legend: bool = True):
    project = "joint_passage"
    training_iterations = 1000

    # Get data
    groups_and_dfs = get_wandb_panel(
        project,
        [],
        attribute_name=attribute,
        training_iterations=training_iterations,
        filter={"obs_noise": 0, "mass_position": 0.75},
    )

    # Plot it

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": "\\renewcommand{\\familydefault}{\\sfdefault}\n\\usepackage{helvet}",
        "font.family": "sans-serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 30,
        "font.size": 30,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 25,
        "legend.title_fontsize": 25,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
    }

    plt.rcParams.update(tex_fonts)

    fig, ax = plt.subplots(figsize=(10, 5))

    CB_color_cycle = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]
    groups_and_dfs = reversed(sorted(groups_and_dfs, key=lambda e: e[0]))
    for i, (group, df) in enumerate(groups_and_dfs):
        if attribute == "sampler_results/episode_reward_mean":
            df[["mean", "std"]] = df[["mean", "std"]] / 3

        mean = df["mean"].to_numpy()
        std = df["std"].to_numpy()
        iteration = df["training_iteration"].to_numpy()

        label = eval(group)["group"]

        (mean_line,) = ax.plot(iteration, mean, label=label, color=CB_color_cycle[i])
        ax.fill_between(
            iteration,
            mean + std,
            mean - std,
            color=mean_line.get_color(),
            alpha=0.3,
        )

    ax.set_xlabel("Training iteration")
    ax.set_ylabel(attribute_name)
    if legend:
        ax.legend(
            fancybox=True,
            shadow=True,
            ncol=1,
            loc="lower left",
            bbox_to_anchor=(0.57, 0.02),
        )
    # ax.set_ylim(-2, 3)
    ax.grid()

    # tikzplotlib.save(
    #     f"{project_name}.tex",
    #     textsize=9,
    # )
    plt.savefig(
        f"{project}_{attribute_name}.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


if __name__ == "__main__":
    # plot_asym_joint(
    #     attribute="evaluation/custom_metrics/mine/wasserstein_mean",
    #     attribute_name="SND",
    #     legend=False,
    # )
    plot_asym_joint(
        attribute="sampler_results/episode_reward_mean",
        attribute_name="Reward",
        legend=True,
    )
