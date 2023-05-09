#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from matplotlib import pyplot as plt

from evaluate.wandb_plot import get_wandb_panel


def plot_asym_joint(
    projects, x_iterations, x_axis_name, attribute, attribute_name, legend: bool = True
):
    projects_dfs = []
    for project in projects:
        # Get data
        projects_dfs.append(
            get_wandb_panel(
                project,
                [],
                attribute_name=attribute,
                x_iterations=x_iterations,
                x_axis_name=x_axis_name,
            )
        )

    # Plot it

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.serif": "Times New Roman",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 20,
        "font.size": 20,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 20,
        "legend.title_fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }

    plt.rcParams.update(tex_fonts)

    fig, axs = plt.subplots(1, 3, figsize=(19, 5))

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

    for j, groups_and_dfs in enumerate(projects_dfs):
        ax = axs[j]
        groups_and_dfs = sorted(groups_and_dfs, key=lambda e: e[0])
        for i, (group, df) in enumerate(groups_and_dfs):
            mean = df["mean"].to_numpy()
            std = df["std"].to_numpy()
            iteration = (df[x_axis_name].to_numpy() * 60_000) / 1_000_000

            label = eval(group)["group"]

            if label == "HetMADDPG":
                label = "MADDPG"

            (mean_line,) = ax.plot(
                iteration,
                mean,
                label=label if j == 0 else None,
                color=CB_color_cycle[i],
            )
            ax.fill_between(
                iteration,
                mean + std,
                mean - std,
                color=mean_line.get_color(),
                alpha=0.3,
            )
        ax.grid()
        ax.set_xlabel("Number of frames (M)")
        ax.set_title(projects[j].split("_")[-1].capitalize())

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel("Mean reward", labelpad=10)

    if legend:
        fig.legend(
            fancybox=True,
            shadow=True,
            ncol=len(projects_dfs[0]),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
        )
    plt.tight_layout()
    plt.savefig(
        f"{project}_{attribute_name}.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


if __name__ == "__main__":
    plot_asym_joint(
        projects=[
            "torchrl_navigation",
            "torchrl_balance",
            "torchrl_sampling",
        ],
        x_iterations=500,
        x_axis_name="train/training_iteration",
        attribute="train/reward/reward_mean",
        attribute_name="Reward",
        legend=True,
    )
