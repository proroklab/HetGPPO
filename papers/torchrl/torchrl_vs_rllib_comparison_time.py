#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from matplotlib import pyplot as plt

from evaluate.wandb_plot import get_wandb_panel


def plot_asym_joint(legend: bool = True):
    x_iterations = 500
    projects = [
        "navigation",
        "balance",
        "sampling",
    ]

    rllib_dfs = []
    for project in projects:
        # Get data
        x_axis_name = "training_iteration"
        attribute = "time_this_iter_s"
        rllib_dfs.append(
            get_wandb_panel(
                project,
                [],
                attribute_name=attribute,
                x_iterations=x_iterations,
                x_axis_name=x_axis_name,
            )
        )
    torchrl_df = []
    for project, x_axis_name in zip(
        projects, ["counters/iter", "counters/total_iter", "counters/iter"]
    ):
        attribute = "timers/iteration_time"
        torchrl_df.append(
            get_wandb_panel(
                "benchmarl",
                [],
                attribute_name=attribute,
                x_iterations=x_iterations,
                x_axis_name=x_axis_name,
                filter={"task_name": project, "algorithm_name": "ippo"},
            )
        )

    # Plot it

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "Times New Roman",
        # "font.serif": "Times New Roman",
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

    for (j, groups_and_dfs), groups_and_dfs_torchrl, x_axis_name in zip(
        enumerate(rllib_dfs),
        torchrl_df,
        ["counters/iter", "counters/total_iter", "counters/iter"],
    ):
        ax = axs[j]
        groups_and_dfs = sorted(groups_and_dfs, key=lambda e: e[0])
        for (i, (group, df)), (groups_torchrl, df_torchrl) in zip(
            enumerate(groups_and_dfs), groups_and_dfs_torchrl
        ):
            mean = df["mean"].to_numpy()
            std = df["std"].to_numpy()
            iteration = (df["training_iteration"].to_numpy() * 60_000) / 1_000_000

            label = "RLlib"

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

            mean = df_torchrl["mean"].to_numpy()
            std = df_torchrl["std"].to_numpy()
            iteration = (df_torchrl[x_axis_name].to_numpy() * 60_000) / 1_000_000
            label = "TorchRL"
            (mean_line,) = ax.plot(
                iteration,
                mean,
                label=label if j == 0 else None,
                color=CB_color_cycle[i + 1],
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
    plt.ylabel("Iteration time (s)", labelpad=10)

    if legend:
        fig.legend(
            fancybox=True,
            shadow=True,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
        )
    plt.tight_layout()
    plt.savefig(
        f"time.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


if __name__ == "__main__":
    plot_asym_joint(
        legend=True,
    )
