#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import List, Dict

import numpy as np
import pandas as pd
import tikzplotlib
import wandb
from matplotlib import pyplot as plt


def get_wandb_panel(
    project_name: str,
    groups: List[str],
    attribute_name: str = "episode_reward_mean",
    training_iterations: int = 400,
    filter: Dict = None,
):
    api = wandb.Api()
    runs = api.runs(f"matteobettini/{project_name}")

    run_history_groups = (
        []
    )  # list of runs, each element is a dataframe with 2 columns ("training_iteration" and attribute_name) the number of rows is the number of samples available

    for run in runs:
        # Filter out run
        skip = False
        if filter is not None:
            for item in filter.items():
                if (
                    item not in run.config["env_config"]["scenario_config"].items()
                    and item not in run.config["model"]["custom_model_config"].items()
                ):
                    skip = True
                    break
        if skip:
            continue

        # Decide group of this run
        group = {"group": run.group}
        for element in groups:
            group[element] = run.config["env_config"]["scenario_config"][element]

        history = run.history(
            keys=[attribute_name],
            samples=training_iterations,
            x_axis="training_iteration",
        )
        history = history.rename(columns={attribute_name: run.name})
        run_history_groups.append((run, history, group))

    unique_groups = list({str(group) for _, _, group in run_history_groups})
    unique_groups_dfs = [
        pd.DataFrame(range(1, training_iterations - 1), columns=["training_iteration"])
        for _ in unique_groups
    ]

    for i, run_history_group in enumerate(run_history_groups):
        run, history, group = run_history_group
        group_index = unique_groups.index(str(group))

        unique_groups_dfs[group_index] = pd.merge(
            unique_groups_dfs[group_index],
            history,
            how="outer",
            on="training_iteration",
        )
    for i, group_df in enumerate(unique_groups_dfs):

        temp_df = group_df.loc[:, group_df.columns.intersection(["training_iteration"])]
        temp_df[["mean", "std"]] = group_df.drop(columns=["training_iteration"]).agg(
            ["mean", "std"], axis="columns"
        )
        unique_groups_dfs[i] = temp_df

    return [
        (group, unique_groups_dfs[i].dropna()) for i, group in enumerate(unique_groups)
    ]


def plot_multiple_data(data_list, project_name, index, noise_list, attribute):

    gppo_mean = []
    hetgppo_mean = []
    gppo_std = []
    hetgppo_std = []
    for data in data_list:
        data = data.iloc[index]
        gppo_mean.append(data[f"GIPPO_{attribute}"])
        hetgppo_mean.append(data[f"HetGIPPO_{attribute}"])
        gppo_std.append(data[f"GIPPO_{attribute}_std"])
        hetgppo_std.append(data[f"HetGIPPO_{attribute}_std"])

    x = noise_list

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{libertine}\n\\usepackage[libertine]{newtxmath}",
        "font.family": "Linux Libertine",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 30,
        "font.size": 13,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 26,
        "legend.title_fontsize": 7,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
    }

    plt.rcParams.update(tex_fonts)

    fig, ax = plt.subplots(figsize=(10, 6))

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

    gppo_mean = np.array(gppo_mean)
    gppo_std = np.array(gppo_std)
    hetgppo_mean = np.array(hetgppo_mean)
    hetgppo_std = np.array(hetgppo_std)

    ax.errorbar(
        x,
        hetgppo_mean,
        yerr=hetgppo_std,
        capsize=10,
        marker="o",
        markersize=8,
        label="HetGPPO",
    )
    ax.fill_between(
        x,
        hetgppo_mean + hetgppo_std,
        hetgppo_mean - hetgppo_std,
        color=CB_color_cycle[0],
        alpha=0.3,
    )

    # ax.plot(x, gppo_mean, color=CB_color_cycle[1])
    ax.errorbar(
        x, gppo_mean, yerr=gppo_std, label="GPPO", capsize=10, marker="o", markersize=8
    )
    ax.fill_between(
        x,
        gppo_mean + gppo_std,
        gppo_mean - gppo_std,
        color=CB_color_cycle[1],
        alpha=0.3,
    )

    ax.set_ylabel("Success rate")
    ax.set_xlabel("Training observation noise")
    plt.xticks(x)
    ax.legend(loc="lower left")

    ax.grid()

    tikzplotlib.save(
        f"{project_name}.tex",
        textsize=9,
    )
    plt.savefig(f"{project_name}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    project = "joint"
    training_iterations = 999
    attribute = "custom_metrics/agent 0/passed_mean"  # episode_reward_mean"
    noise_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    # noise_list = [0.15]
    data_list = []

    for noise in noise_list:
        data = get_wandb_panel(
            project,
            groups=["same_goal"],
            attribute_name=attribute,
            training_iterations=training_iterations,
            filter={
                "fixed_passage": True,
                "obs_noise": noise,
                "asym_package": True,
            },
        )
        data_list.append(data)
    plot_multiple_data(
        data_list,
        project_name=project,
        index=training_iterations - 1,
        attribute=attribute,
        noise_list=noise_list,
    )
    # plot_data(data, project_name=project, training_iterations=training_iterations)
