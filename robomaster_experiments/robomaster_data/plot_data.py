#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os

import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib import pyplot as plt


def get_robot_columns(index: int):
    robot_pos_columns = [
        f"/robomaster_{index}/current_state/state_vector.0",
        f"/robomaster_{index}/current_state/state_vector.1",
        f"/robomaster_{index}/current_state/state_vector.2",
    ]

    robot_vel_columns = [
        f"/robomaster_{index}/current_state/state_vector.3",
        f"/robomaster_{index}/current_state/state_vector.4",
        f"/robomaster_{index}/current_state/state_vector.5",
    ]

    robot_vel_goal = [
        f"/robomaster_{index}/reference_state/vd",
        f"/robomaster_{index}/reference_state/ve",
        f"/robomaster_{index}/reference_state/vn",
    ]
    return robot_pos_columns, robot_vel_columns, robot_vel_goal


def load_and_process_csv(file: str):
    df = pd.read_csv(
        file,
    )

    time = "__time"

    robot_pos_columns, robot_vel_columns, robot_vel_goal = get_robot_columns(1)
    robot2_pos_columns, robot2_vel_columns, robot2_vel_goal = get_robot_columns(2)

    df = df[
        [time]
        + robot_pos_columns
        + robot_vel_columns
        + robot_vel_goal
        + robot2_pos_columns
        + robot2_vel_columns
        + robot2_vel_goal
    ]

    # Drop the first n rows with nan goal
    df = df.loc[df[robot_vel_goal[0]].notnull().argmax() :]
    df = df.loc[df[robot2_vel_goal[0]].notnull().argmax() :]

    # FIll goal nan forwards
    df = df.ffill()

    # Reset time
    df[time] = df[time] - df[time].iloc[0]

    # Drop nan rows
    df.dropna(inplace=True)

    # Set time on x axis
    df = df.set_index(time)

    # Compute completion based on distance to goal
    df["completion"] = (df["/robomaster_1/current_state/state_vector.1"] - 2).clip(
        None, 0
    ).abs() + (df["/robomaster_2/current_state/state_vector.1"] + 2).clip(0, None).abs()
    df["completion"] /= -8
    df["completion"] += 1

    # df[
    #     [
    #         "/robomaster_1/current_state/state_vector.4",
    #         "/robomaster_1/reference_state/ve",
    #         "/robomaster_2/current_state/state_vector.4",
    #         "/robomaster_2/reference_state/ve",
    #         "/robomaster_1/current_state/state_vector.1",
    #         "/robomaster_2/current_state/state_vector.1",
    #     ]
    # ].plot()
    # df[["completion"]].plot()
    # plt.show()

    # df.to_csv(os.path.dirname(os.path.realpath(__file__)) + "/ground_run_processed.csv")

    return df


def get_robot_columns_2(index: int):
    robot_state_columns = [
        f"/robomaster_{index}/pe",
    ]

    robot_ref_state_columns = [
        f"/robomaster_{index}/reference_state/vd",
        f"/robomaster_{index}/reference_state/ve",
        f"/robomaster_{index}/reference_state/vn",
    ]
    return robot_state_columns, robot_ref_state_columns


def load_and_process_csv_2(file: str):
    df = pd.read_csv(
        file,
    )

    time = "time"

    robot_state_columns, robot_ref_state_columns = get_robot_columns_2(1)
    robot2_state_columns, robot2_ref_state_columns = get_robot_columns_2(2)

    df = df[
        [time]
        + robot_state_columns
        + robot_ref_state_columns
        + robot2_state_columns
        + robot2_ref_state_columns
    ]

    # Drop the first n rows with nan goal
    df = df.loc[df[robot_ref_state_columns[0]].notnull().argmax() :]
    df = df.loc[df[robot2_ref_state_columns[0]].notnull().argmax() :]

    # FIll goal nan forwards
    df = df.ffill()

    # Reset time
    df[time] = df[time] - df[time].iloc[0]

    # Drop nan rows
    df.dropna(inplace=True)

    # Set time on x axis
    df = df.set_index(time)

    # Compute completion based on distance to goal
    df["completion"] = (df["/robomaster_1/pe"] - 2).clip(None, 0).abs() + (
        df["/robomaster_2/pe"] + 2
    ).clip(0, None).abs()
    df["completion"] /= -8
    df["completion"] += 1

    # df[
    #     [
    #         "/robomaster_1/current_state/state_vector.4",
    #         "/robomaster_1/reference_state/ve",
    #         "/robomaster_2/current_state/state_vector.4",
    #         "/robomaster_2/reference_state/ve",
    #         "/robomaster_1/current_state/state_vector.1",
    #         "/robomaster_2/current_state/state_vector.1",
    #     ]
    # ].plot()
    # df[["completion"]].plot()
    # plt.show()

    # df.to_csv(os.path.dirname(os.path.realpath(__file__)) + "/ground_run_processed.csv")

    return df


def plot_completions(dfs_het, dfs_homo):
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

    fig, ax = plt.subplots(figsize=(5, 5))

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

    het_completions = []
    het_times = []
    for df in dfs_het:
        completion = df["completion"].clip(0, None).to_numpy()
        time = df.index.values

        time = time[: np.argmax(completion) + 1]
        completion = completion[: np.argmax(completion) + 1]

        het_completions.append(completion)
        het_times.append(time)

    for i, episode_obs in enumerate(het_completions):
        ax.plot(
            het_times[i],
            episode_obs,
            label="HetGPPO" if i == 0 else None,
            color=CB_color_cycle[0],
        )

    homo_completions = []
    homo_times = []
    for df in dfs_homo:
        completion = df["completion"].clip(0, None).to_numpy()
        time = df.index.values

        time = time[: np.argmax(completion) + 1]
        completion = completion[: np.argmax(completion) + 1]

        homo_completions.append(completion)
        homo_times.append(time)

    for i, episode_obs in enumerate(homo_completions):
        ax.plot(
            homo_times[i],
            episode_obs,
            label="GPPO" if i == 0 else None,
            color=CB_color_cycle[1],
        )

    ax.grid()
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Task completion")
    ax.set_xlim(0, 60)
    plt.xticks(np.arange(0, 60, 10))
    # ax.legend()

    tikzplotlib.save(
        f"trial.tex",
        textsize=18,
    )
    plt.savefig(f"trial.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    dfs_het = []
    for i in range(1, 11):
        file = (
            os.path.dirname(os.path.realpath(__file__)) + f"/data/het/heterog-{i}.csv"
        )
        dfs_het.append(load_and_process_csv_2(file))
    dfs_homo = []
    for i in range(1, 11):
        file = os.path.dirname(os.path.realpath(__file__)) + f"/data/homo/homog-{i}.csv"
        dfs_homo.append(load_and_process_csv_2(file))
    plot_completions(dfs_het, dfs_homo)
