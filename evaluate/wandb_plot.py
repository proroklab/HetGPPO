#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import List, Dict

import pandas as pd
import wandb


def get_wandb_panel(
    project_name: str,
    groups: List[str],
    attribute_name: str = "episode_reward_mean",
    x_axis_name: str = "training_iteration",
    x_iterations: int = 400,
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
            samples=x_iterations,
            x_axis=x_axis_name,
        )
        history = history.rename(columns={attribute_name: run.name})
        run_history_groups.append((run, history, group))

    unique_groups = list({str(group) for _, _, group in run_history_groups})
    unique_groups_dfs = [
        pd.DataFrame(range(x_iterations), columns=[x_axis_name]) for _ in unique_groups
    ]

    for i, run_history_group in enumerate(run_history_groups):
        run, history, group = run_history_group
        group_index = unique_groups.index(str(group))

        unique_groups_dfs[group_index] = pd.merge(
            unique_groups_dfs[group_index],
            history,
            how="outer",
            on=x_axis_name,
        )
    for i, group_df in enumerate(unique_groups_dfs):
        temp_df = group_df.loc[:, group_df.columns.intersection([x_axis_name])]
        temp_df[["mean", "std"]] = group_df.drop(columns=[x_axis_name]).agg(
            ["mean", "std"], axis="columns"
        )
        temp_df["std"].fillna(0, inplace=True)
        unique_groups_dfs[i] = temp_df

    return [
        (group, unique_groups_dfs[i].dropna()) for i, group in enumerate(unique_groups)
    ]
