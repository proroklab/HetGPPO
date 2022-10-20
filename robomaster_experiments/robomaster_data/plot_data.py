import os

import pandas as pd
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


def load_and_process_csv2(file: str):
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
    df["completion"] = (df["/robomaster_1/current_state/state_vector.1"] - 2).abs() + (
        df["/robomaster_2/current_state/state_vector.1"] + 2
    ).abs()
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
    df[["completion"]].plot()
    plt.show()

    # df.to_csv(os.path.dirname(os.path.realpath(__file__)) + "/ground_run_processed.csv")

    return df


def success_from_x_position(x1, x2):
    distance_from_goal1 = abs(x1 - 2)
    distance_from_goal2 = abs(2 + x2)
    distance = distance_from_goal1 + distance_from_goal2  # 0 to 8
    distance /= -8
    distance += 1
    return distance


if __name__ == "__main__":
    dfs = []
    for i in range(1, 11):
        file = os.path.dirname(os.path.realpath(__file__)) + f"/het/output_het_{i}.csv"
        dfs.append(load_and_process_csv2(file))
