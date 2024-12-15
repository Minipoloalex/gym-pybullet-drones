import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import time

def plot_moving_average(data, x, y, hue, title, x_label = None, y_label = None, color = None, window = 10):
    x_label = x_label if x_label is not None else x
    y_label = y_label if y_label is not None else y

    g = data.groupby([hue])[y]
    data[f"{y}_avg"] = g.rolling(window = window).mean().droplevel([0])
    data[f"{y}_max"] = g.rolling(window = window).max() .droplevel([0])
    data[f"{y}_min"] = g.rolling(window = window).min() .droplevel([0])

    so.Plot(data, x = x) \
        .add(so.Band(), ymin = f"{y}_min", ymax = f"{y}_max", color = color) \
        .add(so.Line(), y = f"{y}_avg", color = color) \
        .label(x = x_label, y = y_label, title = title, legend=True) \
        .show()

    return

def plot_moving_average_multiple_files(files: dict[str, str]):
    dfs = []
    for file_label, file in files.items():
        file_df = pd.read_csv(file)
        file_df["Environment"] = file_label
        dfs.append(file_df)

    complete_df = pd.concat(dfs, ignore_index=True)
    plot_moving_average(complete_df, "timesteps_total", "episode_reward_mean", "Environment",
                        "Episode Reward Mean", "Timesteps", "Episode Reward", color = "Environment", window=5)

def plot_moving_average_single_file(file: str):
    df = pd.read_csv(file)
    df["_"] = "Multiagent"
    plot_moving_average(df, "timesteps_total", "episode_reward_mean", "_",
                        "Episode Reward Mean", "Timesteps", "Episode Reward", color = None, window=5)

if __name__ == "__main__":
    # CSV files with reward information
    leader_follower_file = "./results/save-leaderfollower-2-cc-kin-one_d_rpm-11.16.2024_14.50.34/PPO/PPO_this-aviary-v0_0_2024-11-16_14-50-35vplh6a_i/progress.csv"
    meet_at_height_file = "./results/save-meet_at_height-5-cc-kin-one_d_rpm-11.27.2024_14.31.48/PPO/PPO_meet_at_height-aviary-v0_0_2024-11-27_14-31-50ov97n_q_/progress.csv"
    meet_at_height_file_multiple_policies = "./results/save-meet_at_height-5-cc-kin-one_d_rpm-12.04.2024_11.02.48/PPO/PPO_meet_at_height-aviary-v0_0_2024-12-04_11-02-491vnhvzh2/progress.csv"
    chase_file = "./results/save-chase-2-cc-kin-one_d_rpm-12.11.2024_11.13.12/PPO/PPO_chase-aviary-v0_0_2024-12-11_11-13-140uousp1s/progress.csv"
    figure_file = "./results/save-figure-5-cc-kin-rpm-12.02.2024_19.46.09/PPO/PPO_figure-aviary-v0_0_2024-12-02_19-46-11btoteb77/progress.csv"

    ldfol_mah_files = {
        "Meet-at-height": meet_at_height_file,
        "Leader-follower": leader_follower_file,
    }

    meet_at_height_files = {
        "Meet-at-height 1 policy": meet_at_height_file,
        "Meet-at-height 5 policies": meet_at_height_file_multiple_policies,
    }

    chase_files = {
        "Meet-at-height": meet_at_height_file,
        "Leader-follower": leader_follower_file,
        "Chase": chase_file,
    }

    plot_moving_average_multiple_files(ldfol_mah_files)
    plot_moving_average_multiple_files(meet_at_height_files)
    plot_moving_average_single_file(leader_follower_file)
    plot_moving_average_multiple_files(chase_files)
    plot_moving_average_single_file(chase_file)
    plot_moving_average_single_file(figure_file)
