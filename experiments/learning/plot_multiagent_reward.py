import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

# Load the CSV file
file = "./results/save-leaderfollower-2-cc-kin-one_d_rpm-11.16.2024_14.50.34/PPO/PPO_this-aviary-v0_0_2024-11-16_14-50-35vplh6a_i/progress.csv"

df = pd.read_csv(file)

# Inspect the data
print(df.head())
print(df.columns)

def plot_moving_average(data, x, y, hue, title, x_label = None, y_label = None, window = 10):
    x_label = x_label if x_label is not None else x
    y_label = y_label if y_label is not None else y

    g = data.groupby([hue])[y]
    data[f"{y}_avg"] = g.rolling(window = window).mean().droplevel([0])
    data[f"{y}_max"] = g.rolling(window = window).max() .droplevel([0])
    data[f"{y}_min"] = g.rolling(window = window).min() .droplevel([0])

    so.Plot(data, x = x) \
        .add(so.Band(), ymin = f"{y}_min", ymax = f"{y}_max") \
        .add(so.Line(), y = f"{y}_avg") \
        .label(x = x_label, y = y_label, title = title, legend=False) \
        .show()

    return

df["_"] = "Multiagent"
plot_moving_average(df, "timesteps_total", "episode_reward_mean", "_",
                    "Episode Reward Mean", "Timesteps", "Episode Reward", window=5)
