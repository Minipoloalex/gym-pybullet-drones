import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional


# Function to load and filter data
def load_data(folders, file_name, label_column, time_limit: Optional[float] = None):
    data_frames = []
    for label, folder in folders.items():
        file_path = f"{folder}/{file_name}"
        df = pd.read_csv(file_path, header=None)
        df[label_column] = label  # Add a column to label the source

        if time_limit is not None:
            df = df[df[df.columns[0]] <= time_limit]  # Filter by time (first column)

        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

# Function to plot multiple metrics in a grid
def plot_metrics(data_dict, rows, cols, hue_column, hue_values, colors, figsize=(12, 6)):
    # Create a figure with a specified number of rows and columns
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    axes = np.array(axes).flatten()  # Flatten axes for easier indexing

    # Loop through the metrics and corresponding axes
    for idx, (metric, data) in enumerate(data_dict.items()):
        if idx >= len(axes):  # If more metrics than plots
            raise ValueError("More metrics than plots")

        for hue in hue_values:
            color = colors[hue]
            hue_data = data[data[hue_column] == hue]
            axes[idx].plot(hue_data[hue_data.columns[0]], hue_data[hue_data.columns[1]], color=color)

        axes[idx].set_title(f"{metric} over Time")
        axes[idx].set_ylabel(metric)
        axes[idx].grid(True)

    fig.legend([hue for hue in hue_values], loc="lower center", ncol=len(hue_values))


    # Adjust layout
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing
    plt.show()

# Function to plot data
def plot_metric(data, y_label, title, hue_column, colors):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x=data.columns[0], y=data.columns[1], hue=hue_column, palette=colors)
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title=hue_column)
    plt.grid(True)
    plt.show()
