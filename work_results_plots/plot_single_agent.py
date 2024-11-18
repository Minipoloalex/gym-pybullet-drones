import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plot_helper import load_data, plot_metric, plot_metrics

hues = ["A2C", "PPO", "SAC"]
folders = {
    "PPO": "~/Desktop/save-flight-sa-11.17.2024_11.47.38/",
    "SAC": "~/Desktop/save-flight-sa-11.17.2024_11.59.25/",
    "A2C": "~/Desktop/save-flight-sa-11.17.2024_12.32.30/",
}
colors = {
    "A2C": "blue",
    "SAC": "green",
    "PPO": "red",
}
files = {
    "Z": "z0.csv",
    "Vz": "vz0.csv",
    "RPM0": "rpm0-0.csv",
}

data_dict = {
    metric: load_data(folders, file_name, "Algorithm")
    for metric, file_name in files.items()
}


plot_metrics(data_dict, rows=3, cols=1, hue_column="Algorithm", hue_values=hues, colors=colors, figsize=(12, 9))
