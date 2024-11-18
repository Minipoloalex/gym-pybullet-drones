import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plot_helper import load_data, plot_metric, plot_metrics

hues = ["No Ground Effect", "Ground Effect"]
folders = {
    "Ground Effect": "~/Desktop/save-flight-gnd-11.16.2024_17.33.38/",
    "No Ground Effect": "~/Desktop/save-flight-gnd-11.16.2024_17.34.19/",
}
colors = {
    "Ground Effect": "red",
    "No Ground Effect": "blue",
}

files = {
    "Z": "z0.csv",
    "Vz": "vz0.csv",
    "RPM0": "rpm0-0.csv",
}

data_dict = {
    metric: load_data(folders, file_name, "Physics", time_limit=0.5)    # filtered by time <= 0.5 seconds
    for metric, file_name in files.items()
}

plot_metrics(data_dict, rows=3, cols=1, hue_column="Physics", hue_values=hues, colors=colors, figsize=(12, 9))
