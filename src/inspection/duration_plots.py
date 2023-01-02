from __future__ import annotations

from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt


def plot_avg_trial_durations(experiments, models):
    plot_multi_bar(models,
                   data_list=[{model: experiment.avg_trial_duration(model)[model] for model in models} for experiment in experiments],
                   xlabel="model names", ylabel="seconds", title="Average trial duration")


def plot_total_durations(experiments, models):
    plot_multi_bar(models,
                   data_list=[{model: experiment.total_run_duration(model)[model] for model in models} for experiment in experiments],
                   xlabel="model names", ylabel="seconds", title="Total run duration")


def plot_multi_bar(categories, data_list: List[Dict], xlabel=None, ylabel=None, title=None):
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab20")

    indices = np.arange(len(categories))
    width = np.min(np.diff(indices)) / (len(indices) + 1)

    for i, data, label in enumerate(zip(data_list, categories)):
        plt.bar(indices + width * i, list(data.values()),
                alpha=1,
                width=width,
                label=label,
                color=cmap(i))

    ax.set_xticks(indices + width)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.legend()
