import numpy as np
import matplotlib.pyplot as plt
from typing import *
from pathlib import Path


class BoxPlot:

    TASK_2_METRICS = {
        "MN": ["f1", "precision", "recall"],
        "CG": ["bleu"]
    }

    def metric_plotter(self, models: List[str], eval_settings: List[str], task: str, data: dict, file_name: Path):
        """
        models: List[str]
        eval_sets: List[str]
        task: CG or MN
        data: { "model_name" : { "eval_setting": { "f1": List[float], .. }
        file_name: Path to save the figure
        """
        num_cols = len(models)
        num_rows = len(eval_settings)
        f, axs = plt.subplots(num_rows, num_cols, sharey=True)
        metrics = self.TASK_2_METRICS[task]
        for cur_row, model in enumerate(models):
            for cur_col, eval_setting in enumerate(eval_settings):
                data_per_plot = []
                label_per_plot = []
                for metric in metrics:
                    metric_array = np.array(data[model][eval_setting][metric])
                    data_per_plot.append(metric_array)
                    label_per_plot.append(metric)
                axs[cur_row, cur_col].boxplot(data_per_plot, labels=label_per_plot)
                # TODO: needs adjusted according to different boxplot (i.e. metrics or diffs)
                axs[cur_row, cur_col].set_ylim([0, 100])
                if cur_row == 0:
                    axs[cur_row, cur_col].set_title(eval_setting)
                if cur_col == 0:
                    axs[cur_row, cur_col].set_ylabel(model)
        plt.tight_layout()
        plt.savefig(file_name)



