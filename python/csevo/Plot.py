from pathlib import Path
from typing import *

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from seutil import IOUtils, LoggingUtils, MiscUtils

from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.Utils import Utils


class Plot:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    TASK_2_MODELS = {
        "ComGen": ["Seq2seq", "Seq2seqAtt", "DeepCom"],
        "MethNam": ["Bi-LSTM", "no-split-Bi-LSTM", "Code2Seq"],
    }

    TASK_2_METRICS = {
        "ComGen": ["bleu", "xmatch"],
        "MethNam": ["f1", "precision", "recall", "xmatch"],
    }

    MODEL_2_NAME = {
        "Seq2seq": "Seq2seq",
        "Seq2seqAtt": "Seq2seqAtt",
        "DeepCom": "Hybrid-DeepCom",
        "Bi-LSTM": "Bi-LSTM",
        "no-split-Bi-LSTM": "Bi-LSTM(no-split)",
        "Code2Seq": "Code2Seq",
    }

    EXP_2_NAME = {
        "mixedproj-2020": "Mixed-project",
        "crossproj-2020": "Cross-project",
        "evo-2020": "Temporally-segmented",
    }

    METRIC_2_NAME = {
        "bleu": "BLEU",
        "xmatch": "xMatch",
        "f1": "F1",
        "precision": "Prec",
        "recall": "Rec",
    }

    EXPS = ["mixedproj-2020", "crossproj-2020", "evo-2020"]

    EVO_YEARS = [2017, 2018, 2019, 2020]

    def __init__(self):
        self.plots_dir: Path = Macros.paper_dir / "figs"
        IOUtils.mk_dir(self.plots_dir)

        # Initialize seaborn
        sns.set()
        sns.set_palette("Dark2")
        sns.set_context("paper")
        mpl.rcParams["axes.titlesize"] = 24
        mpl.rcParams["axes.labelsize"] = 24
        mpl.rcParams["font.size"] = 18
        mpl.rcParams["xtick.labelsize"] = 24
        mpl.rcParams["xtick.major.size"] = 14
        mpl.rcParams["xtick.minor.size"] = 14
        mpl.rcParams["ytick.labelsize"] = 24
        mpl.rcParams["ytick.major.size"] = 14
        mpl.rcParams["ytick.minor.size"] = 14
        mpl.rcParams["legend.fontsize"] = 18
        mpl.rcParams["legend.title_fontsize"] = 18
        # print(mpl.rcParams)
        return

    def make_plots(self, which, options: dict):
        for item in which:
            if item == "draft-learning-curve":
                # TODO: outdated (->remove)
                training_log_path = Path(options.get("training-log-path"))
                output_name = options.get("output-name")
                self.make_plot_draft_learning_curve(training_log_path, output_name)
            elif item == "models-results-metrics-dist":
                task = options["task"]
                models = Utils.get_option_as_list(options, "models", self.TASK_2_MODELS.get(task))
                metrics = Utils.get_option_as_list(options, "metrics", self.TASK_2_METRICS.get(task))
                exps = Utils.get_option_as_list(options, "exps", self.EXPS)
                self.plot_models_results_metrics_dist(task, models, metrics, exps)
            elif item == "models-results-variance-dist":
                task = options["task"]
                models = Utils.get_option_as_list(options, "models", self.TASK_2_MODELS.get(task))
                metrics = Utils.get_option_as_list(options, "metrics", self.TASK_2_METRICS.get(task))
                exps = Utils.get_option_as_list(options, "exps", self.EXPS)
                self.plot_models_results_variance_dist(task, models, metrics, exps)
            elif item == "num-data-evolution":
                self.plot_num_data_evolution(
                    Utils.get_option_as_list(options, "years", self.EVO_YEARS),
                )
            else:
                self.logger.warning(f"Unknown plot {item}")
            # end if
        # end for
        return

    def plot_num_data_evolution(self, years: List[int]):
        # Load data
        raw_dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / "raw-dataset.json", IOUtils.Format.json)

        data = dict()
        for i, year in enumerate(years):
            data[len(data)] = {
                "year": year,
                "new": False,
                "num-data": raw_dataset_metrics[f"num-data_{year}"],
            }

            if i != 0:
                data[len(data)] = {
                    "year": year,
                    "new": True,
                    "num-data": raw_dataset_metrics[f"num-data_{year-1}-{year}"],
                }

        df = pd.DataFrame.from_dict(data, orient="index")

        # Do the plot
        fig: plt.Figure = plt.figure(figsize=(15, 9))
        hue_order = [False, True]
        ax: plt.Axes = sns.barplot(
            data=df,
            x="year",
            order=years,
            y="num-data",
            hue="new",
            hue_order=hue_order,
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("#Meth")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: MiscUtils.itos_human_readable(x, 2)))

        labels = hue_order
        colors = sns.color_palette().as_hex()[:len(labels)]
        handles = [Rectangle(xy=(0, 0), width=1.8, height=1, color=col) for col in colors]
        ax.legend(
            handles=handles,
            labels=[r"$\mathcal{D}(\mathcal{P}, \tau)$", r"$\mathcal{D}(\mathcal{P}, \tau-1, \tau)$"],
            title="",
        )

        fig.tight_layout()
        fig.savefig(self.plots_dir / "num-data-evolution.eps")

    @classmethod
    def get_model_results_all_trials(cls, model: str) -> Dict[str, Dict[str, list]]:
        """
        Gets the model's results, on each exp, of each metric, on test_common set,
        combining all trials.

        Returns:
            mapping of exp -> (mapping of metric -> list of results)
        """
        results = IOUtils.load(Macros.results_dir / "metrics" / f"results-trials-{model}.json")
        results_all_trials = dict()

        for exp, exp_results in results.items():
            exp_results_all_trials = dict()
            for test_set, set_results in exp_results.items():
                # Only use test_common set
                if test_set != Macros.test_common:
                    continue

                for metric, trials_results in set_results.items():
                    metric_results_all_trials = list()

                    # Merge the results from all trials
                    for trial_results in trials_results:
                        if trial_results is not None:
                            metric_results_all_trials += [n for n in trial_results if n != np.NaN and n != "NaN"]

                    exp_results_all_trials[metric] = metric_results_all_trials
            results_all_trials[exp] = exp_results_all_trials

        return results_all_trials

    @classmethod
    def get_model_results_variance(cls, model: str) -> Dict[str, Dict[str, list]]:
        """
        Gets the model's results variance between trials, on each exp, of each metric, on test_common set.

        Suppose one example got results = [a, b, c] on three trials,
        the variance is defined as max(results) - min(results).

        Returns:
            mapping of exp -> (mapping of metric -> list of variances)
        """
        results = IOUtils.load(Macros.results_dir / "metrics" / f"results-trials-{model}.json")
        variances = dict()

        for exp, exp_results in results.items():
            exp_variances = dict()
            for test_set, set_results in exp_results.items():
                # Only use test_common set
                if test_set != Macros.test_common:
                    continue

                for metric, trials_results in set_results.items():
                    metric_variances = list()

                    # Compute the variances per example
                    for i in range(len(trials_results[0])):
                        results_i = [trial_results[i] for trial_results in trials_results]

                        # Don't work on NaN results
                        if np.NaN in results_i or "NaN" in results_i:
                            continue

                        metric_variances.append(max(results_i) - min(results_i))
                    exp_variances[metric] = metric_variances
            variances[exp] = exp_variances

        return variances

    def plot_models_results_metrics_dist(self, task: str, models: List[str], metrics: List[str], exps: List[str]):
        # Load data
        data = dict()
        for model in models:
            model_results = self.get_model_results_all_trials(model)
            for exp in exps:
                for metric in metrics:
                    results = model_results[exp][metric]
                    for result in results:
                        data[len(data)] = {
                            "exp": exp,
                            "metric": metric,
                            "model": model,
                            "result": result,
                        }

        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.replace({
            "exp": self.EXP_2_NAME,
            "metric": self.METRIC_2_NAME,
            "model": self.MODEL_2_NAME,
        })
        exps_names = [self.EXP_2_NAME[x] for x in exps]
        metrics_names = [self.METRIC_2_NAME[x] for x in metrics]
        models_names = [self.MODEL_2_NAME[x] for x in models]

        # print(df)

        # Do the plot
        # Hack: draw axes labels later
        labelsize = mpl.rcParams["axes.labelsize"]
        mpl.rcParams["axes.labelsize"] = 0
        g = sns.catplot(
            data=df,
            x="result",
            y="metric",
            order=metrics_names,
            col="exp",
            col_order=exps_names,
            row="model",
            row_order=models_names,
            # hue="metric",
            # hue_order=metrics,
            sharex=True,
            kind="violin",
            legend=True,
            legend_out=True,
            margin_titles=True,
            height=2 * len(metrics),
            aspect=3.6 / len(metrics),
            kwargs=dict(
                cut=0,
                width=2,
                bw=.05,
                linewidth=0.1,
            ),
            facet_kws=dict(
                gridspec_kws=dict(
                    hspace=0.15,
                )
            )
        )
        mpl.rcParams["axes.labelsize"] = labelsize

        g = g.set(xlim=(0, 100))
        g._margin_titles = True
        g = g.set_titles(
            template="",
            row_template="{row_name}",
            col_template="{col_name}",
        )
        g = g.set_xlabels("Metric", size=labelsize)
        g = g.set_ylabels("")

        g.savefig(self.plots_dir / f"models-results-metrics-dist-{task}.eps")

    def plot_models_results_variance_dist(self, task: str, models: List[str], metrics: List[str], exps: List[str]):
        # Load data
        data = dict()
        for model in models:
            model_results = self.get_model_results_variance(model)
            for exp in exps:
                for metric in metrics:
                    results = model_results[exp][metric]
                    for result in results:
                        data[len(data)] = {
                            "exp": exp,
                            "metric": metric,
                            "model": model,
                            "result": result,
                        }

        df = pd.DataFrame.from_dict(data, orient="index")
        df = df.replace({
            "exp": self.EXP_2_NAME,
            "metric": self.METRIC_2_NAME,
            "model": self.MODEL_2_NAME,
        })
        exps_names = [self.EXP_2_NAME[x] for x in exps]
        metrics_names = [self.METRIC_2_NAME[x] for x in metrics]
        models_names = [self.MODEL_2_NAME[x] for x in models]

        # print(df)

        # Do the plot
        # Hack: draw axes labels later
        labelsize = mpl.rcParams["axes.labelsize"]
        mpl.rcParams["axes.labelsize"] = 0
        g = sns.catplot(
            data=df,
            x="result",
            y="metric",
            order=metrics_names,
            col="exp",
            col_order=exps_names,
            row="model",
            row_order=models_names,
            # hue="metric",
            # hue_order=metrics,
            sharex=True,
            kind="violin",
            legend=True,
            legend_out=True,
            margin_titles=True,
            height=2 * len(metrics),
            aspect=3.6 / len(metrics),
            kwargs=dict(
                cut=0,
                width=2,
                bw=.05,
                linewidth=0.1,
            ),
            facet_kws=dict(
                gridspec_kws=dict(
                    hspace=0.15,
                )
            )
        )
        mpl.rcParams["axes.labelsize"] = labelsize

        g = g.set(xlim=(0, 100))
        g._margin_titles = True
        g = g.set_titles(
            template="",
            row_template="{row_name}",
            col_template="{col_name}",
        )
        g = g.set_xlabels("Metric Variance", size=labelsize)
        g = g.set_ylabels("")

        g.savefig(self.plots_dir / f"models-results-variance-dist-{task}.eps")

    def make_plot_draft_learning_curve(self,
            training_log_path: Path,
            output_name: str,
    ):
        special_plots_dir = self.plots_dir / "draft-learning-curve"
        IOUtils.mk_dir(special_plots_dir)

        fig: plt.Figure = plt.figure(figsize=(12,9))

        # TODO: these metrics may be specific to Code2Seq only
        x_field = "batch"
        yl_field = "training_loss"
        yr_field = "eval F1"

        x_min = 0
        x_max = -np.Inf
        yl_min = np.Inf
        yl_max = -np.Inf
        yr_min = np.Inf
        yr_max = -np.Inf

        # First, get ranges for all metrics (we want to use same ranges in all subplots)
        tvt_2_training_log = dict()
        tvt_2_x = dict()
        tvt_2_yl = dict()
        tvt_2_yr = dict()

        for tvt in [Macros.lat_lat, Macros.evo_lat, Macros.lat_evo, Macros.evo_evo]:
            # TODO: this path is hardcoded and work for Code2Seq 1 trial
            training_log = IOUtils.load(training_log_path / tvt / "trial-0" / "logs" / "train_log.json", IOUtils.Format.json)
            x = [d[x_field] for d in training_log]
            yl = [d[yl_field] for d in training_log]
            yr = [d[yr_field] for d in training_log]

            tvt_2_training_log[tvt] = training_log
            tvt_2_x[tvt] = x
            tvt_2_yl[tvt] = yl
            tvt_2_yr[tvt] = yr

            x_min = min(x_min, min(x))
            x_max = max(x_max, max(x))
            yl_min = min(yl_min, min(yl))
            yl_max = max(yl_max, max(yl))
            yr_min = min(yr_min, min(yr))
            yr_max = max(yr_max, max(yr))
        # end for

        x_lim = (x_min - (x_max - x_min) / 30, x_max + (x_max - x_min) / 30)
        yl_lim = (np.exp(np.log(yl_min) - (np.log(yl_max) - np.log(yl_min)) / 30), np.exp(np.log(yl_max) + (np.log(yl_max) - np.log(yl_min)) / 30))
        yr_lim = (yr_min - (yr_max - yr_min) / 30, yr_max + (yr_max - yr_min) / 30)

        for t_i, t in enumerate([Macros.lat, Macros.evo]):
            for vt_i, vt in enumerate([Macros.lat, Macros.evo]):
                tvt = f"{t}-{vt}"
                tvt_i = (t_i)*2+(vt_i)+1

                x = tvt_2_x[tvt]
                yl = tvt_2_yl[tvt]
                yr = tvt_2_yr[tvt]

                axl: plt.Axes = fig.add_subplot(2, 2, tvt_i)
                axr = axl.twinx()

                colorl = "tab:red"
                colorr = "tab:blue"

                axl.plot(x, yl, color=colorl)
                axr.plot(x, yr, color=colorr)

                axl.set_xlabel(x_field)
                axl.set_xlim(x_lim[0], x_lim[1])

                axl.set_ylabel(yl_field, color=colorl)
                axl.set_yscale("log")
                axl.set_ylim(yl_lim[0], yl_lim[1])

                axr.set_ylabel(yr_field, color=colorr)
                axr.set_ylim(yr_lim[0], yr_lim[1])

                axl.set_title(tvt)
            # end for
        # end for

        fig.tight_layout()
        with IOUtils.cd(special_plots_dir):
            fig.savefig(f"{output_name}.eps")
        # end with
        return
