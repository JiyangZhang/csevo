from typing import *

import collections
from pathlib import Path
import numpy as np

from seutil import IOUtils, LoggingUtils
from seutil import latex

from csevo.Environment import Environment
from csevo.Macros import Macros


class Table:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    COLSEP = "COLSEP"
    ROWSEP = "ROWSEP"

    SYMBOLS = [
        r"\alpha", r"\beta", r"\gamma", r"\delta",
        r"\epsilon", r"\zeta", r"\eta", r"\theta",
        r"\iota", r"\kappa", r"\lambda", r"\mu",
        r"\nu", r"\tau", r"\pi", r"\rho",
    ]

    def __init__(self):
        self.tables_dir: Path = Macros.paper_dir / "tables"
        IOUtils.mk_dir(self.tables_dir)

        self.metrics_dir: Path = Macros.results_dir / "metrics"
        return

    def make_tables(self, which, options):
        if len(which) == 1 and which[0] == "all":
            which = ["dataset-metrics"]
        # end if

        for item in which:
            if item == "dataset-metrics":
                self.make_numbers_dataset_metrics()
                self.make_table_dataset_metrics(version="main")
                self.make_table_dataset_metrics(version="split")
            elif item == "draft-model-results":
                # TODO: outdated (->remove)
                results_path = Path(options.get("results-path"))
                output_name = options.get("output-name")
                self.make_table_draft_model_results(results_path, output_name)
            elif item == "time-wise-dataset-metrics":
                # TODO: outdated (->archive)
                self.make_numbers_timewise_dataset_metrics()
                self.make_table_timewise_dataset_metrics()
            elif item == "time-wise-filtered-dataset-metrics":
                # TODO: outdated (->archive)
                self.make_numbers_timewise_filtered_dataset_metrics(options.get("dataset"), options.get("filter"))
                self.make_table_timewise_filtered_dataset_metrics(options.get("dataset"), options.get("filter"))
            elif item == "models-numbers":
                model = options.get("model")
                self.make_numbers_model_results(model)
            elif item == "evo-models-results":
                # TODO: outdated (->archive)
                self.make_table_evo_results()
            elif item == "models-results":
                self.make_table_models_results(options.get("task"))
            else:
                self.logger.warning(f"Unknown table name is {item}")
            # end if
        # end for
        return

    def make_table_timewise_filtered_dataset_metrics(self, dataset: str = "large", filter: str = "beta"):
        years = range(2013, 2020)
        t_diffs = [f"{t}_Jan_1-{t + 1}_Jan_1" for t in years]
        time_points = [f"{t}-{t + 1}" for t in years]
        # Header
        if filter == "beta":
            file = latex.File(self.tables_dir / (
                f"table-time-wise-{filter}-filtered-{dataset}-dataset-metrics.tex"))
            file.append(r"\begin{table*}")
            file.append(r"\begin{small}")
            file.append(r"\begin{center}")
            caption = f"Method naming statistics after filtering"
            file.append(r"\caption{" + caption + "}")
            file.append(r"\begin{tabular}{l | c }")
            file.append(r"\toprule")

            file.append(r" ")
            for m in ["num-methods"]:
                file.append(r" &")
                file.append(f"{m} ")
            file.append(r" \\")
            file.append(r"\midrule")
            for time, t in zip(t_diffs, time_points):
                file.append(f"{t}")
                file.append(" & " + latex.Macro(f"{dataset}-{filter}-{time}-num-methods").use())
                file.append(r"\\")
            # Footer
            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"\end{center}")
            file.append(r"\end{small}")
            file.append(r"\vspace{\TVDatasetMetrics}")
            file.append(r"\end{table*}")

            file.save()
            return
        for item in ["method", "comment"]:
            file = latex.File(self.tables_dir / (
                f"table-time-wise-{filter}-filtered-{item}-{dataset}-dataset-metrics.tex"))
            file.append(r"\begin{table*}")
            file.append(r"\begin{small}")
            file.append(r"\begin{center}")
            caption = f"{item} statistics after filtering"
            file.append(r"\caption{" + caption + "}")
            file.append(r"\begin{tabular}{l | c c c c c c c}")
            file.append(r"\toprule")
            if item == "method":
                file.append(r" ")
                for m in ["num-methods", "len-avg", "len-mode", "len-median", "len<100", "len<150", "len<200"]:
                    file.append(r" &")
                    file.append(f"{m} ")
                file.append(r" \\")
                file.append(r"\midrule")
                for time, t in zip(t_diffs, time_points):
                    file.append(f"{t}")
                    file.append(" & " + latex.Macro(f"{dataset}-{time}-num-methods").use())
                    for tvt in ["avg", "mode", "median", "less-100", "less-150", "less-200"]:
                        file.append(" & " + latex.Macro(f"{dataset}-{time}-method-tokens-{tvt}").use())
                    file.append(r"\\")
                # Footer
                file.append(r"\bottomrule")
                file.append(r"\end{tabular}")
                file.append(r"\end{center}")
                file.append(r"\end{small}")
                file.append(r"\vspace{\TVDatasetMetrics}")
                file.append(r"\end{table*}")

                file.save()
            elif item == "comment":
                file.append(r" ")
                for m in ["num-methods", "len-avg", "len-mode", "len-median", "len<20", "len<30", "len<50"]:
                    file.append(r" &")
                    file.append(f"{m} ")
                file.append(r" \\")
                file.append(r"\midrule")
                for time, t in zip(t_diffs, time_points):
                    file.append(f"{t}")
                    file.append(" & " + latex.Macro(f"{dataset}-{time}-num-methods").use())
                    for tvt in ["avg", "mode", "median", "less-20", "less-30", "less-50"]:
                        file.append(" & " + latex.Macro(f"{dataset}-{time}-{item}-tokens-{tvt}").use())
                    file.append(r"\\")
                # Footer
                file.append(r"\bottomrule")
                file.append(r"\end{tabular}")
                file.append(r"\end{center}")
                file.append(r"\end{small}")
                file.append(r"\vspace{\TVDatasetMetrics}")
                file.append(r"\end{table*}")

                file.save()
        return

    def make_table_timewise_dataset_metrics(self, dataset: str = "large"):
        file = latex.File(self.tables_dir / (
            f"table-time-wise-{dataset}-dataset-metrics.tex"))
        years = range(2013, 2021)
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\begin{small}")
        file.append(r"\begin{center}")
        caption = r"Dataset statistics " + dataset
        file.append(r"\caption{" + caption + "}")
        file.append(r"\begin{tabular}{l | r r r r r r r r}")
        file.append(r"\toprule")

        file.append(r"  &"
                    r"2013 & "
                    r"2014 & "
                    r"2015 & "
                    r"2016 & "
                    r"2017 & "
                    r"2018 & "
                    r"2019 & "
                    r"2020 \\")
        file.append(r"\midrule")

        for tvt in ["num-methods", "num-projs", "delta"]:
            file.append(f"{tvt}")
            for m in years:
                key = f"{dataset}-{m}_Jan_1-{tvt}"
                file.append(" & " + latex.Macro(key).use())
            # end for
            file.append(r"\\")
        # end for

        # Footer
        file.append(r"\bottomrule")
        file.append(r"\end{tabular}")
        file.append(r"\end{center}")
        file.append(r"\end{small}")
        file.append(r"\vspace{\TVDatasetMetrics}")
        file.append(r"\end{table*}")

        file.save()
        return

    def make_numbers_model_results(self, model: str):
        file = latex.File(self.tables_dir / f"numbers-{model}-results.tex")
        stat_results = IOUtils.load(Macros.results_dir / "metrics" / f"results-stat-{model}.json")

        for exp, exp_stat_results in stat_results.items():
            for test_set, set_stat_results in exp_stat_results.items():
                for metric, metric_stat_results in set_stat_results.items():
                    for stat, number in metric_stat_results.items():
                        macro_name = f"{exp}-{test_set}-{metric}-{model}-{stat}"
                        if number == np.NaN or number == "NaN":
                            macro_value = r"\Fix{NaN}"
                        else:
                            macro_value = f"{number:,.2f}"
                        file.append_macro(latex.Macro(macro_name, macro_value))

        file.save()
        return

    def make_numbers_timewise_dataset_metrics(self, dataset: str = "large"):
        file = latex.File(self.tables_dir / f"numbers-time-wise-{dataset}-dataset-metrics.tex")
        metrics = IOUtils.load(Macros.results_dir / "metrics" / f"time-wise-{dataset}-dataset-stats.json", IOUtils.Format.json)

        for t in metrics.keys():
            for k, v in metrics[t].items():
                file.append_macro(latex.Macro(f"{dataset}-{t}-{k}", f"{v}"))
        # end for

        file.save()
        return

    def make_numbers_timewise_filtered_dataset_metrics(self, dataset: str = "large", filter: str = "beta"):
        file = latex.File(self.tables_dir / f"numbers-time-wise-{filter}-filtered-{dataset}-dataset-metrics.tex")
        metrics = IOUtils.load(Macros.results_dir / "metrics" / f"time-wise-{filter}-filtered-{dataset}-dataset-stats.json",
                               IOUtils.Format.json)

        for t in metrics.keys():
            for k, v in metrics[t].items():
                if k == "num-methods":
                    file.append_macro(latex.Macro(f"{dataset}-{filter}-{t}-{k}", f"{v}"))
                # TODO: change back
                """
                else:
                    file.append_macro(latex.Macro(f"{dataset}-{filter}-{t}-{k}", "{:.1f}".format(v)))
                """
        # end for

        file.save()
        return

    def make_numbers_dataset_metrics(self):
        for task in Macros.tasks:
            file = latex.File(self.tables_dir / f"numbers-{task}-dataset-metrics.tex")

            dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / f"{task}-dataset.json", IOUtils.Format.json)
            for k, v in dataset_metrics.items():
                fmt = f",d" if type(v) == int else f",.2f"
                file.append_macro(latex.Macro(f"ds-{task}-{k}", f"{v:{fmt}}"))

            raw_dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / f"{task}-raw-dataset.json", IOUtils.Format.json)
            for k, v in raw_dataset_metrics.items():
                fmt = f",d" if type(v) == int else f",.2f"
                file.append_macro(latex.Macro(f"raw-ds-{task}-{k}", f"{v:{fmt}}"))

            file.save()
        return

    def make_table_models_results(self, task: str):
        if task == "ComGen":
            models = ["Seq2seq", "Seq2seqAtt", "DeepCom"]
            metrics = ["bleu", "xmatch"]
        elif task == "MethNam":
            models = ["Bi-LSTM", "no-split-Bi-LSTM", "Code2Seq"]
            metrics = ["f1", "precision", "recall", "xmatch"]
        else:
            raise ValueError(f"Invalid task {task}")
        exps = ["mixedproj-2020", "crossproj-2020", "evo-2020"]

        # Load stat sign test results
        no_diff_pairs = IOUtils.load(Macros.results_dir / "metrics" / "sign-test" / f"{task}.json")
        exp_model_metric_2_symbols = collections.defaultdict(list)
        for i, (emm1, emm2, _) in enumerate(no_diff_pairs):
            symbol = self.SYMBOLS[i]
            exp_model_metric_2_symbols[tuple(emm1)].append(symbol)
            exp_model_metric_2_symbols[tuple(emm2)].append(symbol)

        file = latex.File(self.tables_dir / f"table-{task}-models-results.tex")

        # Header
        file.append(r"\begin{table*}")
        file.append(r"\begin{small}")
        file.append(r"\begin{center}")
        table_name = f"Results{task}"
        caption = r"\TC" + table_name
        file.append(r"\caption{" + caption + "}")
        file.append(r"\begin{tabular}{l" + ("|" + "r"*len(metrics))*3 + "}")
        file.append(r"\toprule")

        # Line 1
        for i, exp in enumerate(exps):
            if i == len(exps) - 1:
                multicolumn = "c"
            else:
                multicolumn = "c|"
            file.append(r" & \multicolumn{" + f"{len(metrics)}" + r"}{" + multicolumn + r"}{\UseMacro{TH-exp-" + exp + r"}}")
        file.append(r"\\")

        # Line 2
        file.append(r"\multirow{-2}{*}{\THModel} ")
        for exp in exps:
            for metric in metrics:
                file.append(r" & \UseMacro{TH-metric-" + metric + r"}")
        file.append(r"\\")

        file.append(r"\midrule")

        for model in models:
            file.append(r"\UseMacro{TH-model-" + model + r"}")
            for exp in exps:
                for metric in metrics:
                    suffix = ""
                    symbols = exp_model_metric_2_symbols[(exp, model, metric)]
                    if len(symbols) > 0:
                        suffix = "$^{" + "".join(symbols) + "}$"
                    file.append(
                        r" & "
                        + latex.Macro(f"{exp}-test_common-{metric}-{model}-AVG").use()
                        + suffix
                    )
                                # + r"$\pm$"
                                # + latex.Macro(f"{exp}-test_common-{metric}-{model}-STDEV").use())
            file.append(r"\\")

        # Footer
        file.append(r"\bottomrule")
        file.append(r"\end{tabular}")
        file.append(r"\end{center}")
        file.append(r"\end{small}")
        file.append(r"\vspace{\TV" + table_name + r"}")
        file.append(r"\end{table*}")

        file.save()
        return

    def make_table_methd_name_results(self, task="Method-naming"):
        models = ["Bi-LSTM", "no-split-Bi-LSTM", "Code2Seq"]
        metrics = ["precision", "recall", "f1"]
        file = latex.File(self.tables_dir / f"table-{task}-models-results.tex")
        # evo results
        years = range(13, 18)
        # Header
        file.append(r"\begin{table*}")
        file.append(r"\begin{small}")
        file.append(r"\begin{center}")
        caption = f"{task} results"
        file.append(r"\caption{" + caption + "}")
        # \begin{tabular}{l | c | c |c |c |c}
        coll = r"\begin{tabular}{l"
        for i in range(len(models)*3):
            coll += "|c"
        coll += "}"
        file.append(coll)
        file.append(r"\toprule")

        file.append(r" \multirow{2}{*}{Time-Metrics}")
        for m in models:
            file.append(r"& \multicolumn{3}{c}"+f"{{{m}}}")
        file.append(r"\\")
        for i in range(len(models)):
            for metric in metrics:
                file.append(f"& {metric}")
        file.append(r"\\")
        file.append(r"\midrule")
        for t in years:
            file.append(f"20{t}-20{t + 1}-train")
            for m in models:
                for metric in metrics:
                    m = m.lower()
                    key = f"{m.lower()}-{t}{t + 1}-train-{metric}"
                    file.append(" & " + latex.Macro(key).use())
            file.append(r"\\")
            # end for
            # end for
        # end for

        file.append(f"latest-mixed")
        for m in models:
            for metric in metrics:
                m = m.lower()
                key = f"{m.lower()}-latest-{metric}"
                file.append(" & " + latex.Macro(key).use())
        file.append(r"\\")
        # end for

        file.append(f"latest-cross-project")
        for m in models:
            for metric in metrics:
                m = m.lower()
                key = f"{m.lower()}-cross-proj-latest-{metric}"
                file.append(" & " + latex.Macro(key).use())
        file.append(r"\\")
        # end for

        # Footer
        file.append(r"\bottomrule")
        file.append(r"\end{tabular}")
        file.append(r"\end{center}")
        file.append(r"\end{small}")
        file.append(r"\vspace{\TVDatasetMetrics}")
        file.append(r"\end{table*}")

        file.save()
        return

    def make_table_dataset_metrics(self, version: str):
        for task in Macros.tasks:
            if version == "main":
                file = latex.File(self.tables_dir / f"table-{task}-dataset-metrics-main.tex")
            elif version == "split":
                file = latex.File(self.tables_dir / f"table-{task}-dataset-metrics-split.tex")
            else:
                raise ValueError(f"Invalid version {version}")

            metric_2_th = collections.OrderedDict()
            metric_2_th["num-proj"] = r"\multicolumn{2}{c|}{\UseMacro{TH-ds-num-project}}"
            metric_2_th["num-data"] = r"\multicolumn{2}{c|}{\UseMacro{TH-ds-num-data}}"
            metric_2_th["len-meth-AVG"] = r"& \UseMacro{TH-ds-len-method-avg}"
            metric_2_th["len-meth-MODE"] = r"& \UseMacro{TH-ds-len-method-mode}"
            metric_2_th["len-meth-MEDIAN"] = r"& \UseMacro{TH-ds-len-method-median}"
            metric_2_th["len-meth-le-100"] = r"& \UseMacro{TH-ds-len-method-le100}"
            metric_2_th["len-meth-le-150"] = r"& \UseMacro{TH-ds-len-method-le150}"
            metric_2_th["len-meth-le-200"] = r"\multirow{-6}{*}{\UseMacro{TH-ds-len-method}} & \UseMacro{TH-ds-len-method-le200}"
            metric_2_th["len-com-AVG"] = r"& \UseMacro{TH-ds-len-comment-avg}"
            metric_2_th["len-com-MODE"] = r"& \UseMacro{TH-ds-len-comment-mode}"
            metric_2_th["len-com-MEDIAN"] = r"& \UseMacro{TH-ds-len-comment-median}"
            metric_2_th["len-com-le-20"] = r"& \UseMacro{TH-ds-len-comment-le20}"
            metric_2_th["len-com-le-30"] = r"& \UseMacro{TH-ds-len-comment-le30}"
            metric_2_th["len-com-le-50"] = r"\multirow{-6}{*}{\UseMacro{TH-ds-len-comment}} & \UseMacro{TH-ds-len-comment-le50}"
            metric_2_th["len-name-AVG"] = r"& \UseMacro{TH-ds-len-name-avg}"
            metric_2_th["len-name-MODE"] = r"& \UseMacro{TH-ds-len-name-mode}"
            metric_2_th["len-name-MEDIAN"] = r"& \UseMacro{TH-ds-len-name-median}"
            metric_2_th["len-name-le-3"] = r"& \UseMacro{TH-ds-len-name-le2}"
            metric_2_th["len-name-le-5"] = r"& \UseMacro{TH-ds-len-name-le3}"
            metric_2_th["len-name-le-6"] = r"\multirow{-6}{*}{\UseMacro{TH-ds-len-name}} & \UseMacro{TH-ds-len-name-le6}"

            sep_after_rows = [
                "num-data",
                "len-meth-le-200",
                "len-com-le-50",
            ]

            dt_2_is_raw = collections.OrderedDict()

            if version == "main":
                dt_2_is_raw["all"] = True
                dt_2_is_raw["2020"] = False
                dt_2_is_raw["2019-2020"] = False

                sep_after_cols = []
            elif version == "split":
                for exp in ["mixedproj", "crossproj", "evo"]:
                    for dt in [Macros.train, Macros.val]:
                        dt_2_is_raw[f"{exp}-2020-{dt}"] = False
                dt_2_is_raw[f"2020-{Macros.test_common}"] = False

                sep_after_cols = [
                    f"mixedproj-2020-{Macros.val}",
                    f"crossproj-2020-{Macros.val}",
                ]
            else:
                raise ValueError(f"Invalid version {version}")

            # Header
            file.append(r"\begin{" + ("table*" if version == "split" else "table") + "}")
            file.append(r"\begin{small}")
            file.append(r"\begin{center}")

            if version == "main":
                table_name = "DatasetMetricsMain"
            elif version == "split":
                table_name = "DatasetMetricsSplit"
            else:
                raise ValueError(f"Invalid version {version}")

            file.append(r"\caption{\TC" + table_name + "}")

            if version == "main":
                file.append(r"\begin{tabular}{ l@{\hspace{2pt}}|@{\hspace{2pt}}c@{\hspace{2pt}} | r r r}")
            elif version == "split":
                file.append(
                    r"\begin{tabular}{ l@{\hspace{2pt}}|@{\hspace{2pt}}c@{\hspace{2pt}} | rr @{\hspace{5pt}}c@{\hspace{5pt}} rr @{\hspace{5pt}}c@{\hspace{5pt}} rr r}")
            else:
                raise ValueError(f"Invalid version {version}")

            file.append(r"\toprule")

            if version == "main":
                # Line 1
                file.append(
                    r"\multicolumn{2}{c|}{} & & & \\"
                )

                # Line 2
                file.append(
                    r"\multicolumn{2}{c|}{\multirow{-2}{*}{\THDSStat}} & \multirow{-2}{*}{\UseMacro{TH-ds-all}} & \multirow{-2}{*}{\UseMacro{TH-ds-2020}} & \multirow{-2}{*}{\UseMacro{TH-ds-2019-2020}} \\"
                )
            elif version == "split":
                # Line 1
                file.append(
                    r"\multicolumn{2}{c|}{}"
                    r" & \multicolumn{2}{c}{\UseMacro{TH-ds-mixedproj}} &"
                    r" & \multicolumn{2}{c}{\UseMacro{TH-ds-crossproj}} &"
                    r" & \multicolumn{2}{c}{\UseMacro{TH-ds-evo}}"
                    r" & \\\cline{3-4}\cline{6-7}\cline{9-10}"
                )

                # Line 2
                file.append(
                    r"\multicolumn{2}{c|}{\multirow{-2}{*}{\THDSStat}}"
                    r" & \UseMacro{TH-ds-mixedproj-train} & \UseMacro{TH-ds-mixedproj-val} &"
                    r" & \UseMacro{TH-ds-crossproj-train} & \UseMacro{TH-ds-crossproj-val} &"
                    r" & \UseMacro{TH-ds-evo-train} & \UseMacro{TH-ds-evo-val}"
                    r" & \multirow{-2}{*}{\UseMacro{TH-ds-test}} \\"
                )
            else:
                raise ValueError(f"Invalid version {version}")

            file.append(r"\midrule")

            for metric, row_th in metric_2_th.items():
                file.append(row_th)

                for dt, is_raw in dt_2_is_raw.items():
                    if metric == "num-proj":
                        if dt == f"crossproj-2020-{Macros.train}":
                            macro_name = f"ds-{task}-num-proj_{Macros.train}"
                        elif dt == f"crossproj-2020-{Macros.val}":
                            macro_name = f"ds-{task}-num-proj_{Macros.val}"
                        elif dt == f"2020-{Macros.test_common}":
                            macro_name = f"ds-{task}-num-proj_{Macros.test}"
                        else:
                            macro_name = f"ds-{task}-num-proj"
                    elif is_raw:
                        macro_name = f"raw-ds-{task}-{metric}_{dt}"
                    else:
                        macro_name = f"ds-{task}-{metric}_{dt}"

                    file.append(" & " + latex.Macro(macro_name).use())

                    if dt in sep_after_cols:
                        file.append(" & ")

                file.append(r"\\")

                if metric in sep_after_rows:
                    file.append(r"\midrule")

            # Footer
            file.append(r"\bottomrule")
            file.append(r"\end{tabular}")
            file.append(r"\end{center}")
            file.append(r"\end{small}")
            file.append(r"\vspace{\TV" + table_name + "}")
            file.append(r"\end{" + ("table*" if version == "split" else "table") + "}")

            file.save()
        return

    def make_table_draft_model_results(self,
                                       results_path: Path,
                                       output_name: str,
                                       ):
        special_tables_dir = self.tables_dir / "draft-model-results"
        IOUtils.mk_dir(special_tables_dir)
        file = latex.File(special_tables_dir / f"{output_name}.tex")

        # Header
        file.append(r"\begin{table*}")
        file.append(r"\begin{small}")
        file.append(r"\begin{center}")
        file.append(r"\caption{Model Results (Draft) from " + str(results_path).replace("_", r"\_") + "}")

        metrics = None
        for tvt in [Macros.lat_lat, Macros.evo_lat, Macros.lat_evo, Macros.evo_evo]:
            results = IOUtils.load(results_path / tvt / "test_results.json")

            # Flatten Rouge scores
            if "Rouge" in results:
                if results["Rouge"] == 0:
                    results["Rouge1-F1"] = 0
                    results["Rouge2-F1"] = 0
                    results["RougeL-F1"] = 0
                else:
                    results["Rouge1-F1"] = results["Rouge"]["rouge-1"]["f"]
                    results["Rouge2-F1"] = results["Rouge"]["rouge-2"]["f"]
                    results["RougeL-F1"] = results["Rouge"]["rouge-l"]["f"]
                # end if
                del results["Rouge"]
            # end if

            if metrics is None:
                metrics = list(sorted(results.keys()))

                # Table header line
                file.append(r"\begin{tabular}{l | " + "r" * len(metrics) + "}")
                file.append(r"\toprule")
                file.append("Training-Testing & " + " & ".join(metrics) + r"\\")
                file.append(r"\midrule")
            # end if

            file.append(tvt)
            for m in metrics:
                file.append(f"& {results[m]:.2f}")
            # end for
            file.append(r"\\")
        # end for

        # Footer
        file.append(r"\bottomrule")
        file.append(r"\end{tabular}")
        file.append(r"\end{center}")
        file.append(r"\end{small}")
        file.append(r"\end{table*}")

        file.save()
        return
