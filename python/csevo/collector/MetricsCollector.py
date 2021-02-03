import collections
import os
import re
import traceback
from typing import *

import javalang
import numpy as np
from nltk.tokenize import word_tokenize
from scipy.stats import wilcoxon
from seutil import IOUtils, LoggingUtils
from tqdm import tqdm

from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.Utils import Utils


class MetricsCollector:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    # We only consider the years we actually used
    EVO_YEARS = [2017, 2018, 2019, 2020]
    # EVO_YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

    def __init__(self):
        self.output_dir = Macros.results_dir / "metrics"
        IOUtils.mk_dir(self.output_dir)
        return

    def collect_metrics(self, **options):
        which = Utils.get_option_as_list(options, "which")

        for item in which:
            self.logger.info(f"Collecting metrics: {item}; options: {options}")
            if item == "raw-dataset":
                self.collect_metrics_raw_dataset()
            elif item == "time-wise-datasest":
                # TODO: outdated (->archive)
                self.collect_metrics_timewise_dataset()
            elif item == "time-wise-filtered-dataset":
                # TODO: outdated (->archive)
                self.collect_metrics_filtered_dataset()
            elif item == "dataset":
                self.collect_metrics_dataset()
            elif item == "model-stat-results":
                self.collect_model_stat_results(options["model"])
            elif item == "stat-sign-test":
                self.collect_stat_sign_test(
                    options["output"],
                    Utils.get_option_as_list(options, "exps"),
                    Utils.get_option_as_list(options, "models"),
                    Utils.get_option_as_list(options, "metrics"),
                    options["test_set"],
                    options.get("confidence_level", 0.05),
                )
            else:
                self.logger.warning(f"No such metrics {item}")
            # end if
        # end for
        return

    def collect_stat_sign_test(
            self,
            output_name: str,
            exps: List[str],
            models: List[str],
            metrics: List[str],
            test_set: str,
            confidence_level: float = 0.05,
    ):
        """
        Performs statistical significance tests on the (same) test set, between:
          (1) each pair of models, for each metric and exp;
          (2) each pair of exps, for each metric and exp.

        Using paired Wilcoxon signed-ranked test.

        Generates a list of pairs of (exp, model, metric) that tested to be non significantly different.
        """
        assert output_name is not None
        assert exps is not None
        assert models is not None
        assert metrics is not None

        # Load experiment results
        model_2_results = {}
        for model in models:
            model_2_results[model] = IOUtils.load(self.output_dir / f"results-trials-{model}.json")

        # Get the mapping tuple(exp, model, metric) -> all_trials_results
        exp_model_metric_2_results = {}
        num_results = None
        for model, model_results in model_2_results.items():
            for exp, exp_results in model_results.items():
                for ts, set_results in exp_results.items():
                    # Only operate on the selected test set
                    if ts != test_set:
                        continue

                    for metric, trials_results in set_results.items():

                        # Merge results from all trials
                        all_trials_results = list()
                        bad_results = False
                        for trial_results in trials_results:
                            if trial_results is None:
                                self.logger.warning(f"Some trials for {model}, {exp}, {ts}, {metric} is missing. We can not analyze the results in this case")
                                bad_results = True
                                break
                            else:
                                all_trials_results += trial_results

                        if bad_results:
                            continue

                        if num_results is None:
                            num_results = len(all_trials_results)
                        else:
                            if len(all_trials_results) != num_results:
                                self.logger.warning(f"The number of results is inconsistent between models/exps, thus we can't perform paired test.")
                                self.logger.warning(f"Supposed to be {num_results}, but is {len(all_trials_results)} for {model}, {exp}, {metric}.")
                                raise RuntimeError

                        # Normalize NaN results
                        all_trials_results = [n if n != "NaN" else np.NaN for n in all_trials_results]

                        exp_model_metric_2_results[(exp, model, metric)] = all_trials_results

        no_diff_pairs = []

        # Compare each pair of models, for each metric and exp
        skipped = 0
        for metric in metrics:
            for exp in exps:
                for i_m1, m1 in enumerate(models):
                    results1 = exp_model_metric_2_results.get((exp, m1, metric))
                    for m2 in models[i_m1+1:]:
                        results2 = exp_model_metric_2_results.get((exp, m2, metric))
                        if results1 is None or results2 is None:
                            skipped += 1
                            continue

                        # Compute diff, with removing the NaN cases
                        diffs = [n1 - n2 for n1, n2 in zip(results1, results2) if n1 is not np.NaN and n2 is not np.NaN]

                        # Special case: all 0 (should not happen in practice)
                        if all(n == 0 for n in diffs):
                            self.logger.warning(f"{(exp, m1, metric)} and  {(exp, m2, metric)} are exactly the same")
                            no_diff_pairs.append(((exp, m1, metric), (exp, m2, metric), 1))
                            continue

                        # Do wilcoxon test
                        _, confidence = wilcoxon(diffs)
                        if confidence > confidence_level:
                            no_diff_pairs.append(((exp, m1, metric), (exp, m2, metric), confidence))

        if skipped > 0:
            self.logger.warning(f"Skipped {skipped} pairs of comparisons between models due to missing data")

        # Compare each pair of exps, for each metric and model
        skipped = 0
        for metric in metrics:
            for model in models:
                for i_e1, e1 in enumerate(exps):
                    results1 = exp_model_metric_2_results.get((e1, model, metric))
                    for e2 in exps[i_e1+1:]:
                        results2 = exp_model_metric_2_results.get((e2, model, metric))
                        if results1 is None or results2 is None:
                            skipped += 1
                            continue

                        # Compute diff, with removing the NaN cases
                        diffs = [n1 - n2 for n1, n2 in zip(results1, results2) if n1 is not np.NaN and n2 is not np.NaN]

                        # Special case: all 0 (should not happen in practice)
                        if all(n == 0 for n in diffs):
                            self.logger.warning(f"{(e1, model, metric)} and  {(e2, model, metric)} are exactly the same")
                            no_diff_pairs.append(((e1, model, metric), (e2, model, metric), 1))
                            continue

                        # Do wilcoxon test
                        _, confidence = wilcoxon(diffs)
                        if confidence > confidence_level:
                            no_diff_pairs.append(((e1, model, metric), (e2, model, metric), confidence))

        if skipped > 0:
            self.logger.warning(f"Skipped {skipped} pairs of comparisons between exps due to missing data")

        # Save results, in both json format and txt format
        sign_test_dir = self.output_dir / "sign-test"
        IOUtils.mk_dir(sign_test_dir)
        IOUtils.dump(sign_test_dir / f"{output_name}.json", no_diff_pairs)
        IOUtils.dump(
            sign_test_dir / f"{output_name}.txt",
            "".join([f"{ndp[0]}, {ndp[1]}, confidence {ndp[2]}\n" for ndp in no_diff_pairs]),
            IOUtils.Format.txt,
        )
        return

    def collect_metrics_dataset(self):
        eval_settings = ["mixedproj", "crossproj", "evo", "crossproj-evo"]
        for task in ["CG", "MN"]:
            # Load dataset after split
            data_type_2_data_list: Dict[str, List] = dict()
            for dt in ["2020-test_common"] + [
                f"{es}-2020-{t}"
                for es in eval_settings
                for t in [Macros.train, Macros.val, Macros.test_standard]
            ]:
                data_type_2_data_list[dt] = IOUtils.load(Macros.data_dir / "models-data" / f"{task}-shared" / f"{dt}.json", IOUtils.Format.json)

            data_type_2_data_list["2020"] = sum([data_type_2_data_list[f"mixedproj-2020-{t}"] for t in [Macros.train, Macros.val, Macros.test_standard]], [])
            data_type_2_data_list["2019-2020"] = sum([data_type_2_data_list[f"evo-2020-{t}"] for t in [Macros.train, Macros.val, Macros.test_standard]], [])

            # Load project list
            projects = IOUtils.load(Macros.data_dir / f"projects-github-{task}-100.json")

            # Load project split
            project_split = IOUtils.load(Macros.data_dir / f"projects-split-{task}-100.json")
            data_type_2_project_list: Dict[str, List] = {
                Macros.train: project_split["train"],
                Macros.val: project_split["val"],
                Macros.test: project_split["test"],
            }

            # Compute metrics
            metrics = dict()

            # Add some project metrics
            for data_type, project_list in data_type_2_project_list.items():
                metrics[f"num-proj_{data_type}"] = len(project_list)

            metrics[f"num-proj"] = len(projects)

            # Add other metrics on each data type
            for data_type, data_list in tqdm(data_type_2_data_list.items()):
                for name, number in self.collect_metrics_data_list(data_list).items():
                    metrics[f"{name}_{data_type}"] = number

            IOUtils.dump(self.output_dir / f"{task}-dataset.json", metrics)

    def collect_model_stat_results(self, model: str):
        results = IOUtils.load(self.output_dir / f"results-trials-{model}.json")
        stat_results = dict()

        for exp, exp_results in results.items():
            exp_stat_results = dict()
            for test_set, set_results in exp_results.items():
                set_stat_results = dict()
                for metric, trials_results in set_results.items():
                    metric_stat_results = dict()

                    # Merge the results from all trials
                    all_trials_results = list()
                    for trial_results in trials_results:
                        if trial_results is not None:
                            all_trials_results += [n for n in trial_results if n != np.NaN and n != "NaN"]

                    # Do statistics
                    for stat, func in Utils.SUMMARIES_FUNCS.items():
                        metric_stat_results[stat] = func(all_trials_results)

                    set_stat_results[metric] = metric_stat_results
                exp_stat_results[test_set] = set_stat_results
            stat_results[exp] = exp_stat_results

        IOUtils.dump(self.output_dir / f"results-stat-{model}.json", stat_results, IOUtils.Format.jsonPretty)
        return

    @staticmethod
    def collect_metrics_timewise_dataset(dataset: str = "large"):
        projects = [proj for proj in os.listdir(Macros.repos_results_dir)]
        YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        total_projs = 0
        time_stats = dict()
        for t in YEARS:
            time_stats[f"{t}_Jan_1"] = {
                "num-methods": 0,
                "num-projs": 0
            }
        for proj in projects:
            bad_time = 0
            revision_file = Macros.repos_results_dir / proj / "collector" / "method-project-revision.json"
            if not os.path.isfile(revision_file):
                MetricsCollector.logger.info(f"Project {proj} does not have revision data.")
                IOUtils.rm(Macros.repos_results_dir / proj)
                continue
            proj_revisions = IOUtils.load(revision_file)
            try:
                for t in proj_revisions:
                    if len(t["method_ids"]) > 0:
                        time_stats[t["year"]]["num-projs"] += 1
                        time_stats[t["year"]]["num-methods"] += len(t["method_ids"])
                    else:
                        bad_time += 1
                if bad_time == 8:
                    MetricsCollector.logger.info(f"Project {proj} does not have revision data.")
                    IOUtils.rm(Macros.repos_results_dir / proj)
                else:
                    total_projs += 1
            except:
                MetricsCollector.logger.info(f"{revision_file} can not load.")
                continue
        output_dir = Macros.results_dir / "metrics"
        for t in YEARS:
            if t == 2013:
                time_stats[f"{t}_Jan_1"]["delta"] = "N/A"
            else:
                time_stats[f"{t}_Jan_1"]["delta"] = time_stats[f"{t}_Jan_1"]["num-methods"] - \
                                                    time_stats[f"{t - 1}_Jan_1"]["num-methods"]
        IOUtils.dump(output_dir / f"time-wise-{dataset}-dataset-stats.json", time_stats, IOUtils.Format.jsonPretty)

    @staticmethod
    def collect_metrics_filtered_dataset(dataset: str = "large", filter: str = "alpha"):
        projects = [proj for proj in os.listdir(Macros.repos_results_dir)]
        YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        time_stats = dict()
        for t in YEARS[:-1]:
            time_stats[f"{t}_Jan_1-{t + 1}_Jan_1"] = {
                "num-methods": 0,
                "method-tokens": list(),
                "comment-tokens": list(),
            }
        for proj in projects:
            revision_diff_file = Macros.repos_results_dir / proj / "collector" / f"method-project-{filter}-filtered.json"
            method_data = Macros.repos_results_dir / proj / "collector" / "method-data.json"
            if not os.path.isfile(revision_diff_file):
                MetricsCollector.logger.info(f"Project {proj} does not have revision data.")
                IOUtils.rm(Macros.repos_results_dir / proj)
                continue
            proj_revisions_diff = IOUtils.load(revision_diff_file)
            method_data = IOUtils.load(method_data)
            try:
                for t in proj_revisions_diff:
                    time_stats[t["time"]]["num-methods"] += len(set(t["method_ids"]))
                    for m_id in t["method_ids"]:
                        method = method_data[m_id]
                        assert method["id"] == m_id
                        # tokenize code and comment
                        tokenized_code = list(javalang.tokenizer.tokenize(method["code"]))
                        tokenized_nl = " ".join(
                            word_tokenize(method["comment_summary"].replace("\n", " "))).lower().split()
                        time_stats[t["time"]]["method-tokens"].append(len(tokenized_code))
                        time_stats[t["time"]]["comment-tokens"].append(len(tokenized_nl))
            except:
                raise
        """
        # get stats based on time_stats
        for k, v in time_stats.items():
            v["method-tokens-avg"] = mean(v["method-tokens"])
            v["method-tokens-median"] = median(v["method-tokens"])
            v["method-tokens-mode"] = mode(v["method-tokens"])
            for i, length in enumerate(sorted(v["method-tokens"])):
                if length >= 100 and sorted(v["method-tokens"])[i - 1] < 100:
                    v["method-tokens-less-100"] = float(i / (len(v["method-tokens"]))) * 100
                if length >= 150 and sorted(v["method-tokens"])[i - 1] < 150:
                    v["method-tokens-less-150"] = float(i / len(v["method-tokens"])) * 100
                if length >= 200 and sorted(v["method-tokens"])[i - 1] < 200:
                    v["method-tokens-less-200"] = float(i / len(v["method-tokens"])) * 100
            v.pop("method-tokens")
            v["comment-tokens-avg"] = mean(v["comment-tokens"])
            v["comment-tokens-avg"] = fmean(v["comment-tokens"])
            v["comment-tokens-median"] = median(v["comment-tokens"])
            v["comment-tokens-mode"] = mode(v["comment-tokens"])
            for i, length in enumerate(sorted(v["comment-tokens"])):
                if length >= 20 and sorted(v["comment-tokens"])[i - 1] < 20:
                    v["comment-tokens-less-20"] = float(i / (len(v["comment-tokens"]))) * 100
                if length >= 30 and sorted(v["comment-tokens"])[i - 1] < 30:
                    v["comment-tokens-less-30"] = float(i / len(v["comment-tokens"])) * 100
                if length >= 50 and sorted(v["comment-tokens"])[i - 1] < 50:
                    v["comment-tokens-less-50"] = float(i / len(v["comment-tokens"])) * 100
            v.pop("comment-tokens")
        """
        output_dir = Macros.results_dir / "metrics"
        if filter == "beta":
            IOUtils.dump(output_dir / f"time-wise-{filter}-filtered-{dataset}-dataset-stats.json", time_stats,
                         IOUtils.Format.jsonPretty)
        IOUtils.dump(output_dir / f"time-wise-filtered-{dataset}-dataset-stats.json", time_stats, IOUtils.Format.jsonPretty)

    def collect_metrics_raw_dataset(self):
        metrics = dict()
        metrics["num-year"] = len(self.EVO_YEARS)

        # Load the raw dataset from _results
        data_type_2_data_list = collections.defaultdict(list)
        for task in ["CG", "MN"]:
            # Load project list
            projects = IOUtils.load(Macros.data_dir / f"projects-github-{task}-100.json")

            # Load unfiltered data & filters
            for proj in tqdm(projects):
                proj_collector_dir = Macros.repos_results_dir / proj / "collector"
                proj_data_list = IOUtils.load(proj_collector_dir / "method-data.json", IOUtils.Format.json)
                if task == "CG":
                    at_years_filters = IOUtils.load(proj_collector_dir / f"method-project-{task}-revision.json", IOUtils.Format.json)
                else:
                    at_years_filters = IOUtils.load(proj_collector_dir / f"method-project-revision.json", IOUtils.Format.json)
                all_indexes = set()
                for year in self.EVO_YEARS:
                    indexes = [yf["method_ids"] for yf in at_years_filters if yf["year"] == f"{year}_Jan_1"][0]
                    all_indexes |= set(indexes)
                    data_type_2_data_list[f"{year}"] += [proj_data_list[i] for i in indexes]

                data_type_2_data_list["all"] += [proj_data_list[i] for i in sorted(all_indexes)]

                new_years_filters = IOUtils.load(proj_collector_dir / f"method-project-{task}-filtered.json", IOUtils.Format.json)
                for year in self.EVO_YEARS[:-1]:
                    indexes = [yf["method_ids"] for yf in new_years_filters if yf["time"] == f"{year}_Jan_1-{year+1}_Jan_1"][0]
                    data_type_2_data_list[f"{year}-{year+1}"] += [proj_data_list[i] for i in indexes]

            # Add metrics on each data type
            for data_type, data_list in tqdm(data_type_2_data_list.items()):
                for name, number in self.collect_metrics_data_list(data_list).items():
                    metrics[f"{name}_{data_type}"] = number

            IOUtils.dump(self.output_dir / f"{task}-raw-dataset.json", metrics)

    @classmethod
    def length_code(cls, code: str) -> int:
        try:
            return len(list(javalang.tokenizer.tokenize(code)))
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            return np.NaN

    @classmethod
    def length_nl(cls, nl: str) -> int:
        try:
            return len(list(word_tokenize(nl)))
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            return np.NaN

    @classmethod
    def subtokenize_code(cls, tokens: List[str]) -> List[str]:
        """Subtokenize the code."""
        subtokens = list()
        for token in tokens:
            curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
            subtokens = subtokens + [c.lower() for c in curr]
        return subtokens

    @classmethod
    def length_name(cls, name: str) -> int:
        try:
            return len(cls.subtokenize_code([name]))
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            return np.NaN

    @classmethod
    def collect_metrics_data_list(cls, data_list) -> dict:
        metrics = dict()
        metrics[f"num-data"] = len(data_list)

        len_meth_list = [cls.length_code(d["code"]) for d in data_list]
        len_meth_list = [n for n in len_meth_list if n is not np.NaN]
        for s, func in Utils.SUMMARIES_FUNCS.items():
            metrics[f"len-meth-{s}"] = func(len_meth_list)

        for k in [100, 150, 200]:
            metrics[f"len-meth-le-{k}"] = 100 * (len([n for n in len_meth_list if n <= k]) / len(len_meth_list))

        len_com_list = [cls.length_nl(d["comment_summary"]) for d in data_list]
        len_com_list = [n for n in len_meth_list if n is not np.NaN]
        for s, func in Utils.SUMMARIES_FUNCS.items():
            metrics[f"len-com-{s}"] = func(len_com_list)

        for k in [20, 30, 50]:
            metrics[f"len-com-le-{k}"] = 100 * (len([n for n in len_com_list if n <= k]) / len(len_com_list))

        len_name_list = [cls.length_name(d["name"]) for d in data_list]
        len_name_list = [n for n in len_name_list if n is not np.NaN]
        for s, func in Utils.SUMMARIES_FUNCS.items():
            metrics[f"len-name-{s}"] = func(len_name_list)

        for k in [1, 2, 3, 4, 5, 6]:
            metrics[f"len-name-le-{k}"] = 100 * (len([n for n in len_name_list if n <= k]) / len(len_name_list))

        return metrics
