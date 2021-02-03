from typing import *

from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.eval.metrics import bleu, f1, precision, recall, xmatch
import numpy as np
from seutil import LoggingUtils, IOUtils


class ModelResultsCollector:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    TASK_2_METRICS = {
        "methd-name": ["f1", "precision", "recall", "xmatch"],
        "com-gen": ["bleu", "xmatch"]
    }

    EVAL_SETTINGS = ["evo", "crossproj", "mixedproj"]
    YEARS = [2020]

    MODELS_DATA_MAPPINGS = {
        "Seq2seq": "DeepCom",
        "Seq2seqAtt": "DeepCom",
    }

    MODELS_REFS_REL_PATH = {
        "DeepCom": "test/test.token.nl",
        "Seq2seq": "test/test.token.nl",
        "Seq2seqAtt": "test/test.token.nl",
        "Bi-LSTM": "tgt-test.txt",
        "no-split-Bi-LSTM": "tgt-test.txt",
        "Code2Seq": "ref.txt"
    }

    METRIC_2_FUNC: Dict[str, Callable[[List[str], List[str]], Any]] = {
        "bleu": bleu,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "xmatch": xmatch,
    }

    def collect_results(self, model: str, task: str, re_eval: bool = False):
        if re_eval:
            self.eval_using_outputs(model, self.TASK_2_METRICS[task])
        self.collect_all_results(model, self.TASK_2_METRICS[task])

    def eval_using_outputs(self, model: str, metrics: List[str]):
        """
        For all the eval-settings and years, performs the per-data evaluation on the outputs.
        Reads the outputs from output_test_{common,standard}.txt files and
        generates predictions to results_test_{common,standard}.json files.
        """
        model_work_dir = Macros.data_dir / "models-work" / model
        # The data for this model is saved at the directory with the same name under models-data, unless otherwise
        # redirected as specified as MODELS_DATA_MAPPINGS
        model_data_dir = Macros.data_dir / "models-data" / self.MODELS_DATA_MAPPINGS.get(model, model)
        for eval_setting in self.EVAL_SETTINGS:
            for year in self.YEARS:
                exp = f"{eval_setting}-{year}"
                for trial in range(Macros.trials):
                    trial_dir = model_work_dir / exp / f"trial-{trial}"
                    for test_set in [Macros.test_common, Macros.test_standard]:
                        cur_results = dict()

                        output_file = trial_dir / f"output_{test_set}.txt"
                        if not output_file.exists():
                            self.logger.warning(f"Not found output file {output_file}")
                            continue
                        output_str = IOUtils.load(output_file, IOUtils.Format.txt)
                        outputs = [s.split() for s in output_str.splitlines()]

                        ref_file = model_data_dir / f"{exp}-{test_set}" / self.MODELS_REFS_REL_PATH[model]
                        if model == "Code2Seq": ref_file = trial_dir / f"ref_{test_set}.txt"
                        if not ref_file.exists():
                            self.logger.warning(f"Not found ref file {ref_file}")
                            continue
                        ref_str = IOUtils.load(ref_file, IOUtils.Format.txt)
                        refs = [s.split() for s in ref_str.splitlines()]

                        error_ids_file = model_data_dir / f"error-ids-{exp}-{test_set}.json"
                        if model in ["Code2Seq", "Bi-LSTM", "no-split-Bi-LSTM"]:
                            error_ids_file = model_data_dir / f"{exp}-{test_set}-error-ids.json"
                        if not error_ids_file.exists():
                            self.logger.warning(f"Not found error-ids file {error_ids_file}. Assuming it's empty")
                            error_ids = []
                        else:
                            error_ids = IOUtils.load(error_ids_file)

                        for metric_name in metrics:
                            metric_func = self.METRIC_2_FUNC[metric_name]

                            # Compute scores per data, while inserting NaN for error ids
                            metric_results = list()
                            i = 0
                            for ref, output in zip(refs, outputs):
                                if i in error_ids:
                                    metric_results.append(np.NaN)
                                else:
                                    metric_results.append(metric_func(ref, output))
                                i += 1

                            cur_results[metric_name] = metric_results

                        results_file = trial_dir / f"results_{test_set}.json"
                        IOUtils.dump(results_file, cur_results)

    def collect_all_results(self, model: str, metrics: List[str]):
        # Mapping of eval_setting-year -> metric -> test_set -> [trials]
        all_results: Dict[str, Dict[str, Dict[str, List[any]]]]

        # Load existing results, if any
        results_file = Macros.results_dir / "metrics" / f"results-trials-{model}.json"
        if results_file.exists():
            self.logger.info(f"Loading existing metrics from {results_file}")
            all_results = IOUtils.load(results_file)
        else:
            all_results = {}

        model_work_dir = Macros.data_dir / "models-work" / model
        for eval_setting in self.EVAL_SETTINGS:
            for year in self.YEARS:
                exp = f"{eval_setting}-{year}"
                exp_results = all_results.setdefault(exp, {})
                for test_set in [Macros.test_common, Macros.test_standard]:
                    set_results = exp_results.setdefault(test_set, {})

                    for trial in range(Macros.trials):
                        trial_dir = model_work_dir / exp / f"trial-{trial}"
                        cur_results_file = trial_dir / f"results_{test_set}.json"
                        if not cur_results_file.exists():
                            self.logger.warning(f"Results not found at {cur_results_file}")
                            # Set default value for set_results[mname], but don't touch existing results if any
                            for mname in metrics:
                                set_results.setdefault(mname, [None]*Macros.trials)
                        else:
                            results = IOUtils.load(cur_results_file)
                            for mname in metrics:
                                metric = results[mname]
                                set_results.setdefault(mname, [None]*Macros.trials)[trial] = metric

        # Save extracted/updated results
        IOUtils.dump(results_file, all_results, IOUtils.Format.jsonPretty)
        return

    def collect_evo_results(self, model: str, metrics: List[str], debug: bool=False):
        evo_results = dict()
        model_work_dir = Macros.data_dir/"models-work"/f"{model}" if not debug else Macros.data_dir/"models-work"/f"{model}-debug"
        for t in [13, 14, 15, 16, 17]:
            exp = f"{t}{t + 1}-train"
            tmp_result = {k: 0 for k in metrics}
            for trial in range(Macros.trials):
                trial_dir = model_work_dir/ exp / f"trial-{trial}"
                result_file = f"{trial_dir}/test_result.json"
                metrics_dict = IOUtils.load(result_file)
                for k, v in metrics_dict.items():
                    tmp_result[k.lower()] += v
            for k, v in tmp_result.items():
                tmp_result[k] = round(v / Macros.trials, 2)
            evo_results[exp] = tmp_result
        output_dir = Macros.results_dir / "metrics"
        IOUtils.dump(output_dir / f"{model}-evo-results.json", evo_results, IOUtils.Format.jsonPretty)

    def collect_lat_results(self, model: str, metrics: List[str], debug: bool=False):
        lat_results = {k: 0 for k in metrics}
        model_work_dir = Macros.data_dir / "models-work" / f"{model}-latest" if not debug \
            else Macros.data_dir / "models-work" / f"{model}-latest-debug"
        for trial in range(Macros.trials):
            trial_dir = model_work_dir / "latest" / f"trial-{trial}"
            result_file = f"{trial_dir}/test_result.json"
            metrics_dict = IOUtils.load(result_file)
            for k, v in metrics_dict.items():
                lat_results[k.lower()] += v
        for k, v in lat_results.items():
            lat_results[k] = round(v / Macros.trials, 2)

        output_dir = Macros.results_dir / "metrics"
        IOUtils.dump(output_dir / f"{model}-latest-results.json", lat_results, IOUtils.Format.jsonPretty)
