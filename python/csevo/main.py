from typing import *

from pathlib import Path
import pkg_resources
import random
import sys
import time
from os import listdir
from seutil import CliUtils, IOUtils, LoggingUtils
from tqdm import tqdm

from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.Utils import Utils

# Check seutil version
EXPECTED_SEUTIL_VERSION = "0.4.12"
if pkg_resources.get_distribution("seutil").version != EXPECTED_SEUTIL_VERSION:
    print(
        f"seutil version does not meet expectation! Expected version: {EXPECTED_SEUTIL_VERSION}, current installed version: {pkg_resources.get_distribution('seutil').version}",
        file=sys.stderr)
    print(
        f"Hint: either upgrade seutil, or modify the expected version (after confirmation that the version will work)",
        file=sys.stderr)
    sys.exit(-1)
# end if


logging_file = Macros.python_dir / "experiment.log"
LoggingUtils.setup(filename=str(logging_file))

logger = LoggingUtils.get_logger(__name__)


# ==========
# Table & Plot

def make_tables(**options):
    from csevo.Table import Table
    which = Utils.get_option_as_list(options, "which")

    table_maker = Table()
    table_maker.make_tables(which, options)
    return


def make_plots(**options):
    from csevo.Plot import Plot
    which = Utils.get_option_as_list(options, "which")

    plot_maker = Plot()
    plot_maker.make_plots(which, options)
    return


def make_numbers(**options):
    from csevo.Table import Table
    table_maker = Table()
    model = options.get("model")
    use_latest = Utils.get_option_as_boolean(options, "use_latest")
    debug = Utils.get_option_as_boolean(options, "debug")
    table_maker.make_numbers_model_results(model, use_latest, debug)


# ==========
# Data collection, sample

def parse_projects_deepcom():
    from csevo.collector.DataCollector import DataCollector
    dc = DataCollector()
    dc.parse_projects(Macros.data_dir / "DeepCom-projects.txt")


def collect_data(**options):
    from csevo.collector.DataCollector import DataCollector
    project_urls_file = Path(options.get("project_urls_file", Macros.data_dir / "projects-github-dpcom.txt"))
    skip_collected = Utils.get_option_as_boolean(options, "skip_collected")
    beg = options.get("beg")
    cnt = options.get("cnt")

    dc = DataCollector()
    dc.collect_projects(project_urls_file, skip_collected=skip_collected, beg=beg, cnt=cnt)
    return


def get_github_top_repos(**options):
    from csevo.collector.DataCollector import DataCollector
    dc = DataCollector()
    dc.get_github_top_repos()
    return


def store_repo_results(**options):
    from csevo.collector.Database import Database
    from csevo.collector.DataCollector import DataCollector

    local = Utils.get_option_as_boolean(options, "local")
    force_update = Utils.get_option_as_boolean(options, "force_update")
    repos_results_dir = Path(options.get("repos_results_dir", Macros.repos_results_dir))

    db = Database(local=local)
    dc = DataCollector(database=db)
    dc.store_repo_results(repos_results_dir, force_update=force_update)
    return


def split_project_data(**options):
    from csevo.filter.DataSpliter import DataSpliter
    spliter = DataSpliter()
    task = options.get("task")
    debug = Utils.get_option_as_boolean(options, "debug")
    spliter.project_data_split(task, debug)
    return


def split_projects(**options):
    from csevo.filter.DataSpliter import DataSpliter
    spliter = DataSpliter()
    random_seed = options.get("random_seed")
    debug = Utils.get_option_as_boolean(options, "debug")
    if debug:
        method_file = Macros.data_dir / "latest-debug" / "method-data.json"
    else:
        method_file = Macros.data_dir / "latest" / "method-data.json"
    spliter.split_project(method_file, random_seed, debug)


def filter_data(**options):
    from csevo.filter.AlphaFilter import AlphaFilter
    from csevo.filter.BetaFIlter import BetaFilter
    which = options.get("which")
    if which == "alpha":
        proj_file = options.get("proj_list", Macros.data_dir / "projects-github-MN-100.json")
        alpha_filter = AlphaFilter()
        alpha_filter.process_data_concurrent(proj_file)
    elif which == "beta":
        beta_filter = BetaFilter()
        beta_filter.process_data_concurrent()
    else:
        print("not implemented")


def cut_data(**options):
    from csevo.collector.DataCollector import DataCollector
    dc = DataCollector()
    dc.data_cut(options.get("data_size", 100))  # default to 100, but can be adjust


# ==========
# Metrics collection

def collect_metrics(**options):
    from csevo.collector.MetricsCollector import MetricsCollector

    mc = MetricsCollector()
    mc.collect_metrics(**options)
    return


def collect_metrics_time_wise(**options):
    from csevo.collector.MetricsCollector import MetricsCollector
    which = options.get("which")
    dataset = options.get("dataset")
    if which == "raw":
        MetricsCollector.collect_metrics_timewise_dataset()
    elif which == "filtered":
        filter = options.get("filter")
        MetricsCollector.collect_metrics_filtered_dataset(dataset, filter)


def collect_model_results(**options):
    from csevo.collector.ModelResultsCollector import ModelResultsCollector
    collector = ModelResultsCollector()
    model = options.get("model")
    task = options.get("task")
    re_eval = Utils.get_option_as_boolean(options, "re_eval")
    collector.collect_results(model, task, re_eval)


# ==========
# Machine learning

def split_dataset(**options):
    from csevo.collector.Database import Database
    from csevo.processor.DatasetSplitter import DatasetSplitter

    local = Utils.get_option_as_boolean(options, "local")

    db = Database(local=local)
    ds = DatasetSplitter(database=db)
    ds.split_dataset()
    return


def clean_comgen_data(**options):
    from csevo.filter.DataFilter import DataFilter
    config_file_name = options.get("config")
    config_file = Macros.config_dir / config_file_name
    df = DataFilter(config_file)
    project_file = options.get("proj_file", Macros.data_dir / "projects-github-CG-100.json")
    projects = IOUtils.load(project_file)
    for proj in tqdm(projects):
        method_data_file = Macros.repos_results_dir / proj / "collector" / "method-data.json"
        filtered_data_file = Macros.repos_results_dir / proj / "collector" / "method-project-alpha-filtered.json"
        revision_data_file = Macros.repos_results_dir / proj / "collector" / "method-project-revision.json"
        # Data filtering and cleaning
        method_data_list = IOUtils.load(method_data_file)
        clean_method_data_list = list()
        clean_method_id_list = list()
        for ex in method_data_list:
            new_ex = ex
            new_ex["code"], new_ex["comment_summary"] = df.data_filter(ex["code"], ex["comment_summary"])
            if new_ex["code"] != "" and new_ex["comment_summary"] != "":
                clean_method_data_list.append(new_ex)
                clean_method_id_list.append(new_ex["id"])
        # dump the clean method index for comment generation task
        IOUtils.dump(Macros.repos_results_dir / proj / "collector" / "clean-method-idx.json", clean_method_id_list,
                     IOUtils.Format.jsonNoSort)
        # update alpha-filtered data
        filtered_data_list = IOUtils.load(filtered_data_file)
        for delta_data in filtered_data_list:
            new_clean_filtered_method_ids = set(delta_data["method_ids"]).intersection(clean_method_id_list)
            delta_data["method_ids"] = list(new_clean_filtered_method_ids)
        IOUtils.dump(Macros.repos_results_dir / proj / "collector" / "method-project-CG-filtered.json", filtered_data_list, IOUtils.Format.jsonNoSort)
        # update project revision data
        revision_data_list = IOUtils.load(revision_data_file)
        for year_data in revision_data_list:
            new_clean_latest_method_ids = set(year_data["method_ids"]).intersection(clean_method_id_list)
            year_data["method_ids"] = list(new_clean_latest_method_ids)
        IOUtils.dump(Macros.repos_results_dir / proj / "collector" / "method-project-CG-revision.json", revision_data_list, IOUtils.Format.jsonNoSort)

def process_data_shared(**options):
    from csevo.processor.DataProcessor import DataProcessor
    output_dir = options.get("output_dir", Macros.data_dir / "models-data")
    task = options.get("task")
    years = Utils.get_option_as_list(options, "years")
    eval_settings = Utils.get_option_as_list(options, "eval_settings")
    dp = DataProcessor()
    dp.process_shared(output_dir, years, eval_settings, task)
    return


def process_data(**options):
    from csevo.processor.DataProcessor import DataProcessor
    model = options["model"]  # e.g., "DeepCom"
    output_dir = options.get("output_dir", Macros.data_dir / "models-data")
    task = options.get("task")
    year = options["year"]
    eval_setting = options["eval_setting"]  # one of {"evo", "crossproj", "mixedproj"}
    dp = DataProcessor()
    dp.process(model, output_dir, task, year, eval_setting)
    return


# def load_data(**options):
#     from csevo.processor.DataProcessor import DataProcessor
#     data_type = options.get("data_type")
#     task = options.get("task")
#     dp = DataProcessor()
#     dp.load_data(data_type, task)


def prepare_model(**options):
    from csevo.ml.TACCRunner import TACCRunner

    work_dir = Path(options.get("work_dir", Macros.data_dir / "models-work"))
    model = options["model"]
    year = options["year"]
    eval_setting = options["eval_setting"]
    debug = Utils.get_option_as_boolean(options, "debug")
    runner = TACCRunner(work_dir)
    runner.prepare_model(model, year, eval_setting, debug)
    return


def prepare_model_local(**options):
    from csevo.ml.LocalRunner import LocalRunner

    work_dir = Path(options.get("work_dir", Macros.data_dir / "models-work"))
    model = options["model"]
    use_latest = Utils.get_option_as_boolean(options, "use_latest")
    debug = Utils.get_option_as_boolean(options, "debug")
    cross_proj = Utils.get_option_as_boolean(options, "cross_proj")
    runner = LocalRunner(work_dir)
    runner.prepare_model(model, use_latest, debug, cross_proj)
    return


def run_models(**options):
    from csevo.ml.TACCRunner import TACCRunner

    work_dir = Path(options.get("work_dir", Macros.data_dir / "models-work"))
    mode = options.get("mode", Macros.train)
    models = Utils.get_option_as_list(options, "models")
    exps = Utils.get_option_as_list(options, "exps")
    trials = Utils.get_option_as_list(options, "trials")
    timeout = options.get("timeout")
    beg = options.get("beg", 0)
    cnt = options.get("cnt", -1)
    local = Utils.get_option_as_boolean(options, "local")
    runner = TACCRunner(work_dir)
    if not local:
        runner.run_models(mode, models, exps, trials, timeout, beg, cnt)
    else:
        runner.run_models_local(mode, models, exps, trials, timeout, beg, cnt)
    return


# ==========
# Main

def normalize_options(opts: dict) -> dict:
    # Set a different log file
    if "log_path" in opts:
        logger.info(f"Switching to log file {opts['log_path']}")
        LoggingUtils.setup(filename=opts['log_path'])
    # end if

    # Set debug mode
    if "debug" in opts and str(opts["debug"]).lower() != "false":
        Environment.is_debug = True
        logger.debug("Debug mode on")
        logger.debug(f"Command line options: {opts}")
    # end if

    # Set parallel mode - all automatic installations are disabled
    if "parallel" in opts and str(opts["parallel"]).lower() != "false":
        Environment.is_parallel = True
        logger.warning(f"Parallel mode on")
    # end if

    # Set/report random seed
    if "random_seed" in opts:
        Environment.random_seed = int(opts["random_seed"])
    else:
        Environment.random_seed = time.time_ns()
    # end if
    random.seed(Environment.random_seed)
    logger.info(f"Random seed is {Environment.random_seed}")

    # Automatically update data and results repo
    Environment.require_data()
    Environment.require_results()
    return opts


if __name__ == "__main__":
    CliUtils.main(sys.argv[1:], globals(), normalize_options)
