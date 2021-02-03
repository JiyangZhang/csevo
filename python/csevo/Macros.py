from typing import *

import os
from pathlib import Path


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    python_dir: Path = this_dir.parent
    debug_dir: Path = python_dir / "debug"
    project_dir: Path = python_dir.parent
    paper_dir: Path = project_dir / "papers" / "icse21"
    config_dir: Path = python_dir / "configs"

    collector_dir: Path = project_dir / "collector"
    collector_version = "0.1-dev"

    data_dir: Path = project_dir / "csevo-data"
    results_dir: Path = project_dir / "csevo-results"
    repos_downloads_dir: Path = project_dir / "_downloads"
    repos_results_dir: Path = project_dir / "_results"
    ml_logs_dir: Path = python_dir / "ml-logs"

    train = "train"
    val = "val"
    test = "test"
    test_common = "test_common"
    test_standard = "test_standard"
    evo = "evolution"
    lat = "latest"
    evo_evo = f"{evo}-{evo}"
    lat_evo = f"{lat}-{evo}"
    evo_lat = f"{evo}-{lat}"
    lat_lat = f"{lat}-{lat}"

    multi_processing = 8
    # trials = 3
    trials = 1
    train_ratio = 0.6
    val_ration = 0.1
    test_ration = 0.3

    tasks = ["CG", "MN"]

    DeepCom_hash = "30bce20caa826892665f5198711d659e941b0fff"
    Code2Seq_hash = "da714c7c59b4c24eaaeedbeae31fc55f6c3f16e9"

    # TODO private info
    mongodb_port = 20144
    mongodb_server = "luzhou.ece.utexas.edu"
