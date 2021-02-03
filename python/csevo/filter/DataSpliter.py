from os import listdir, path
from tqdm import tqdm
from typing import *
import sys
from pathlib import Path
import random
import ijson
from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.data.MethodData import MethodData
from csevo.Environment import Environment
from csevo.filter.AbstractFilter import AbstractFilter
from csevo.Macros import *


class DataSpliter:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.1
    TEST_RATIO = 0.3

    TIME = 2019
    LATEST = 2020

    def project_data_split(self, proj: str, task: str = "CG"):
        """
        Split methods from each project. Will get 6 files for each project:
        19-20-methods-train.json. 19-20-methods-val.json, 19-20-methods-test.json
        latest-methods-train.json, latest-methods-val.json, latest-methods-test.json
        """
        if task == "CG":
            revision_data = IOUtils.load(Macros.repos_results_dir / proj / "collector" / f"method-project-{task}-revision.json")
        else:
            revision_data = IOUtils.load(Macros.repos_results_dir / proj / "collector" / f"method-project-revision.json")

        filtered_methods = IOUtils.load(Macros.repos_results_dir / proj / "collector" / f"method-project-{task}-filtered.json")
        # First split evo data
        new_method_ids = [delta_data["method_ids"] for delta_data in filtered_methods if delta_data["time"] ==
                          "2019_Jan_1-2020_Jan_1"][0]
        # Reset random seed to ensure reproducibility
        random.seed(Environment.random_seed)

        # Shuffle the methods
        random.Random(Environment.random_seed).shuffle(new_method_ids)
        train_index = round(len(new_method_ids) * self.TRAIN_RATIO)
        valid_index = train_index + round(len(new_method_ids) * self.VAL_RATIO)
        new_train_methods = new_method_ids[: train_index]
        new_valid_methods = new_method_ids[train_index: valid_index]
        new_test_methods = new_method_ids[valid_index:]

        IOUtils.dump(Macros.repos_results_dir / proj / "collector" / f"19-20-methods-{task}-train.json", new_train_methods)
        IOUtils.dump(Macros.repos_results_dir / proj / "collector" / f"19-20-methods-{task}-val.json", new_valid_methods)
        IOUtils.dump(Macros.repos_results_dir / proj / "collector" / f"19-20-methods-{task}-test.json", new_test_methods)

        # Second split latest data
        latest_method_ids = \
        [year_data["method_ids"] for year_data in revision_data if year_data["year"] == "2020_Jan_1"][0]
        old_method_ids = list(set(latest_method_ids) - set(new_method_ids))
        # Reset random seed to ensure reproducibility
        random.seed(Environment.random_seed)

        # Shuffle the old_method ids
        random.Random(Environment.random_seed).shuffle(new_method_ids)
        train_index = round(len(old_method_ids) * self.TRAIN_RATIO)
        valid_index = train_index + round(len(old_method_ids) * self.VAL_RATIO)
        old_train_methods = old_method_ids[: train_index]
        old_valid_methods = old_method_ids[train_index: valid_index]
        old_test_methods = old_method_ids[valid_index:]

        latest_train_methods = old_train_methods + new_train_methods
        latest_valid_methods = old_valid_methods + new_valid_methods
        latest_test_methods = old_test_methods + new_test_methods

        IOUtils.dump(Macros.repos_results_dir / proj / "collector" / f"latest-methods-{task}-train.json", latest_train_methods)
        IOUtils.dump(Macros.repos_results_dir / proj / "collector" / f"latest-methods-{task}-test.json", latest_test_methods)
        IOUtils.dump(Macros.repos_results_dir / proj / "collector" / f"latest-methods-{task}-val.json", latest_valid_methods)

    def split_project(self, method_file: Path, random_seed: int, debug: bool = False):
        """
        Split projects into train, val, test according to the project names. Will get 2 new files:
        project-list.json, project-split.json.
        """
        proj_list = set()
        with open(method_file, "r") as f:
            objects = ijson.items(f, "item")
            for o in objects:
                proj_list.add(o["prj_name"])
        num_proj = len(proj_list)
        proj_list = list(proj_list)
        if debug:
            output_dir = Path("/tmp/nlpast-data-10")
        else:
            output_dir = Path("/tmp/nlpast-data-880")

        IOUtils.dump(output_dir / "project-list.json", proj_list)

        random.seed(random_seed)
        random.shuffle(proj_list)
        train_index = round(num_proj * 0.8)
        valid_index = train_index + round(num_proj * 0.1)
        train_projs = proj_list[: train_index]
        valid_projs = proj_list[train_index: valid_index]
        test_projs = proj_list[valid_index:]
        project_split = {
            "train": train_projs,
            "val": valid_projs,
            "test": test_projs
        }
        IOUtils.dump(output_dir / "project-split.json", project_split)
