from typing import *

import ijson
import random
from os import listdir
from pathlib import Path
from seutil import LoggingUtils, IOUtils, BashUtils
from tqdm import tqdm

from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.filter.DataFilter import DataFilter
from csevo.filter.DataSpliter import DataSpliter

class DataProcessor:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)
    EVO_YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    TASKS = {
        "CG": "com-gen",
        "MN": "methd-name"
    }

    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.1
    TEST_RATIO = 0.3

    def __init__(self):
        return

    def process_shared(self, output_dir: Path, years: List[str], eval_settings: List[str], task: str = "CG"):
        """
        Extracts the train/val/test method-data for all eval_setting/year.
        This is a shared step for the processing for all models, so do this first (and once).
        1. split the data into train/val/test
        2. extract for every setting
        """
        shared_data_dir = output_dir / f"{task}-shared"
        IOUtils.mk_dir(shared_data_dir)

        # Load project list
        projects = IOUtils.load(Macros.data_dir / f"projects-github-{task}-100.json")

        # Load data
        projects_2_data_list: Dict[str, List] = dict()

        for proj in tqdm(projects):
            # split data split method in the projects, create 19-20-methods-train.json and latest-methods-val.json files
            ds = DataSpliter()
            ds.project_data_split(proj, task)
            method_data_list = IOUtils.load(Macros.repos_results_dir/proj/"collector"/"method-data.json")
            projects_2_data_list[proj] = method_data_list

        # split data across projects
        num_proj = len(projects)
        random.seed(Environment.random_seed)
        random.Random(Environment.random_seed).shuffle(projects)
        train_index = round(num_proj * self.TRAIN_RATIO)
        valid_index = train_index + round(num_proj * self.VAL_RATIO)
        train_projs = projects[: train_index]
        valid_projs = projects[train_index: valid_index]
        test_projs = projects[valid_index:]
        project_split = {
            "train": train_projs,
            "val": valid_projs,
            "test": test_projs
        }
        #project_split = IOUtils.load(Macros.data_dir/f"projects-split-{task}-100.json")
        IOUtils.dump(Macros.data_dir/f"projects-split-{task}-100.json", project_split, IOUtils.Format.jsonNoSort)
        assert len(project_split["test"]) > len(project_split["val"])

        data_type_2_project_list: Dict[str, List] = {
            Macros.train: project_split["train"],
            Macros.val: project_split["val"],
            Macros.test: project_split["test"],
        }

        for year in years:
            data_type_2_data_list: Dict[str, List] = dict()
            year = int(year)
            # test_common: D_test(P_test, year-1, year)
            data_type_2_data_list[f"{year}-{Macros.test_common}"] = list()
            for proj in tqdm(data_type_2_project_list[Macros.test]):
                filter_indexes_file = Macros.repos_results_dir / proj / "collector" / f"19-20-methods-{task}-test.json"
                filter_indexes = IOUtils.load(filter_indexes_file)
                data_type_2_data_list[f"{year}-{Macros.test_common}"] += [projects_2_data_list[proj][i] for i in filter_indexes]

            for eval_setting in eval_settings:
                if eval_setting == "evo":
                    # train: D(P, year-3, year-2)
                    # val: D(P, year-2, year-1)
                    # test_standard: D(P, year-1, year)
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.train}"] = list()
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.val}"] = list()
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.test_standard}"] = list()
                    for proj in tqdm(projects):
                        all_filter_indexes_file = Macros.repos_results_dir / proj / "collector" / f"method-project-{task}-filtered.json"

                        all_filter_indexes = IOUtils.load(all_filter_indexes_file)
                        train_filter_indexes = [af["method_ids"] for af in all_filter_indexes if af["time"] == f"{year-3}_Jan_1-{year-2}_Jan_1"][0]
                        train_filter_indexes += [af["method_ids"] for af in all_filter_indexes if af["time"] == f"{year-4}_Jan_1-{year-3}_Jan_1"][0]
                        val_filter_indexes = [af["method_ids"] for af in all_filter_indexes if af["time"] == f"{year-2}_Jan_1-{year-1}_Jan_1"][0]
                        test_standard_filter_indexes = [af["method_ids"] for af in all_filter_indexes if af["time"] == f"{year-1}_Jan_1-{year}_Jan_1"][0]

                        proj_data_list = projects_2_data_list[proj]

                        data_type_2_data_list[f"{eval_setting}-{year}-{Macros.train}"] += [proj_data_list[i] for i in train_filter_indexes]
                        data_type_2_data_list[f"{eval_setting}-{year}-{Macros.val}"] += [proj_data_list[i] for i in val_filter_indexes]
                        data_type_2_data_list[f"{eval_setting}-{year}-{Macros.test_standard}"] += [proj_data_list[i] for i in test_standard_filter_indexes]
                elif eval_setting == "crossproj-evo":
                    # train: D(P_train, year-3, year-2)
                    # val: D(P_val, year-2, year-1)
                    # test: D(P_test, year-1, year)
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.train}"] = list()
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.val}"] = list()
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.test_standard}"] = list()
                    for data_type_tvt, project_list in data_type_2_project_list.items():
                        if data_type_tvt == Macros.test:
                            data_type_2_data_list[f"{eval_setting}-{year}-{Macros.test_standard}"] = data_type_2_data_list[f"{year}-{Macros.test_common}"]
                        else:
                            for proj in tqdm(project_list):
                                all_filter_indexes_file = Macros.repos_results_dir / proj / "collector" / f"method-project-{task}-filtered.json"
                                all_filter_indexes = IOUtils.load(all_filter_indexes_file)

                                proj_data_list = projects_2_data_list[proj]
                                if data_type_tvt == Macros.train:
                                    train_filter_indexes = [af["method_ids"] for af in all_filter_indexes if af["time"] == f"{year - 3}_Jan_1-{year - 2}_Jan_1"][0]
                                    train_filter_indexes += [af["method_ids"] for af in all_filter_indexes if af["time"] == f"{year - 4}_Jan_1-{year - 3}_Jan_1"][0]
                                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.train}"] += [proj_data_list[i] for i in train_filter_indexes]
                                elif data_type_tvt == Macros.val:
                                    val_filter_indexes = [af["method_ids"] for af in all_filter_indexes if af["time"] == f"{year - 2}_Jan_1-{year - 1}_Jan_1"][0]
                                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.val}"] += [proj_data_list[i] for i in val_filter_indexes]
                elif eval_setting == "crossproj":
                    # train: D(P_train, year)
                    # val: D(P_val, year)
                    # test_standard: D(P_test, year)
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.train}"] = list()
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.val}"] = list()
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.test_standard}"] = list()

                    for data_type_tvt, project_list in data_type_2_project_list.items():
                        data_type = data_type_tvt if data_type_tvt != Macros.test else Macros.test_standard
                        for proj in tqdm(project_list):

                            latest_filter_indexes = list()
                            for t in [Macros.train, Macros.val, Macros.test]:
                                filter_indexes_file = Macros.repos_results_dir / proj / "collector" / f"latest-methods-{task}-{t}.json"
                                latest_filter_indexes += IOUtils.load(filter_indexes_file)

                            data_type_2_data_list[f"{eval_setting}-{year}-{data_type}"] += [projects_2_data_list[proj][i] for i in latest_filter_indexes]
                elif eval_setting == "mixedproj":
                    # train: D_train(P, year)
                    # val: D_val(P, year)
                    # test_standard: D_test(P, year)
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.train}"] = list()
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.val}"] = list()
                    data_type_2_data_list[f"{eval_setting}-{year}-{Macros.test_standard}"] = list()

                    for proj in tqdm(projects):
                        proj_data_list = projects_2_data_list[proj]
                        for data_type_tvt, data_type in zip(
                                [Macros.train, Macros.val, Macros.test],
                                [Macros.train, Macros.val, Macros.test_standard]
                        ):
                            filter_indexes_file = Macros.repos_results_dir / proj / "collector" / f"latest-methods-{task}-{data_type_tvt}.json"
                            filter_indexes = IOUtils.load(filter_indexes_file)
                            data_type_2_data_list[f"{eval_setting}-{year}-{data_type}"] += [proj_data_list[i] for i in filter_indexes]

            for dt, data_list in data_type_2_data_list.items():
                IOUtils.dump(shared_data_dir / f"{dt}.json", data_list, IOUtils.Format.json)
        return

    def process(self, model: str, output_dir: Path, task: str, year: int, eval_setting: str):
        """
        Main entry for processors of different models.
        :param model: the model name, one of {"DeepCom", "ast-attendgru"}
        :param output_dir: the output directory (usually data/models)
        :param task: the task name, either "CG" or "MN"
        :param year: the year that the testing data should be on
        :param eval_setting: the evaluation setting, one of {"evo", "crossproj", "mixedproj"}
        """
        assert year == self.EVO_YEARS[-1]  # TODO: Only support the latest year for now
        assert task in self.TASKS.keys()

        model_data_dir = output_dir/model

        if model == "DeepCom":
            from csevo.processor.DeepComProcessor import DeepComProcessor
            processor = DeepComProcessor()
        elif model == "DeepCom-Preorder":
            from csevo.processor.DeepComProcessor import DeepComProcessor
            processor = DeepComProcessor()
        elif model == "Bi-LSTM":
            from csevo.processor.BiLSTMProcessor import BiLSTMProcessor
            processor = BiLSTMProcessor()
        elif model == "no-split-Bi-LSTM":
            from csevo.processor.BiLSTMProcessor import BiLSTMProcessor
            processor = BiLSTMProcessor()
        elif model == "Transformer":
            from csevo.processor.TransformerProcessor import TransformerProcessor
            processor = TransformerProcessor()
            data_prefix = f"{eval_setting}-{year}"
            processor.process_data(model_data_dir, data_prefix)
            return
        elif model == "ASTAttendGRU":
            from csevo.processor.ASTAttendGRUProcessor import ASTAttendGRUProcessor
            processor = ASTAttendGRUProcessor()
        elif model == "Code2Seq":
            from csevo.processor.Code2SeqProcessor import Code2SeqProcessor
            processor = Code2SeqProcessor()
        else:
            raise ValueError(f"Illegal model {model}")
        # end if
        error_ids = None

        # Load dataset after split (from shared directory)
        shared_data_dir = output_dir / f"{task}-shared"
        self.logger.info(f"Loading dataset from {shared_data_dir}")
        data_type_2_data_list: Dict[str, List] = dict()
        data_type_2_data_list[Macros.test_common] = IOUtils.load(shared_data_dir / f"{year}-{Macros.test_common}.json", IOUtils.Format.json)
        for dt in [Macros.train, Macros.val, Macros.test_standard]:
            data_type_2_data_list[dt] = IOUtils.load(shared_data_dir / f"{eval_setting}-{year}-{dt}.json", IOUtils.Format.json)

        # Process each set
        for data_type, data_list in data_type_2_data_list.items():
            sub_dir_name = f"{eval_setting}-{year}-{data_type}"

            if data_type in [Macros.test_common, Macros.test_standard]:
                data_type_tvt = Macros.test
            else:
                data_type_tvt = data_type

            model_dt_output_dir = model_data_dir / sub_dir_name
            IOUtils.mk_dir(model_dt_output_dir)
            if model == "DeepCom":
                error_ids = processor.process_data(data_list, data_type_tvt, model_dt_output_dir, "sbt")
            elif model == "DeepCom-Preorder":
                error_ids = processor.process_data(data_list, data_type_tvt, model_dt_output_dir, "Preorder")
            elif model == "Code2Seq":
                error_ids = processor.process_data(data_list, data_type_tvt, model_dt_output_dir)
            elif model == "Bi-LSTM":
                processor.process_data(data_list, data_type_tvt, model_dt_output_dir)
            elif model == "no-split-Bi-LSTM":
                processor.process_data(data_list, data_type_tvt, model_dt_output_dir, split=False)
            if error_ids is not None:
                self.logger.warning(f"Error data count: {len(error_ids)}")
                IOUtils.dump(model_data_dir / f"error-ids-{sub_dir_name}.json", error_ids, IOUtils.Format.json)
        # extra step for Open-NMT data
        if model == "Bi-LSTM" or model == "no-split-Bi-LSTM":
            # build dataset used by Open-NMT
            BashUtils.run(
                f"onmt_preprocess -train_src {model_data_dir}/{eval_setting}-{year}-{Macros.train}/src-train.txt "
                f"-train_tgt {model_data_dir}/{eval_setting}-{year}-{Macros.train}/tgt-train.txt "
                f"-valid_src {model_data_dir}/{eval_setting}-{year}-{Macros.val}/src-val.txt "
                f"-valid_tgt {model_data_dir}/{eval_setting}-{year}-{Macros.val}/tgt-val.txt "
                f"-save_data {model_data_dir}/{eval_setting}-{year}-{Macros.train}/biLSTM --src_seq_length 200 --src_seq_"
                f"length_trunc 200", expected_return_code=0)

        return
