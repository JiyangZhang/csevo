from typing import *

import copy
from pathlib import Path

from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.ml.TACCRunner import TACCRunnerConsts


class Code2SeqRunner:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, work_dir: Path, year: int, eval_setting: str):
        self.year = year
        self.eval_setting = eval_setting
        self.work_dir: Path = work_dir / f"{self.eval_setting}-{self.year}"
        self.model_data_dir: Path = Macros.data_dir / "models-data" / "Code2Seq"
        self.base_config_file: Path = Macros.python_dir / "configs" / "Code2Seq.yaml"
        self.code_dir: Path = self.work_dir / "code"

        return

    REPO_URL = "git@github.com:JiyangZhang/Code2Seq.git"
    REPO_SHA = Macros.Code2Seq_hash
    CONDA_ENV = "Code2Seq"

    def prepare(self):
        self.prepare_code()
        self.prepare_data()
        self.prepare_configs_and_scripts(list(range(Macros.trials)))
        return

    def prepare_code(self):
        IOUtils.rm_dir(self.code_dir)
        IOUtils.mk_dir(self.code_dir.parent)
        with IOUtils.cd(self.code_dir.parent):
            BashUtils.run(f"git clone {self.REPO_URL} {self.code_dir.name}", expected_return_code=0)
        # end with

        with IOUtils.cd(self.code_dir):
            BashUtils.run(f"git checkout {self.REPO_SHA}", expected_return_code=0)
        # end with
        return

    def prepare_data(self):
        data_prefix = f"{self.eval_setting}-{self.year}"
        data_dir = self.work_dir / "data"

        self.logger.info(f"Preparing the data for {self.eval_setting} {self.year} at {self.work_dir}")
        IOUtils.rm_dir(data_dir)
        IOUtils.mk_dir(data_dir)

        # Copy train/val/test_common/test_standard data
        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.train}/code2seq.train.c2s {self.work_dir}/data/",
                      expected_return_code=0)
        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.val}/code2seq.val.c2s {self.work_dir}/data/",
                      expected_return_code=0)
        BashUtils.run(
            f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_common}/code2seq.test.c2s {self.work_dir}/data/code2seq.{Macros.test_common}.c2s",
            expected_return_code=0)
        BashUtils.run(
            f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_standard}/code2seq.test.c2s {self.work_dir}/data/code2seq.{Macros.test_standard}.c2s",
            expected_return_code=0)

        # Copy vocab
        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.train}/code2seq.dict.c2s {self.work_dir}/data/",
                      expected_return_code=0)
        return

    def prepare_configs_and_scripts(self, trials: List[int]):
        exp_dir = self.work_dir

        for trial in trials:
            trial_dir = exp_dir / f"trial-{trial}"
            IOUtils.mk_dir(trial_dir)

            model_dir = trial_dir / "models"
            IOUtils.mk_dir(model_dir)
            log_dir = trial_dir / "logs"
            IOUtils.mk_dir(log_dir)
            data = str(exp_dir / "data/code2seq")
            val_data = data + ".val.c2s"
            train_log = trial_dir / "training-trace.json"

            train_script_file = trial_dir / f"{Macros.train}.sh"
            # Copy config file
            BashUtils.run(f"cp {self.base_config_file} {trial_dir}/config.yaml", expected_return_code=0)
            output_file = trial_dir / "output_tmp.txt"
            reference_file = trial_dir / "ref_tmp.txt"
            config_file = trial_dir / "config.yaml"
            train_script = f"#!/bin/bash\n" \
                           f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                           f"conda activate {self.CONDA_ENV}\n" \
                           f"module load cuda/10.0 cudnn/7.6.2\n" \
                           f"cd {self.code_dir}\n" \
                           f"python -u code2seq.py --data {data} --test {val_data} --log {train_log} --config {config_file} " \
                           f"--pred_file {output_file} --ref_file {reference_file} "\
                           f"--save_prefix {model_dir}/model --gpu_id $1 &> {trial_dir}/train-log.txt"
            IOUtils.dump(train_script_file, train_script, IOUtils.Format.txt)
            BashUtils.run(f"chmod +x {train_script_file}", expected_return_code=0)

            for test_type in [Macros.test_common, Macros.test_standard]:
                test_data = exp_dir / "data" / f"code2seq.{test_type}.c2s"
                output_file = trial_dir / f"output_{test_type}.txt"
                reference_file = trial_dir / f"ref_{test_type}.txt"
                test_script_file = trial_dir / f"{test_type}.sh"
                test_script = f"#!/bin/bash\n" \
                              f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                              f"conda activate {self.CONDA_ENV}\n" \
                              f"module load cuda/10.0 cudnn/7.6.2\n" \
                              f"cd {self.code_dir}\n" \
                              f"python3 code2seq.py --load {model_dir}/model_best --test {test_data} --config {config_file} " \
                              f"--pred_file {output_file} --ref_file {reference_file} "\
                              f"--gpu_id $1 &> {trial_dir}/{test_type}-log.txt\n" \
                              f"python3 eval_utils.py {reference_file} {output_file} {trial_dir}/results_{test_type}.json\n"
                IOUtils.dump(test_script_file, test_script, IOUtils.Format.txt)
                BashUtils.run(f"chmod +x {test_script_file}", expected_return_code=0)

        return
