from typing import *
import random
import copy
from pathlib import Path

import yaml
from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.ml.TACCRunner import TACCRunnerConsts


class BiLSTMRunner:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, work_dir: Path, year: int, eval_setting: str):
        # TODO: change to new API
        self.year = year
        self.eval_setting = eval_setting
        self.work_dir: Path = work_dir / f"{self.eval_setting}-{self.year}"
        self.model_data_dir: Path = Macros.data_dir / "models-data" / "Bi-LSTM"
        self.code_dir: Path = self.work_dir/"code"
        self.base_config_file: Path = Macros.python_dir/"configs"/"Bi-LSTM.yaml"
        self.data_dir = self.work_dir / "data"

        return

    REPO_URL = "https://github.com/JiyangZhang/OpenNMT-py.git"
    REPO_SHA = "60125c807d1cb18099a69dbfba699bcdf30560b1"
    CONDA_ENV = "csevo"

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

        # copy eval code
        BashUtils.run(f"cp {Macros.this_dir}/eval/eval_utils.py {self.code_dir}/")
        return

    def prepare_data(self):
        data_prefix = f"{self.eval_setting}-{self.year}"
        IOUtils.rm_dir(self.data_dir)
        IOUtils.mk_dir(self.data_dir)

        # build dataset used by Open-NMT
        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.train}/biLSTM* {self.data_dir}/",
                      expected_return_code=0)

        BashUtils.run(
            f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_common}/src-test.txt {self.data_dir}/src-{Macros.test_common}.txt",
            expected_return_code=0)
        BashUtils.run(
            f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_common}/tgt-test.txt {self.data_dir}/tgt-{Macros.test_common}.txt",
            expected_return_code=0)

        BashUtils.run(
            f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_standard}/src-test.txt {self.data_dir}/src-{Macros.test_standard}.txt",
            expected_return_code=0)
        BashUtils.run(
            f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_standard}/tgt-test.txt {self.data_dir}/tgt-{Macros.test_standard}.txt",
            expected_return_code=0)

        return

    def prepare_configs_and_scripts(self, trials: List[int]):
        with open(self.base_config_file, "r") as f:
            base_config = yaml.load(f)
        exp_dir = self.work_dir
        for trial in trials:
            seed = random.randint(0, 9)
            trial_dir = exp_dir / f"trial-{trial}"
            IOUtils.mk_dir(trial_dir)

            config = copy.copy(base_config)
            config["data"] = str(self.data_dir / "biLSTM")
            config["save_model"] = str(trial_dir / "bestLSTM")
            config_file = trial_dir / "config.yaml"
            with open(config_file, "w+") as f:
                yaml.dump(config, f)
            train_script_file = trial_dir/"train.sh"
            train_script = f"#!/bin/bash\n" \
                           f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                           f"module load cuda/10.1 cudnn/7.6.2\n" \
                           f"conda activate {self.CONDA_ENV}\n" \
                           f"cd {self.code_dir}\n" \
                           f"export MKL_SERVICE_FORCE_INTEL=1\n"\
                           f"python3 train.py --config {config_file} --world_size 1 --gpu_ranks 0 -keep_checkpoint 1 " \
                           f"--seed {seed} &> {trial_dir}/train-log.txt\n"
            IOUtils.dump(train_script_file, train_script, IOUtils.Format.txt)
            BashUtils.run(f"chmod +x {train_script_file}", expected_return_code=0)

            for test_type in [Macros.test_common, Macros.test_standard]:
                test_script_file = trial_dir/f"{test_type}.sh"
                output_file = trial_dir / f"output_{test_type}.txt"
                test_script = f"#!/bin/bash\n" \
                              f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                              f"module load cuda/10.1 cudnn/7.6.2\n" \
                              f"conda activate {self.CONDA_ENV}\n" \
                              f"cd {self.code_dir}\n" \
                              f"export MKL_SERVICE_FORCE_INTEL=1\n"\
                              f"python3 translate.py "\
                              f"--model {trial_dir}/*.pt --output {output_file} --src {self.data_dir}/src-{test_type}.txt "\
                              f"&> {trial_dir}/{test_type}-log.txt\n" \
                              f"python3 eval_utils.py " \
                              f"{self.data_dir}/tgt-{test_type}.txt {output_file} {trial_dir}/results_{test_type}.json\n"
                IOUtils.dump(test_script_file, test_script, IOUtils.Format.txt)
                BashUtils.run(f"chmod +x {test_script_file}", expected_return_code=0)

            # end for
        return

