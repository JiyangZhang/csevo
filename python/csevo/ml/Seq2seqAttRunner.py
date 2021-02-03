from typing import *

import copy
from pathlib import Path

from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.Environment import Environment
from csevo.Macros import Macros

from csevo.ml.TACCRunner import TACCRunnerConsts


class Seq2seqAttRunner:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, work_dir: Path, year: int, eval_setting: str):
        self.year = year
        self.eval_setting = eval_setting
        self.work_dir: Path = work_dir / f"{self.eval_setting}-{self.year}"

        # Using the same data source as DeepCom
        self.model_data_dir: Path = Macros.data_dir / "models-data" / "DeepCom"

        self.code_dir: Path = self.work_dir/"code"
        self.base_config_file: Path = Macros.python_dir/"configs"/"Seq2seq-Attention.yaml"
        return

    REPO_URL = "https://github.com/JiyangZhang/DeepCom.git"
    REPO_SHA = Macros.DeepCom_hash
    CONDA_ENV = "DeepCom"

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

        self.logger.info(f"Preparing the data for {self.eval_setting} {self.year} at {data_dir}")
        IOUtils.rm_dir(data_dir)
        IOUtils.mk_dir(data_dir)

        # Copy train/val/test_common/test_standard data
        BashUtils.run(f"cp -r {self.model_data_dir}/{data_prefix}-{Macros.train}/train {data_dir}/train", expected_return_code=0)
        BashUtils.run(f"cp -r {self.model_data_dir}/{data_prefix}-{Macros.val}/valid {data_dir}/valid", expected_return_code=0)
        BashUtils.run(f"cp -r {self.model_data_dir}/{data_prefix}-{Macros.test_common}/test {data_dir}/{Macros.test_common}", expected_return_code=0)
        BashUtils.run(f"cp -r {self.model_data_dir}/{data_prefix}-{Macros.test_standard}/test {data_dir}/{Macros.test_standard}", expected_return_code=0)

        # Copy vocab
        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.train}/vocab* {data_dir}/", expected_return_code=0)
        return

    def prepare_configs_and_scripts(self, trials: List[int]):
        data_dir = self.work_dir / "data"
        base_config = IOUtils.load(self.base_config_file, IOUtils.Format.jsonPretty)

        for trial in trials:
            trial_dir = self.work_dir / f"trial-{trial}"
            IOUtils.mk_dir(trial_dir)

            config = copy.copy(base_config)
            config["data_dir"] = str(data_dir)
            config["model_dir"] = str(trial_dir / "model")
            config["output"] = str(trial_dir / "output.txt")

            config_file = trial_dir / "config.json"
            IOUtils.dump(config_file, config, IOUtils.Format.jsonPretty)

            training_trace_file = trial_dir / "training-trace.json"

            train_script_file = trial_dir / f"{Macros.train}.sh"
            # The gpu-id argument is necessary for tensorflow, even if we are using CUDA_VISIBLE_DEVICES
            train_script = f"#!/bin/bash\n" \
                           f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                           f"conda activate {self.CONDA_ENV}\n" \
                           f"module load cuda/10.0 cudnn/7.6.2\n" \
                           f"cd {self.code_dir}/translate\n" \
                           f"python3 __main__.py {config_file} --train -v --train-log {training_trace_file} --gpu-id $1 &> {trial_dir}/log-{Macros.train}.txt\n"
            IOUtils.dump(train_script_file, train_script, IOUtils.Format.txt)
            BashUtils.run(f"chmod +x {train_script_file}", expected_return_code=0)

            for test_type in [Macros.test_common, Macros.test_standard]:
                output_file = trial_dir / f"output_{test_type}.txt"
                config["output"] = str(output_file)
                test_config_file = trial_dir / f"config_{test_type}.json"
                IOUtils.dump(test_config_file, config, IOUtils.Format.jsonPretty)

                test_script_file = trial_dir / f"{test_type}.sh"
                test_script = f"#!/bin/bash\n" \
                              f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                              f"conda activate {self.CONDA_ENV}\n" \
                              f"module load cuda/10.0 cudnn/7.6.2\n" \
                              f"cd {self.code_dir}/translate\n" \
                              f"python3 __main__.py {test_config_file} --eval {data_dir}/{test_type}/test.token.code {data_dir}/{test_type}/test.token.sbt {data_dir}/{test_type}/test.token.nl --gpu-id $1 &> {trial_dir}/log-{test_type}.txt\n" \
                              f"python3 Bleu.py {data_dir}/{test_type}/test.token.nl {trial_dir}/output_{test_type}.txt {trial_dir}/results_{test_type}.json\n"
                IOUtils.dump(test_script_file, test_script, IOUtils.Format.txt)
                BashUtils.run(f"chmod +x {test_script_file}", expected_return_code=0)

        return
