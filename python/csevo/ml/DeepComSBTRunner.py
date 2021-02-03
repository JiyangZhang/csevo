from typing import *

import copy
from pathlib import Path

from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.Environment import Environment
from csevo.Macros import Macros

from csevo.ml.TACCRunner import TACCRunnerConsts


class DeepComSBTRunner:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, work_dir: Path, use_latest: bool):
        # TODO: change to new API
        self.work_dir: Path = work_dir
        self.use_latest: bool = use_latest

        self.code_dir: Path = self.work_dir/"code"
        self.model_data_dir: Path = Macros.data_dir/"models-data"/"DeepCom-SBT"
        self.base_config_file: Path = Macros.python_dir/"configs"/"DeepCom-SBT.yaml"
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
        with IOUtils.cd(self.code_dir.parent):
            BashUtils.run(f"git clone {self.REPO_URL} {self.code_dir.name}", expected_return_code=0)
        # end with

        with IOUtils.cd(self.code_dir):
            BashUtils.run(f"git checkout {self.REPO_SHA}", expected_return_code=0)
        # end with
        return

    def prepare_data(self):
        if not self.use_latest:
            for t in range(13, 18):
                exp_dir = self.work_dir / f"{t}{t+1}-train"
                self.logger.info(f"Preparing the data for {t}-{t+1} at {exp_dir}")
                IOUtils.rm_dir(exp_dir)
                IOUtils.mk_dir(exp_dir)

                # Copy train data
                BashUtils.run(f"cp -r {self.model_data_dir}/20{t}-20{t+1}-train/train {exp_dir}/", expected_return_code=0)

                # Copy val test data
                BashUtils.run(f"cp -r {self.model_data_dir}/20{t+1}-20{t+2}-val/valid {exp_dir}/", expected_return_code=0)
                BashUtils.run(f"cp -r {self.model_data_dir}/20{t+2}-20{t+3}-test/test {exp_dir}/", expected_return_code=0)

                # Copy vocab
                BashUtils.run(f"cp {self.model_data_dir}/20{t}-20{t+1}-train/vocab* {exp_dir}/", expected_return_code=0)
                # end for
            # end for
        else:
            exp_dir = self.work_dir / "latest"
            IOUtils.rm_dir(exp_dir)
            IOUtils.mk_dir(exp_dir)
            # Copy Train data
            BashUtils.run(f"cp -r {self.model_data_dir}/latest/train {exp_dir}/", expected_return_code=0)

            BashUtils.run(f"cp -r {self.model_data_dir}/latest/valid {exp_dir}/", expected_return_code=0)
            BashUtils.run(f"cp -r {self.model_data_dir}/latest/test {exp_dir}/", expected_return_code=0)

            # Copy vocab
            BashUtils.run(f"cp {self.model_data_dir}/latest/vocab* {exp_dir}/", expected_return_code=0)
        return

    def prepare_configs_and_scripts(self, trials: List[int]):
        base_config = IOUtils.load(self.base_config_file, IOUtils.Format.jsonPretty)
        if not self.use_latest:
            exps = [f"{t}{t+1}-train" for t in range(13, 18)]
            for exp in exps:
                exp_dir = self.work_dir/exp
                for trial in trials:
                    trial_dir = exp_dir/f"trial-{trial}"
                    IOUtils.mk_dir(trial_dir)

                    output_file = trial_dir / "output.txt"

                    config = copy.copy(base_config)
                    config["data_dir"] = str(exp_dir)
                    config["model_dir"] = str(trial_dir/"model")
                    config["output"] = str(output_file)

                    config_file = trial_dir/"config.json"
                    # TODO change to yaml
                    IOUtils.dump(config_file, config, IOUtils.Format.jsonPretty)

                    train_script_file = trial_dir/"train.sh"
                    train_script = f"#!/bin/bash\n" \
                                   f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                                   f"conda activate {self.CONDA_ENV}\n" \
                                   f"module load cuda/10.0 cudnn/7.6.2\n" \
                                   f"cd {self.code_dir}/translate\n" \
                                   f"python3 __main__.py {config_file} --train -v --gpu-id $1 &> {trial_dir}/log-train.txt\n"
                    IOUtils.dump(train_script_file, train_script, IOUtils.Format.txt)
                    BashUtils.run(f"chmod +x {train_script_file}", expected_return_code=0)

                    test_script_file = trial_dir/"test.sh"
                    test_script = f"#!/bin/bash\n" \
                                  f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                                  f"conda activate {self.CONDA_ENV}\n" \
                                  f"module load cuda/10.0 cudnn/7.6.2\n" \
                                  f"cd {self.code_dir}/translate\n" \
                                  f"python3 __main__.py {config_file} --eval {exp_dir}/test/test.token.code {exp_dir}/test/test.token.sbt {exp_dir}/test/test.token.nl &> {trial_dir}/log-test.txt"
                    IOUtils.dump(test_script_file, test_script, IOUtils.Format.txt)
                    BashUtils.run(f"chmod +x {test_script_file}", expected_return_code=0)

                    eval_script_file = trial_dir/"val.sh"
                    eval_script = f"#!/bin/bash\n" \
                                   f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                                   f"conda activate {self.CONDA_ENV}\n" \
                                   f"module load cuda/10.0 cudnn/7.6.2\n" \
                                   f"cd {self.code_dir}/translate\n" \
                                   f"python3 Bleu.py {exp_dir}/test/test.token.nl {trial_dir}/output.txt {trial_dir}\n"
                    IOUtils.dump(eval_script_file, eval_script, IOUtils.Format.txt)
                    BashUtils.run(f"chmod +x {eval_script_file}", expected_return_code=0)
                # end for
            # end for
        else:
            exp_dir = self.work_dir / "latest"
            for trial in trials:
                trial_dir = exp_dir / f"trial-{trial}"
                IOUtils.mk_dir(trial_dir)

                output_file = trial_dir / "output.txt"

                config = copy.copy(base_config)
                config["data_dir"] = str(exp_dir)
                config["model_dir"] = str(trial_dir / "model")
                config["output"] = str(output_file)

                config_file = trial_dir / "config.json"
                IOUtils.dump(config_file, config, IOUtils.Format.jsonPretty)

                train_script_file = trial_dir / "train.sh"
                train_script = f"#!/bin/bash\n" \
                               f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                               f"conda activate {self.CONDA_ENV}\n" \
                               f"module load cuda/10.0 cudnn/7.6.2\n" \
                               f"cd {self.code_dir}/translate\n" \
                               f"python3 __main__.py {config_file} --train -v --gpu-id $1 &> {trial_dir}/log-train.txt\n"
                IOUtils.dump(train_script_file, train_script, IOUtils.Format.txt)
                BashUtils.run(f"chmod +x {train_script_file}", expected_return_code=0)

                test_script_file = trial_dir / "test.sh"
                test_script = f"#!/bin/bash\n" \
                              f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                              f"conda activate {self.CONDA_ENV}\n" \
                              f"module load cuda/10.0 cudnn/7.6.2\n" \
                              f"cd {self.code_dir}/translate\n" \
                              f"python3 __main__.py {config_file} --eval {exp_dir}/test/test.token.code {exp_dir}/test/test.token.sbt {exp_dir}/test/test.token.nl &> {trial_dir}/log-test.txt"
                IOUtils.dump(test_script_file, test_script, IOUtils.Format.txt)
                BashUtils.run(f"chmod +x {test_script_file}", expected_return_code=0)

                eval_script_file = trial_dir / "val.sh"
                eval_script = f"#!/bin/bash\n" \
                              f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                              f"conda activate {self.CONDA_ENV}\n" \
                              f"module load cuda/10.0 cudnn/7.6.2\n" \
                              f"cd {self.code_dir}/translate\n" \
                              f"python3 Bleu.py {exp_dir}/test/test.token.nl {trial_dir}/output.txt {trial_dir}\n"
                IOUtils.dump(eval_script_file, eval_script, IOUtils.Format.txt)
                BashUtils.run(f"chmod +x {eval_script_file}", expected_return_code=0)
        return

