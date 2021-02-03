from typing import *

import copy
from pathlib import Path

import yaml
from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.ml.TACCRunner import TACCRunnerConsts


class TransformerRunner:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, work_dir: Path, year: int, eval_setting: str):
        self.year = year
        self.eval_setting = eval_setting
        self.work_dir: Path = work_dir / f"{self.eval_setting}-{self.year}"
        self.model_data_dir: Path = Macros.data_dir / "models-data" / "Transformer"
        self.base_config_file: Path = Macros.python_dir / "configs" / "Transformer.yaml"
        self.code_dir: Path = self.work_dir / "code"
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

        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.train}/transformer.* {self.data_dir}/",
                      expected_return_code=0)

        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_common}/src-test.txt {self.data_dir}/src-{Macros.test_common}.txt", expected_return_code=0)
        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_common}/tgt-test.txt {self.data_dir}/tgt-{Macros.test_common}.txt", expected_return_code=0)

        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_standard}/src-test.txt {self.data_dir}/src-{Macros.test_standard}.txt", expected_return_code=0)
        BashUtils.run(f"cp {self.model_data_dir}/{data_prefix}-{Macros.test_standard}/tgt-test.txt {self.data_dir}/tgt-{Macros.test_standard}.txt", expected_return_code=0)

        return

    def prepare_configs_and_scripts(self, trials: List[int]):

        exp_dir = self.work_dir
        for trial in trials:
            trial_dir = exp_dir/f"trial-{trial}"
            IOUtils.mk_dir(trial_dir)

            train_script_file = trial_dir/"train.sh"
            train_script = f"#!/bin/bash\n" \
                           f"source {TACCRunnerConsts.conda_init_path[TACCRunnerConsts.get_cur_cluster()]}\n" \
                           f"module load cuda/10.1 cudnn/7.6.2\n" \
                           f"conda activate {self.CONDA_ENV}\n" \
                           f"cd {self.code_dir}\n" \
                           f"export MKL_SERVICE_FORCE_INTEL=1\n"\
                           f"python3 train.py " \
                           f"-data {self.data_dir}/transformer -save_model {trial_dir}/bestTransformer "\
                           f"-layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 "\
                           f"-encoder_type transformer -decoder_type transformer -position_encoding "\
                           f"-train_steps 50000  -max_generator_batches 2 -dropout 0.1 "\
                           f"-batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 "\
                           f"-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 " \
                           f"-max_grad_norm 0 -param_init 0 -param_init_glorot -early_stopping 10 -keep_checkpoint 1 " \
                           f"-label_smoothing 0.1 -valid_steps 500 -save_checkpoint_steps 500 -report_every 500 " \
                           f"--world_size 1 --gpu_ranks 0 " \
                           f"&> {trial_dir}/train-log.txt\n"
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
