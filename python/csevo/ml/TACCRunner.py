"""
This script is for Jiyang's TACC account
"""
from typing import *

import os
from pathlib import Path
import re
import tempfile
import time
import traceback

from seutil import LoggingUtils, BashUtils, IOUtils

from csevo.Environment import Environment
from csevo.Macros import Macros

# TODO private info


class TACCRunnerConsts:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    maverick2 = "maverick2"
    stampede2 = "stampede2"

    user = os.getenv('USER')
    work = os.getenv('WORK')
    home = os.getenv('HOME')

    submit_cd = 600  # 10min
    tacc_logs_dir = Macros.python_dir / "tacc-logs"

    conda_init_path = {
        maverick2: f"{work}/miniconda3/etc/profile.d/conda.sh",
        stampede2: f"{work}/opt/anaconda3/etc/profile.d/conda.sh",
    }
    allocation = "Coq-CC"
    if user == "pynie":
        conda_init_path["maverick2"] = f"{work}/opt/anaconda3/etc/profile.d/conda.sh"

    conda_env = {
        maverick2: "csevo",
        stampede2: "csevo",
    }

    queue = {
        maverick2: "gtx",
        stampede2: "normal",
    }

    queue_limit = {
        maverick2: 8,
        stampede2: 50,
    }

    modules = {
        maverick2: ["cuda/10.1"],
        stampede2: [],
    }

    timeout = {
        maverick2: "12:00:00",
        stampede2: "06:00:00",
    }

    max_timeout_hour = {
        maverick2: 24,
        stampede2: 48,
    }

    @classmethod
    def get_cur_cluster(cls) -> str:
        hostname = BashUtils.run(f"hostname").stdout.strip()
        if hostname.endswith("maverick2.tacc.utexas.edu"):
            return cls.maverick2
        elif hostname.endswith("stampede2.tacc.utexas.edu"):
            return cls.stampede2
        else:
            cls.logger.warning("Currently not on TACC")
            return cls.maverick2
        # end if


class TACCRunner:
    """
    Helper class for running things on the TACC clusters.

    Currently support Stampede2 and Maverick2.
    """

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        return

    def prepare_model(self, model: str, year: int, eval_setting: str, debug: bool = False):
        sub_dir_name = f"{model}"
        if debug:
            sub_dir_name = f"{sub_dir_name}-debug"
        model_work_dir = self.work_dir / sub_dir_name

        IOUtils.mk_dir(model_work_dir)

        if model == "DeepCom":
            from csevo.ml.DeepComRunner import DeepComRunner
            runner = DeepComRunner(model_work_dir, year, eval_setting)
        elif model == "Seq2seq":
            from csevo.ml.Seq2seqRunner import Seq2seqRunner
            runner = Seq2seqRunner(model_work_dir, year, eval_setting)
        elif model == "Seq2seqAtt":
            from csevo.ml.Seq2seqAttRunner import Seq2seqAttRunner
            runner = Seq2seqAttRunner(model_work_dir, year, eval_setting)
        elif model == "DeepCom-SBT":
            from csevo.ml.DeepComSBTRunner import DeepComSBTRunner
            runner = DeepComSBTRunner(model_work_dir, year, eval_setting)
        elif model == "DeepCom-Preorder":
            from csevo.ml.DeepComPreorderRunner import DeepComPreorderRunner
            runner = DeepComPreorderRunner(model_work_dir, year, eval_setting)
        elif model == "Code2Seq":
            from csevo.ml.Code2SeqRunner import Code2SeqRunner
            runner = Code2SeqRunner(model_work_dir, year, eval_setting)
        elif model == "Bi-LSTM":
            from csevo.ml.BiLSTMRunner import BiLSTMRunner
            runner = BiLSTMRunner(model_work_dir, year, eval_setting)
        elif model == "no-split-Bi-LSTM":
            from csevo.ml.NoSplitBiLSTMRunner import BiLSTMRunner
            runner = BiLSTMRunner(model_work_dir, year, eval_setting)
        elif model == "Transformer":
            from csevo.ml.TransformerRunner import TransformerRunner
            runner = TransformerRunner(model_work_dir, year, eval_setting)
        else:
            raise ValueError(f"Model {model} not ready yet")
        # end if

        runner.prepare()
        return

    def run_models(
            self,
            mode: str,
            models: List[str],
            exps: List[str],
            trials: List[int],
            timeout_hour: Optional[int],
            beg: int = 0,
            cnt: int = -1,
    ):
        """
        :param mode: train
        :param models: DeepCom
        :param exps: evolution-evolution ...
        :param trials: 0, 1, 2 ...
        :param timeout_hour: 24
        """
        if mode not in [Macros.train, Macros.test_common, Macros.test_standard]:
            raise ValueError(f"mode has to be one of {Macros.train}, {Macros.test_common}, {Macros.test_standard}")
        # end if

        assert beg >= 0
        assert cnt >= -1

        # Sort the models, exps, and trials lists to ensure the traversal order is stable
        models.sort()
        exps.sort()
        trials.sort()

        # Assuming each model uses one GPU
        scripts = list()  # a list of commands
        cur_script = ""
        cur_script_cnt = 0
        total_script_cnt = 0
        for model in models:
            model_work_dir = self.work_dir / model

            for exp in exps:
                for trial in trials:
                    # Only output the jobs whose indexes are in the interval [beg, beg+cnt)
                    if (total_script_cnt < beg) or (cnt > 0 and total_script_cnt >= beg + cnt):
                        total_script_cnt += 1
                        continue

                    cur_script += f"CUDA_VISIBLE_DEVICES={cur_script_cnt} {model_work_dir}/{exp}/trial-{trial}/{mode}.sh {cur_script_cnt}&\n"
                    cur_script_cnt += 1
                    if cur_script_cnt == 4:
                        cur_script += "wait\n"
                        scripts.append(cur_script)
                        cur_script = ""
                        cur_script_cnt = 0

                    total_script_cnt += 1

        if cur_script_cnt != 0:
            cur_script += "wait\n"
            scripts.append(cur_script)
            cur_script = ""
            cur_script_cnt = 0

        self.submit_multi_scripts(cluster=TACCRunnerConsts.get_cur_cluster(),
            name=f"csevo-{mode}",
            log_path=Macros.ml_logs_dir/f"csevo-{mode}",
            scripts=scripts,
            timeout=f"{timeout_hour:02}:00:00",
        )
        return


    def run_models_local(
            self,
            mode: str,
            models: List[str],
            exps: List[str],
            trials: List[int],
            timeout_hour: Optional[int],
            beg: int = 0,
            cnt: int = -1,
    ):
        """
        :param mode: train
        :param models: DeepCom
        :param exps: evolution-evolution ...
        :param trials: 0, 1, 2 ...
        :param timeout_hour: 24
        """
        if mode not in [Macros.train, Macros.test_common, Macros.test_standard]:
            raise ValueError(f"mode has to be one of {Macros.train}, {Macros.test_common}, {Macros.test_standard}")
        # end if

        assert beg >= 0
        assert cnt >= -1

        # Sort the models, exps, and trials lists to ensure the traversal order is stable
        models.sort()
        exps.sort()
        trials.sort()

        user = os.getenv("USER")
        home = os.getenv("HOME")
        re_work_dir = re.compile(rf"/work/\d+/{user}/maverick2")

        # Assuming each model uses one GPU
        total_script_cnt = 0
        for model in models:
            model_work_dir = self.work_dir / model

            for exp in exps:
                for trial in trials:
                    # Only output the jobs whose indexes are in the interval [beg, beg+cnt)
                    if (total_script_cnt < beg) or (cnt > 0 and total_script_cnt >= beg + cnt):
                        total_script_cnt += 1
                        continue

                    trial_dir = model_work_dir/exp/f"trial-{trial}"

                    # Modify the script to remove TACC stuff
                    script = IOUtils.load(trial_dir/f"{mode}.sh", IOUtils.Format.txt)
                    script = script.replace("\nmodule", "\n# module")
                    script = re_work_dir.sub(home, script)

                    # Replace the paths in config files as well
                    orig_configs = dict()
                    for config_file in trial_dir.glob("config*.json"):
                        config_content = IOUtils.load(config_file, IOUtils.Format.txt)
                        orig_configs[config_file] = config_content
                        config_content = re_work_dir.sub(home, config_content)
                        IOUtils.dump(config_file, config_content, IOUtils.Format.txt)

                    # Try to execute the script
                    try:
                        self.logger.info(f"Executing: {script}")
                        fd, fname = tempfile.mkstemp(suffix=".sh")
                        IOUtils.dump(fname, script, IOUtils.Format.txt)
                        os.close(fd)
                        BashUtils.run(f"chmod +x {fname}", expected_return_code=0)
                        BashUtils.run(f"{fname} 0\n", expected_return_code=0)
                    except RuntimeError:
                        traceback.print_exc()

                    # Revert the config files
                    for config_file, config_content in orig_configs.items():
                        IOUtils.dump(config_file, config_content, IOUtils.Format.txt)

                    total_script_cnt += 1
        return

    @classmethod
    def get_num_running_jobs(cls) -> int:
        return int(BashUtils.run(f"squeue -u {TACCRunnerConsts.user} | wc -l", expected_return_code=0).stdout) - 1

    @classmethod
    def run_script(cls, script: str, timeout_hour: int = 2):
        cluster = TACCRunnerConsts.get_cur_cluster()
        cls.submit_script(
            cluster=cluster,
            name="csevo-run",
            log_path=TACCRunnerConsts.tacc_logs_dir/"csevo-run",
            script=script,
            timeout=f"{timeout_hour:02}:00:00",
        )
        return

    @classmethod
    def submit_multi_scripts(cls,
            cluster: str,
            name: str,
            log_path: Path,
            scripts: List[str],
            queue: str = None,
            timeout: str = None,
            require_conda: bool = True,
            conda_env: str = None,
            modules: List[str] = None,
    ) -> List[int]:
        cls.logger.info(f"In total {len(scripts)} scripts to submit")
        sid = 0
        job_ids = list()
        queue_limit = TACCRunnerConsts.queue_limit[cluster]
        while sid < len(scripts):
            if cls.get_num_running_jobs() >= queue_limit:
                cls.logger.warning(f"Number of running jobs reach limit {queue_limit}, will retry after {TACCRunnerConsts.submit_cd} seconds at {time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.localtime(time.time()+TACCRunnerConsts.submit_cd))}")
                time.sleep(TACCRunnerConsts.submit_cd)
                continue
            # end if

            cls.logger.info(f"Submitting job #{sid+1}/{len(scripts)}")

            script = scripts[sid]
            try:
                job_id = cls.submit_script(cluster, name, log_path, script, queue, timeout, require_conda, conda_env, modules)
            except KeyboardInterrupt:
                raise
            except:
                cls.logger.warning(f"Fail to submit script {script}\n"
                                   f"Will retry after {TACCRunnerConsts.submit_cd} seconds at {time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.localtime(time.time()+TACCRunnerConsts.submit_cd))}")
                time.sleep(TACCRunnerConsts.submit_cd)
                continue
            # end try

            cls.logger.info(f"Submitted job #{sid+1} with job id {job_id}")
            job_ids.append(job_id)
            sid += 1
        # end while
        return job_ids

    @classmethod
    def submit_script(cls,
            cluster: str,
            name: str,
            log_path: Path,
            script: str,
            queue: str = None,
            timeout: str = None,
            require_conda: bool = True,
            conda_env: str = None,
            modules: List[str] = None,
    ) -> int:
        # Get default values
        if modules is None:
            modules = TACCRunnerConsts.modules[cluster]
        # end if
        if queue is None:
            queue = TACCRunnerConsts.queue[cluster]
        # end if
        if timeout is None:
            timeout = TACCRunnerConsts.timeout[cluster]
        # end if
        if conda_env is None:
            conda_env = TACCRunnerConsts.conda_env[cluster]
        # end if

        # Prepare submit script
        IOUtils.mk_dir(log_path)

        s = f"""#!/bin/bash
#SBATCH -J {name}               # Job name
#SBATCH -o {log_path}/%j.stdout # Name of stdout output file(%j expands to jobId)
#SBATCH -e {log_path}/%j.stderr # Name of stderr output file(%j expands to jobId)
#SBATCH -p {queue}              # Queue name
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t {timeout}            # Max run time (hh:mm:ss)
#SBATCH --mail-user=jiyang.zhang@utexas.edu
#SBATCH --mail-type=ALL
# The next line is required if the user has more than one project
#SBATCH -A {TACCRunnerConsts.allocation}      # Allocation name to charge job against

module reset
module unload python2
"""
        for m in modules:
            s += f"module load {m}\n"
        # end for
        s += f"""
module list
echo "START: $(date)"

# Launch serial code...
# Do not use ibrun or any other MPI launcher
"""

        if require_conda:
            s += f"""
unset PYTHONPATH
source {TACCRunnerConsts.conda_init_path[cluster]}
conda activate {conda_env}
"""

        s += f"""
cd {Macros.python_dir}
{script}

echo "END: $(date)"
"""

        # Submit the script
        submit_script = BashUtils.get_temp_file()
        IOUtils.dump(submit_script, s, IOUtils.Format.txt)
        receipt = BashUtils.run(f"sbatch {submit_script}", expected_return_code=0).stdout

        # Get job id as the last number in output
        job_id = int(receipt.splitlines()[-1].split()[-1])

        # Save the script at log_path as well
        BashUtils.run(f"mv {submit_script} {log_path}/{job_id}.sh")

        return job_id
