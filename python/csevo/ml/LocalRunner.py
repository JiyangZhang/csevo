from typing import *

import os
from pathlib import Path
import time

from seutil import LoggingUtils, BashUtils, IOUtils

from csevo.Environment import Environment
from csevo.Macros import Macros


class LocalRunner:
    """
    Helper class for running things on the TACC clusters.

    Currently support Stampede2 and Maverick2.
    """

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        return

    def prepare_model(self, model: str, use_latest: bool = False, debug:bool = False, cross_proj: bool=False):
        if not use_latest:
            model_work_dir = self.work_dir / model
        else:
            model_work_dir = self.work_dir / f"{model}-latest"
            # end if
        if cross_proj:
            model_work_dir = Path(f"{model_work_dir}-cross-proj")
        if debug:
            model_work_dir = Path(f"{model_work_dir}-debug")

        IOUtils.mk_dir(model_work_dir)

        if model == "DeepCom":
            from csevo.ml.DeepComRunner import DeepComRunner
            runner = DeepComRunner(model_work_dir, use_latest)
        elif model == "Seq2seq":
            from csevo.ml.Seq2seqRunner import Seq2seqRunner
            runner = Seq2seqRunner(model_work_dir, use_latest)
        elif model == "Seq2seqAtt":
            from csevo.ml.Seq2seqAttRunner import Seq2seqAttRunner
            runner = Seq2seqAttRunner(model_work_dir, use_latest)
        elif model == "DeepCom-SBT":
            from csevo.ml.DeepComSBTRunner import DeepComSBTRunner
            runner = DeepComSBTRunner(model_work_dir, use_latest)
        elif model == "DeepCom-Preorder":
            from csevo.ml.DeepComPreorderRunner import DeepComPreorderRunner
            runner = DeepComPreorderRunner(model_work_dir, use_latest)
        elif model == "Code2Seq":
            from csevo.ml.Code2SeqRunner import Code2SeqRunner
            runner = Code2SeqRunner(model_work_dir, use_latest, debug, cross_proj)
        elif model == "Bi-LSTM":
            from csevo.ml.BiLSTMRunner import BiLSTMRunner
            runner = BiLSTMRunner(model_work_dir, use_latest, debug, cross_proj)
        elif model == "no-split-Bi-LSTM":
            from csevo.ml.NoSplitBiLSTMRunner import BiLSTMRunner
            runner = BiLSTMRunner(model_work_dir, use_latest, debug, cross_proj)
        elif model == "Transformer":
            from csevo.ml.TransformerRunner import TransformerRunner
            runner = TransformerRunner(model_work_dir, use_latest)
        else:
            raise ValueError(f"Model {model} not ready yet")
        # end if

        runner.prepare()
        return