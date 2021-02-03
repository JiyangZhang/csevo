from pathlib import Path


from seutil import LoggingUtils, IOUtils, BashUtils


from csevo.Environment import Environment
from csevo.Macros import Macros


class TransformerProcessor:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)


    def __init__(self):
        super(TransformerProcessor, self).__init__()
        return

    def process_data(self, model_data_dir: Path, data_prefix: str):
        """
        Assume we have the raw data file generated bu Bi-LSTM model processor: src-train.txt, tgt-train.txt, src-val.txt, tgt-val.txt
        :param model_data_dir: the dir for storing the data for transformer.
        :param data_prefix: e.g. evo-2020, mixedproj-2020
        :return:
        """
        self.logger.info(f"Start processing")

        BashUtils.run(f"onmt_preprocess -train_src {model_data_dir}/{data_prefix}-{Macros.train}/src-train.txt "
                      f"-train_tgt {model_data_dir}/{data_prefix}-{Macros.train}/tgt-train.txt "
                      f"-valid_src {model_data_dir}/{data_prefix}-{Macros.val}/src-val.txt "
                      f"-valid_tgt {model_data_dir}/{data_prefix}-{Macros.val}/tgt-val.txt "
                      f"-save_data {model_data_dir}/{data_prefix}-{Macros.train}/transformer --src_seq_length 200"
                      f" --src_seq_length_trunc 200 --shard_size 0", expected_return_code=0)