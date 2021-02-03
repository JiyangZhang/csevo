from typing import *

import abc
from pathlib import Path

from seutil import LoggingUtils

from csevo.data.MethodData import MethodData
from csevo.Environment import Environment


class AbstractProcessor:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self):
        return

    @abc.abstractmethod
    def process_data(self, method_data_list: List[dict], data_type: str, output_dir: Path, traversal="None") -> List[int]:
        """
        Processes the list of method data, for the given data_type.
        :param method_data_list: list of MethodData
        :param data_type: the data_type (one of {train, val, test})
        :param output_dir: the directory to put the processed data, prepared for this model
        :return: the list of data indexes (in the method_data_list) that failed to process
        """
        raise NotImplementedError
