import abc
from typing import *

from pathlib import Path

from seutil import LoggingUtils

from csevo.data.MethodData import MethodData
from csevo.Environment import Environment


class AbstractFilter:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)
    YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

    def __init__(self):
        return

    @abc.abstractmethod
    def process_data(self, project_dir: Path):
        """
        Processes the list of yearly method data to produce evo data.
        :param revision_file: the file name of originally collected project-revision file.
        :param method_file: the file name of method data.
        :param output_dir: the directory to put the processed data, prepared for this model.
        :param which: specify which filter function is used here.
        """
        raise NotImplementedError