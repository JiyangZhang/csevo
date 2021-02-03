from typing import *
import sys
import collections
import javalang
import nltk
from nltk.tokenize import word_tokenize
import multiprocessing
from pathlib import Path
import re
from tqdm import tqdm
import javalang
import os

from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.data.MethodData import MethodData
from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.processor.AbstractProcessor import AbstractProcessor


class BiLSTMProcessor:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    # As used in journal version
    MAX_VOCAB = 30000

    modifiers = ['public', 'private', 'protected', 'static']

    RE_WORDS = re.compile(r'''
            # Find words in a string. Order matters!
            [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
            [A-Z]?[a-z]+ |  # Capitalized words / all lower case
            [A-Z]+ |  # All upper case
            \d+ | # Numbers
            .+
        ''', re.VERBOSE)

    def __init__(self):
        super(BiLSTMProcessor, self).__init__()
        nltk.download('punkt')
        return

    def process_data(self, method_data_list: List[MethodData], data_type: str, output_dir: Path, split: bool=True):
        Environment.require_collector()

        log_file = output_dir / "collector-log.txt"
        data_file = output_dir / "method-data.json"
        IOUtils.dump(data_file, IOUtils.jsonfy(method_data_list), IOUtils.Format.json)

        config = {
            "transform": True,
            "model": "BiLSTM",
            "dataType": data_type,
            "dataFile": str(data_file),
            "logFile": str(log_file),
            "outputDir": str(output_dir),
        }
        config_file = output_dir / "collector-config.json"
        IOUtils.dump(config_file, config, IOUtils.Format.jsonPretty)

        self.logger.info(f"Starting the Java collector. Check log at {log_file} and outputs at {output_dir}")
        rr = BashUtils.run(f"java -jar {Environment.collector_jar} {config_file}", expected_return_code=0)
        if rr.stdout:
            self.logger.warning(f"Stdout of collector:\n{rr.stdout}")
        # end if
        if rr.stderr:
            self.logger.warning(f"Stderr of collector:\n{rr.stderr}")
        # end if
        # build raw dataset
        if split:
            self.tokenizeFile(output_dir/f"{data_type}.raw.txt", data_type)
        else:
            self.noSplit(output_dir/f"{data_type}.raw.txt", data_type)

        error_ids = IOUtils.load(str(output_dir) + "-error-ids.json")
        print(f"Number of error id is: {len(error_ids)}")
        # BashUtils.run(f"rm {output_dir}-error-ids.json", expected_return_code=0)
        return error_ids

    def split_subtokens(self, str):
        return [subtok for subtok in self.RE_WORDS.findall(str) if not subtok == '_']


    def tokenizeFile(self, file_path: Path, data_type: str):
        lines = 0
        with open(file_path, 'r', encoding="utf-8") as file:
            with open(os.path.dirname(file_path)+f'/tgt-{data_type}.txt', 'w') as method_names_file:
                with open(os.path.dirname(file_path)+f'/src-{data_type}.txt', 'w') as method_contents_file:
                    for line in file:
                        lines += 1
                        line = line.rstrip()
                        parts = line.split('|', 1)
                        method_name = parts[0]
                        method_content = parts[1]
                        try:
                            tokens = list(javalang.tokenizer.tokenize(method_content))
                        except:
                            tokens = method_content.split(' ')
                        if len(method_name) > 0 and len(tokens) > 0:
                            try:
                                method_contents_file.write(' '.join(
                                    [' '.join(self.split_subtokens(i.value)) for i in tokens if not i.value in self.modifiers]) + '\n')
                                method_names_file.write(method_name + '\n')
                            except :
                                method_contents_file.write(" ".join(method_content.split()) + '\n')
                                method_names_file.write(method_name + '\n')
                        else:
                            print('ERROR in len of: ' + method_name + ', tokens: ' + str(tokens))
        print(str(lines))

    def noSplit(self, file_path: Path, data_type: str):
        lines = 0
        with open(file_path, 'r', encoding="utf-8") as file:
            with open(os.path.dirname(file_path) + f'/tgt-{data_type}.txt', 'w') as method_names_file:
                with open(os.path.dirname(file_path) + f'/src-{data_type}.txt', 'w') as method_contents_file:
                    for line in file:
                        lines += 1
                        line = line.rstrip()
                        parts = line.split('|', 1)
                        method_name = parts[0]
                        method_content = parts[1]
                        try:
                            tokens = list(javalang.tokenizer.tokenize(method_content))
                        except:
                            print('ERROR in tokenizing: ' + method_content)
                            # tokens = method_content.split(' ')
                        if len(method_name) > 0 and len(tokens) > 0:
                            try:
                                method_contents_file.write(' '.join([i.value for i in tokens]) + '\n')
                                method_names_file.write(method_name + '\n')
                            except:
                                method_contents_file.write(method_content.replace("\t", " ") + '\n')
                                method_names_file.write(method_name + '\n')
                        else:
                            print('ERROR in len of: ' + method_name + ', tokens: ' + str(tokens))
        print(str(lines))
