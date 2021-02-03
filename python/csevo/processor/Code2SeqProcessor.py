import pickle
from typing import *

from pathlib import Path
import os
import numpy as np
from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.data.MethodData import MethodData
from csevo.Environment import Environment
from csevo.processor.AbstractProcessor import AbstractProcessor

from .common import Common


class Code2SeqProcessor(AbstractProcessor):
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, max_data_contexts=1000, max_contexts=200, subtoken_vocab_size=186277, target_vocab_size=26347):
        super(Code2SeqProcessor, self).__init__()
        self.max_data_contexts = max_data_contexts
        self.max_contexts = max_contexts
        self.subtoken_vocab_size = subtoken_vocab_size
        self.target_vocab_size = target_vocab_size
        return

    def process_data(self, method_data_list: List[MethodData], data_type: str, output_dir: Path, traversal=None) -> List[int]:
        Environment.require_collector()

        log_file = output_dir / "collector-log.txt"
        data_file = output_dir / "method-data.json"
        IOUtils.dump(data_file, IOUtils.jsonfy(method_data_list), IOUtils.Format.json)

        config = {
            "transform": True,
            "model": "Code2Seq",
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
        # build dataset
        num_training_examples = 0
        num_examples = self.process_file(file_path=output_dir, data_file_role=data_type,
                                         max_contexts=int(self.max_contexts),
                                         max_data_contexts=int(self.max_data_contexts))
        # for training data: build histogram, build vocab and save
        if data_type == "train":
            num_training_examples = num_examples
            self.create_vocab(config, num_training_examples)

        error_ids = IOUtils.load(str(output_dir)+"-error-ids.json")
        # BashUtils.run(f"rm {output_dir}-error-ids.json", expected_return_code=0)
        return error_ids

    def create_vocab(self, config, num_examples: int):
        """From traning data, create the vocabulary and dictrionary"""
        output_dir = config["outputDir"]
        rr = BashUtils.run(f"csevo/processor/histogram.sh {output_dir}")
        if rr.stdout:
            self.logger.warning(f"Stdout of collector:\n{rr.stdout}")
            # end if
        if rr.stderr:
            self.logger.warning(f"Stderr of collector:\n{rr.stderr}")
            # end if
        subtoken_histogram_path = output_dir + "/histo.ori.c2s"
        node_histogram_path = output_dir + "/histo.node.c2s"
        target_histogram = output_dir + "/histo.tgt.c2s"
        subtoken_to_count = Common.load_histogram(subtoken_histogram_path,
                                                  max_size=int(self.subtoken_vocab_size))
        node_to_count = Common.load_histogram(node_histogram_path,
                                              max_size=None)
        target_to_count = Common.load_histogram(target_histogram,
                                                max_size=int(self.target_vocab_size))
        print('subtoken vocab size: ', len(subtoken_to_count))
        print('node vocab size: ', len(node_to_count))
        print('target vocab size: ', len(target_to_count))
        save_dict_file_path = output_dir + '/code2seq.dict.c2s'
        with open(save_dict_file_path, 'wb') as file:
            pickle.dump(subtoken_to_count, file)
            pickle.dump(node_to_count, file)
            pickle.dump(target_to_count, file)
            pickle.dump(self.max_data_contexts, file)
            pickle.dump(num_examples, file)
            print(f'Dictionaries saved to: {save_dict_file_path}')
        # delete histo data file
        os.system("rm "+subtoken_histogram_path+" "+node_histogram_path+" "+target_histogram)

    def process_file(self, file_path, data_file_role, max_contexts, max_data_contexts):
        sum_total = 0
        sum_sampled = 0
        total = 0
        max_unfiltered = 0
        max_contexts_to_sample = max_data_contexts if data_file_role == 'train' else max_contexts
        output_path = f'{file_path}/code2seq.{data_file_role}.c2s'
        with open(output_path, 'w') as outfile:
            with open(file_path/(data_file_role + ".raw.txt"), 'r') as file:
                for line in file:
                    parts = line.rstrip('\n').split(' ')
                    target_name = parts[0]
                    contexts = parts[1:]
                    if len(contexts) > max_unfiltered:
                        max_unfiltered = len(contexts)
                    contexts = [cotxt.replace(target_name, "MethodName") for cotxt in contexts]

                    sum_total += len(contexts)
                    if len(contexts) > max_contexts_to_sample:
                        contexts = np.random.choice(contexts, max_contexts_to_sample, replace=False)

                    sum_sampled += len(contexts)

                    csv_padding = " " * (max_data_contexts - len(contexts))
                    total += 1
                    outfile.write(target_name + ' ' + " ".join(contexts) + csv_padding + '\n')
        # delete useless files

        print('File: ' + output_path)
        print('Average total contexts: ' + str(float(sum_total) / total))
        print('Average final (after sampling) contexts: ' + str(float(sum_sampled) / total))
        print('Total examples: ' + str(total))
        print('Max number of contexts per word: ' + str(max_unfiltered))
        return total
