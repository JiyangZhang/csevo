from typing import *

import collections
import javalang
import nltk
from nltk.tokenize import word_tokenize
import multiprocessing
from pathlib import Path
import re
from tqdm import tqdm

from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.data.MethodData import MethodData
from csevo.Environment import Environment
from csevo.Macros import Macros
from csevo.processor.AbstractProcessor import AbstractProcessor


class DeepComProcessor(AbstractProcessor):
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    # As used in journal version
    MAX_VOCAB = 30000

    def __init__(self):
        super(DeepComProcessor, self).__init__()
        nltk.download('punkt')
        return

    def process_data(self, method_data_list, data_type, output_dir, traversal) -> List[int]:
        self.logger.info(f"Start processing")

        # Use DeepCom's required names
        data_type = {
            Macros.train: "train",
            Macros.val: "valid",
            Macros.test: "test",
            "debug": "debug",
        }[data_type]

        # Initialize vocab, error_ids (shared between processes)
        manager = multiprocessing.Manager()
        code_vocab = manager.dict()
        nl_vocab = manager.dict()
        sbt_vocab = manager.dict()
        vocabs_lock = manager.Lock()
        error_ids = manager.list()
        error_ids_lock = manager.Lock()

        # Multi-processing, split the tasks evenly
        tasks_each_process = len(method_data_list) // Macros.multi_processing + 1
        processes = list()
        for pid in range(Macros.multi_processing):
            beg = pid * tasks_each_process
            method_data_list_p = method_data_list[beg:beg + tasks_each_process]
            output_dir_p = output_dir / str(pid)
            IOUtils.mk_dir(output_dir_p)
            process = multiprocessing.Process(target=self.process_data_mp,
                                              args=(method_data_list_p, data_type, output_dir_p,
                                                    pid, beg,
                                                    code_vocab, nl_vocab, sbt_vocab, vocabs_lock,
                                                    error_ids, error_ids_lock, traversal
                                                    ))
            process.start()
            processes.append(process)
        # end for

        for process in processes:
            process.join()
        # end for

        # Merge results
        code_file_name = data_type + ".token.code"
        nl_file_name = data_type + ".token.nl"
        sbt_file_name = data_type + ".token.sbt"
        data_type_output_dir = output_dir / data_type
        IOUtils.mk_dir(data_type_output_dir)
        for pid in range(Macros.multi_processing):
            for fname in [code_file_name, nl_file_name, sbt_file_name]:
                BashUtils.run(f"cat {output_dir}/{pid}/{fname} >> {data_type_output_dir}/{fname}")
            # end for
            IOUtils.rm_dir(output_dir / str(pid))
        # end for
        error_ids.sort()

        # Build vocab
        if data_type == "train":
            code_vocab_file = output_dir / "vocab.code"
            nl_vocab_file = output_dir / "vocab.nl"
            sbt_vocab_file = output_dir / "vocab.sbt"
            fcv = open(code_vocab_file, "w+")
            fnv = open(nl_vocab_file, "w+")
            fsv = open(sbt_vocab_file, "w+")
            # write vocab to files
            special_tokens = ['<S>', '</S>', '<UNK>', '<KEEP>', '<DEL>', '<INS>', '<SUB>', '<NONE>']

            # Filter based on frequency, keep first MAX_VOCAB
            code_vocabs_list = special_tokens + list(code_vocab.keys())[:self.MAX_VOCAB]
            nl_vocabs_list = special_tokens + list(nl_vocab.keys())[:self.MAX_VOCAB]
            sbt_vocabs_list = special_tokens + list(sbt_vocab.keys())[:self.MAX_VOCAB]
            for v in code_vocabs_list:
                fcv.write(v + "\n")
            for v in nl_vocabs_list:
                fnv.write(v + "\n")
            for v in sbt_vocabs_list:
                fsv.write(v + "\n")
            fcv.close()
            fsv.close()
            fnv.close()
        # end if

        return list(error_ids)

    def process_data_mp(self, method_data_list: List[dict], data_type: str, output_dir: Path,
                        pid, beg,
                        code_vocab, nl_vocab, sbt_vocab, vocabs_lock,
                        error_ids, error_ids_lock, traversal: str
                        ) -> List[int]:
        code_file = output_dir / (data_type + ".token.code")
        nl_file = output_dir / (data_type + ".token.nl")
        sbt_file = output_dir / (data_type + ".token.sbt")

        fc = open(code_file, "w+")
        fn = open(nl_file, "w+")
        fs = open(sbt_file, "w+")

        for method_i, method in enumerate(tqdm(method_data_list, desc=f"Process {pid}", position=pid)):
            # process code
            code = method["code"].strip()
            tokens = list(javalang.tokenizer.tokenize(code))
            tks = []
            for tk in tokens:
                if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
                    tks.append('str_')
                elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
                    tks.append('num_')
                elif tk.__class__.__name__ == 'Boolean':
                    tks.append('bool_')
                else:
                    tks.append(tk.value)
            sub_code = self.subtokenize_code(tks)

            # process comments
            nl = " ".join(word_tokenize(method["comment_summary"].replace("\n", " "))).lower()

            # process sbt
            code = " ".join(tks)
            sbt = self.ast_to_sbt(code, traversal)
            if sbt == -1:
                with error_ids_lock:
                    error_ids.append(beg + method_i)
                # end with
                self.logger.debug(f"Error in parsing sbt, discard.")
            elif sbt == 0:
                pass
            else:
                # write data to files
                fs.write(sbt + "\n")
                fc.write(sub_code + "\n")
                fn.write(nl + "\n")
                # update the vocab set

                with vocabs_lock:
                    self.update_vocab(code_vocab, sub_code.split())
                    self.update_vocab(nl_vocab, nl.split())
                    self.update_vocab(sbt_vocab, sbt.split())
                # end with
            # end if
        # end for
        fc.close()
        fn.close()
        fs.close()

        self.logger.info(f"Finish processing")

        return error_ids

    @classmethod
    def update_vocab(cls, vocab: dict, seq: list):
        counter = collections.Counter(seq)
        for k, v in counter.items():
            if k not in vocab:
                vocab[k] = v
            else:
                vocab[k] += v
            # end if
        # end for
        return

    def subtokenize_code(self, tokens: List[str]) -> str:
        """Subtokenize the code."""
        subtokens = list()
        for token in tokens:
            curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
            subtokens = subtokens + [c.lower() for c in curr]
        return " ".join(subtokens)

    def ast_to_sbt(self, code: str, traversal: str):
        """Convert the code to SBT. If it encounters error, return -1."""
        if traversal == "None":
            return 0
        code = code.strip()
        tokens = javalang.tokenizer.tokenize(code)
        token_list = list(javalang.tokenizer.tokenize(code))
        length = len(token_list)
        parser = javalang.parser.Parser(tokens)
        try:
            tree = parser.parse_member_declaration()
        except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
            self.logger.debug(f"Can not handle this method: {code}")
            return -1
        flatten = []
        for path, node in tree:
            flatten.append({'path': path, 'node': node})

        ign = False
        outputs = []
        stop = False
        for i, Node in enumerate(flatten):
            d = collections.OrderedDict()
            path = Node['path']
            node = Node['node']
            children = []
            for child in node.children:
                child_path = None
                if isinstance(child, javalang.ast.Node):
                    child_path = path + tuple((node,))
                    for j in range(i + 1, len(flatten)):
                        if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                            children.append(j)
                if isinstance(child, list) and child:
                    child_path = path + (node, child)
                    for j in range(i + 1, len(flatten)):
                        if child_path == flatten[j]['path']:
                            children.append(j)
            d["id"] = i
            d["type"] = str(node)
            if children:
                d["children"] = children
            value = None
            if hasattr(node, 'name'):
                value = node.name
            elif hasattr(node, 'value'):
                value = node.value
            elif hasattr(node, 'position') and node.position:
                for i, token in enumerate(token_list):
                    if node.position == token.position:
                        pos = i + 1
                        value = str(token.value)
                        while (pos < length and token_list[pos].value == '.'):
                            value = value + '.' + token_list[pos + 1].value
                            pos += 2
                        break
            elif type(node) is javalang.tree.This \
                    or type(node) is javalang.tree.ExplicitConstructorInvocation:
                value = 'this'
            elif type(node) is javalang.tree.BreakStatement:
                value = 'break'
            elif type(node) is javalang.tree.ContinueStatement:
                value = 'continue'
            elif type(node) is javalang.tree.TypeArgument:
                value = str(node.pattern_type)
            elif type(node) is javalang.tree.SuperMethodInvocation \
                    or type(node) is javalang.tree.SuperMemberReference:
                value = 'super.' + str(node.member)
            elif type(node) is javalang.tree.Statement \
                    or type(node) is javalang.tree.BlockStatement \
                    or type(node) is javalang.tree.ForControl \
                    or type(node) is javalang.tree.ArrayInitializer \
                    or type(node) is javalang.tree.SwitchStatementCase:
                value = 'None'
            elif type(node) is javalang.tree.VoidClassReference:
                value = 'void.class'
            elif type(node) is javalang.tree.SuperConstructorInvocation:
                value = 'super'

            if value is not None and type(value) is type('str'):
                d['value'] = value
            if not children and not value:
                self.logger.debug(f'Leaf has no value: {type(node)}, {code}')
                ign = True
            outputs.append(d)
        if not ign and traversal == "sbt":
            sbt = self.get_sbt(outputs, outputs[0])
        elif not ign and traversal == "Preorder":
            sbt = self.get_ast(outputs, outputs[0], traversal)
        else:
            return -1
        return sbt

    def get_sbt(self, ast: List[Dict], root: Dict) -> str:
        """Convert AST to SBT"""
        sbt = []
        t = root['type'].split("(")[0]
        if ('children' not in root.keys()):
            sbt.append("( " + t + " ) " + t + " ")
        else:
            childrens = [n for n in ast if n['id'] in root['children']]
            sbt.append("( " + t + " ")
            for c in childrens:
                sbt += self.get_sbt(ast, c)
            sbt.append(") " + t + " ")
        return "".join(sbt)

    def get_ast(self, ast: List[Dict], root: Dict, traversal: str = "Preorder"):
        """Traverse using Preorder"""
        if traversal == "Preorder":
            ast_pre = []
            t = root['type'].split("(")[0]
            ast_pre.append(t + " ")
            if "children" not in root.keys():
                return "".join(ast_pre)
            else:
                childrens = [n for n in ast if n["id"] in root["children"]]
                for c in childrens:
                    ast_pre += self.get_ast(ast, c, "Preorder").split()
            return "".join(ast_pre)
