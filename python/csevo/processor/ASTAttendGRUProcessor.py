from typing import *

import collections
import javalang
from nltk.tokenize import word_tokenize
from pathlib import Path
import pickle
import re
from tqdm import tqdm

from seutil import LoggingUtils

from csevo.data.MethodData import MethodData
from csevo.Environment import Environment
from csevo.processor.AbstractProcessor import AbstractProcessor
from csevo.processor.Tokenizer import Tokenizer


class ASTAttendGRUProcessor(AbstractProcessor):

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self):
        super(ASTAttendGRUProcessor, self).__init__()
        return

    def process_data(self, method_data_list: List[MethodData], data_type: str, output_dir: Path) -> List[int]:
        """Process the data for model ast-attendgru
        Each time this function is called will pickle dump corresponding files.
        e.g. ctrain.pkl, dtrain.pkl, strain.pkl, comstok.pkl"""

        dats_tok = Tokenizer()
        coms_tok = Tokenizer()
        asts_tok = Tokenizer()
        code_list = list()
        nl_list = list()
        ast_list = list()
        for method in tqdm(method_data_list, desc="Processing"):
            # process code
            code = method.code.strip()
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
            nl = " ".join(word_tokenize(method.comment_summary.replace("\n", " "))).lower()
            # process ast
            code = " ".join(tks)
            ast = self.ast_to_sbt(code)
            if ast == -1:
                self.logger.info(f"Error in parsing sbt, discard.")
            else:
                # add data to lists
                code_list.append((method.id, sub_code))
                nl_list.append((method.id, nl))
                ast_list.append((method.id, ast))
        assert len(code_list) == len(ast_list)
        # First build tokenizer
        if data_type == "train":
            dats_tok.train(code_list, 10000)
            coms_tok.train(nl_list, 10000)
            asts_tok.train(ast_list, 10000)
            pickle.dump(dats_tok, open(output_dir / "dats.tok", 'wb'))
            pickle.dump(coms_tok, open(output_dir / "coms.tok", "wb"))
            pickle.dump(asts_tok, open(output_dir / "smls.tok", "wb"))
        # Second build dataset dictionary
        dat_dict = dats_tok.texts_to_sequences(code_list, 100)
        com_dict = coms_tok.texts_to_sequences(nl_list, 13)
        ast_dict = asts_tok.texts_to_sequences(ast_list, 100)
        pickle.dump(dat_dict, open(output_dir / ("d" + data_type + ".pkl"), 'wb'))
        pickle.dump(coms_tok, open(output_dir / ("c" + data_type + ".pkl"), "wb"))
        pickle.dump(asts_tok, open(output_dir / ("s" + data_type + ".pkl"), "wb"))

    def gen_dataset_pkl(output_dir: Path):
        """Run this function to merge data files after finish processing
        train, valid, test set running process_ast_attendgru """

        dats_tok = pickle.load(open(output_dir / "dats.tok", 'rb'), encoding='utf-8')
        coms_tok = pickle.load(open(output_dir / "coms.tok", 'rb'), encoding='utf-8')
        asts_tok = pickle.load(open(output_dir / "smls.tok", 'rb'), encoding='utf-8')
        dataset = dict()
        dataset['dtrain'] = pickle.load(open(output_dir / "dtrain.pkl", 'rb'), encoding='utf-8')
        dataset['dval'] = pickle.load(open(output_dir / "dvalid.pkl", 'rb'), encoding='utf-8')
        dataset['dtest'] = pickle.load(open(output_dir / "dtest.pkl", 'rb'), encoding='utf-8')
        dataset['ctrain'] = pickle.load(open(output_dir / "ctrain.pkl", 'rb'), encoding='utf-8')
        dataset['cval'] = pickle.load(open(output_dir / "cvalid.pkl", 'rb'), encoding='utf-8')
        dataset['ctest'] = pickle.load(open(output_dir / "ctest.pkl", 'rb'), encoding='utf-8')
        dataset['strain'] = pickle.load(open(output_dir / "strain.pkl", 'rb'), encoding='utf-8')
        dataset['sval'] = pickle.load(open(output_dir / "svalid.pkl", 'rb'), encoding='utf-8')
        dataset['stest'] = pickle.load(open(output_dir / "stest.pkl", 'rb'), encoding='utf-8')
        dataset['comstok'] = coms_tok
        dataset['datstok'] = dats_tok
        dataset['smltok'] = asts_tok
        config = dict()
        config['comvocabsize'] = coms_tok.vocab_size
        config['smlvocabsize'] = asts_tok.vocab_size
        config['datvocabsize'] = dats_tok.vocab_size
        config['datlen'] = 100
        config['comlen'] = 13
        config['smllen'] = 100
        dataset['config'] = config
        pickle.dump(dataset, open(output_dir / "dataset.pkl"), "wb")
        return

    def subtokenize_code(self, tokens: List[str]) -> str:
        """Subtokenize the code."""
        subtokens = list()
        for token in tokens:
            curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
            subtokens = subtokens + [c.lower() for c in curr]
        return " ".join(subtokens)

    def ast_to_sbt(self, code: str) -> str:
        """Convert the code to SBT. If it encounters error, return -1."""
        code = code.strip()
        tokens = javalang.tokenizer.tokenize(code)
        token_list = list(javalang.tokenizer.tokenize(code))
        length = len(token_list)
        parser = javalang.parser.Parser(tokens)
        try:
            tree = parser.parse_member_declaration()
        except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
            print(code)
            print("Can not handle this method!")
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
                print('Leaf has no value!')
                print(type(node))
                print(code)
                ign = True
            outputs.append(d)
        if not ign:
            sbt = self.get_sbt(outputs, outputs[0])
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
