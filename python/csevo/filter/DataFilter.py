import re
from pathlib import Path
import yaml

from seutil import LoggingUtils, IOUtils, BashUtils
from csevo.Environment import Environment


class DataFilter:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)
    SPECIAL_TAGS = ['{', '}', '@code', '@docRoot', '@inheritDoc', '@link', '@linkplain', '@value']

    def __init__(self, config: Path):
        """filter api specified by yml config file"""
        with open(config, 'r') as f:
            self.config = yaml.load(f)

    def data_filter(self, code: str, nl: str) -> (str, str):
        """Method to filter nl and code"""
        if nl == "":
            return code, nl
        if self.config["remove_html"]:
            nl = DataFilter.remove_html_tag(nl)
        if self.config["remove_url"]:
            nl = DataFilter.remove_urls(nl)
        if self.config["remove_non_acsii"]:
            nl = DataFilter.remove_non_acsii(nl)

        if self.config["code_truncate"]:
            code_tokens = code.split()
            if len(code_tokens) > self.config["code"]["max_len"]:
                code_tokens = code_tokens[: self.config["code"]["max_len"]]
            elif len(code_tokens) < self.config["code"]["min_len"]:
                code_tokens = []
            code = " ".join(code_tokens)
        if self.config["nl_truncate"]:
            nl_tokens = nl.split()
            if len(nl_tokens) > self.config["nl"]["max_len"]:
                nl_tokens = nl_tokens[: self.config["nl"]["max_len"]]
            elif len(nl_tokens) < self.config["nl"]["min_len"]:
                nl_tokens = []
            nl = " ".join(nl_tokens)
        return code, nl

    @staticmethod
    def remove_html_tag(line: str) -> str:
        """Helper method for to remove html tag in the natural language comments.
        Cited from Sheena.
        """
        clean_tags = re.compile('<.*?>')
        line = re.sub(clean_tags, '', line)

        for tag in DataFilter.SPECIAL_TAGS:
            line = line.replace(tag, '')
        return line

    @staticmethod
    def remove_urls(line: str) -> str:
        """Helper method to remove urls in the natural language comments."""
        return re.sub(r'http\S+', '', line, flags=re.DOTALL)

    @staticmethod
    def remove_non_acsii(line: str) -> str:
        """Helper method to remove non-ascii characters in the natural language comments"""
        encoded_line = line.encode("ascii", "ignore")
        decode_line = encoded_line.decode()
        return decode_line
