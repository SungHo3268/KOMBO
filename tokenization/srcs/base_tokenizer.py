from abc import abstractmethod
from typing import List
import configparser


class BaseTokenizer:
    """Tokenizer meta class"""
    def __init__(self, config_path):
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_path)

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError("Tokenizer::tokenize() is not implemented")       # it must be overridden

    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        raise NotImplementedError("Tokenizer::detokenize() is not implemented")       # it must be overridden
