import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import numpy as np
import os
from torch import Tensor
import logging

from hw_asr.base.base_text_encoder import BaseTextEncoder

logger = logging.getLogger(__name__)


class CharTextEncoder(BaseTextEncoder):

    def __init__(self, alphabet: List[str], word_chars=None, corpus=None):
        self.ind2char = {k: v for k, v in enumerate(sorted(alphabet))}
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.ind2char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError as e:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return ''.join([self.ind2char[int(ind)] for ind in vector]).strip()

    def dump(self, file):
        with Path(file).open('w') as f:
            json.dump(self.ind2char, f)

    @classmethod
    def from_file(cls, file):
        with Path(file).open() as f:
            ind2char = json.load(f)
        a = cls([])
        a.ind2char = ind2char
        a.char2ind = {v: k for k, v in ind2char}
        return a

    @staticmethod
    def get_corpus_from_path(corpus_path):
        words = set()

        for fold_num in os.listdir(corpus_path):
            for part_num in os.listdir(os.path.join(corpus_path, fold_num)):
                with open(os.path.join(
                        corpus_path,
                        fold_num, part_num,
                        f"{fold_num}-{part_num}.trans.txt")
                ) as f:
                    lines = f.read().split('\n')
                    lines = [line[line.find(' ') + 1:] for line in lines]

                    for line in lines:
                        for word in line.split(' '):
                            words.add(word.lower())

            words = list(words)
            return ' '.join(words)

    @classmethod
    def get_simple_alphabet(cls, corpus_path=None):
        corpus = None
        if corpus_path is not None:
            corpus = cls.get_corpus_from_path(corpus_path)
            logger.info(f"construct corpus {len(corpus)} len")

        return cls(alphabet=list(ascii_lowercase + ' '), word_chars=ascii_lowercase, corpus=corpus)
