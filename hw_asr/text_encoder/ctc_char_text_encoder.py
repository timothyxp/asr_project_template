from typing import List, Tuple, Union

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder
from word_beam_search import WordBeamSearch
import multiprocessing as mp


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str], word_chars=None, corpus=None):
        super().__init__(alphabet)
        self.alphabet = ''.join(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.decoder = build_ctcdecoder(alphabet)
        self.word_chars = word_chars
        self.corpus = corpus

    def ctc_decode(self, inds: Union[List[int], torch.Tensor]) -> str:
        if type(inds) is torch.Tensor:
            inds = inds.detach().cpu().numpy()

        if len(inds) == 0:
            return ''

        ctc_inds = [inds[0]]

        for ind in inds[1:]:
            if ind != ctc_inds[-1]:
                if ind == 0 and self.ind2char[ctc_inds[-1]] == ' ':
                    continue

                ctc_inds.append(ind)

        return ''.join(self.ind2char[ind] for ind in ctc_inds if ind != 0)

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100, n_jobs=1) -> List[str]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        if len(probs.shape) == 2:
            probs = probs.unsqueeze(1)
        else:
            probs = probs.permute(1, 0, 2)

        probs = probs.detach().cpu().numpy()

        assert len(probs.shape) == 3
        char_length, n_samples, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        decoder = WordBeamSearch(
            beam_size,
            "Words",
            0,
            self.corpus.encode("utf-8"),
            self.alphabet.encode("utf-8"),
            self.word_chars.encode("utf-8")
        )

        probs[:, :, [0, -1]] = probs[:, :, [-1, 0]]
        labels_arr = decoder.compute(probs)
        res_str = []

        for label_str in labels_arr:
            res_str.append([])
            s = ''.join([self.alphabet[label] for label in label_str])
            res_str[-1].append(s)

        return [''.join(res) for res in res_str]

