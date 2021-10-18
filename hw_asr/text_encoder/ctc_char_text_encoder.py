from typing import List, Tuple, Union

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder

from pyctcdecode import build_ctcdecoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []

        decoder = build_ctcdecoder()

        return sorted(hypos, key=lambda x: x[1], reverse=True)
