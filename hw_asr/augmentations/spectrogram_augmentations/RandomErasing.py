from torch import Tensor
import random
import torch
from hw_asr.augmentations.base import AugmentationBase


class RandomErase(AugmentationBase):
    def __init__(self, dim=-1, erase_range=(2, 5), fill_value=0, *args, **kwargs):
        self.dim = dim
        self.erase_range = erase_range
        self.fill_value = fill_value

    def __call__(self, data: Tensor):
        erase_len = random.randint(*self.erase_range)
        erase_ind = data.shape[self.dim] - erase_len - 1
        erase_ind = random.randint(0, erase_ind)

        index = torch.arange(erase_ind, erase_ind + erase_len)
        data = data.index_fill_(self.dim, index, self.fill_value)

        return data
