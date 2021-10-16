from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from hw_asr.base import BaseModel


class RNNModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=512, num_layers=3, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.rnn = nn.GRU(n_feats, hidden_size, num_layers=num_layers, batch_first=True, bias=False)

        self.out = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=n_class)
        )

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        result = spectrogram * 100000

        # result = self.bn1(spectrogram.permute(0, 2, 1)).permute(0, 2, 1)

        result, _ = self.rnn(result)

        result = self.out(result)
        return result

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
