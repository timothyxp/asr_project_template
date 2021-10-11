from torch import nn
import torch
from torch.nn import Sequential

from hw_asr.base import BaseModel


class RNNModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=512, num_layers=3, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.bn1 = nn.BatchNorm1d(n_feats)
        self.rnn = nn.LSTM(n_feats, hidden_size, num_layers=1, batch_first=True, bias=False)

        self.out = nn.Linear(in_features=hidden_size, out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        spectrogram = spectrogram * 100000
        #torch.save(spectrogram, "spectrogram1.pth")

        result = self.bn1(spectrogram.permute(0, 2, 1)).permute(0, 2, 1)
      #  torch.save(result, "spectrogram2.pth")

        result, (h_n, c_n) = self.rnn(result)

      #  torch.save(result, "result.pth")

        result = self.out(result)
        return result

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
