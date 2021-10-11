import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(self, *args, **kwargs) -> Tensor:
        log_probs = torch.transpose(kwargs["log_probs"], 0, 1)
        input_lengths = kwargs["log_probs_length"]
        targets = kwargs["text_encoded"]
        target_lengths = kwargs["text_encoded_length"]

        # torch.save(log_probs, "log_probs.pth")
        # torch.save(input_lengths, "input_lengths.pth")
        # torch.save(targets, "targets.pth")
        # torch.save(target_lengths, "target_lengths.pth")
        # raise ValueError()

        return super().forward(log_probs=log_probs, targets=targets,
                               input_lengths=input_lengths, target_lengths=target_lengths)
