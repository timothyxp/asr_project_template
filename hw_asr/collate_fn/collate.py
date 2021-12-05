import logging
from typing import List, Dict, Union
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch: Dict[str, Union[list, torch.Tensor]] = defaultdict(list)

    for dataset_item in dataset_items:
        result_batch['text_encoded_length'].append(dataset_item['text_encoded'].shape[1])
        result_batch['spectrogram_length'].append(dataset_item['spectrogram'].shape[2])

        for key, item in dataset_item.items():
            if key in ['spectrogram', 'text_encoded']:
                item = item.squeeze(0)
                item = item.T

            result_batch[key].append(item)

    result_batch['text_encoded_length'] = torch.LongTensor(result_batch['text_encoded_length'])
    result_batch['spectrogram_length'] = torch.LongTensor(result_batch['spectrogram_length'])

    result_batch['spectrogram'] = pad_sequence(result_batch['spectrogram'], batch_first=True)
    result_batch['text_encoded'] = pad_sequence(result_batch['text_encoded']).T

    return result_batch

