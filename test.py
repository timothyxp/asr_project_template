import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from hw_asr.datasets.utils import get_dataloaders
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
import hw_asr.model as module_model
import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
from hw_asr.trainer import Trainer
from hw_asr.utils import prepare_device, set_random_seed, MetricTracker
from hw_asr.utils import ROOT_PATH
import collections
from hw_asr.utils.parse_config import ConfigParser
import torch.nn.functional as F

DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / "default_test_config.json"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file, jobs):
    logger = config.get_logger("test")

    # text_encoder
    corpus_path = ROOT_PATH / 'data' / 'datasets' / 'librispeech' / 'test-clean'
    text_encoder = CTCCharTextEncoder.get_simple_alphabet(corpus_path=str(corpus_path))

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)
    device, device_ids = prepare_device(config["n_gpu"])

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    model = model.to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        for metric_dict in config["metrics"]
    ]

    if config.resume:
        logger.info("Loading checkpoint: {} ...".format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint["state_dict"]
        if config["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    metrics_tracker = MetricTracker(
        "loss", *[m.name for m in metrics]
    )

    results = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloaders["val"])):
            batch = Trainer.move_batch_to_device(batch, device)
            batch["logits"] = model(**batch)
            batch["log_probs"] = F.log_softmax(batch['logits'], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)

            batch['beam_text'] = text_encoder.ctc_beam_search(
                batch["probs"], batch['log_probs_length'], beam_size=100, n_jobs=jobs
            )

            loss = loss_module(**batch)
            metrics_tracker.update("loss", loss.detach().cpu().numpy(), n=len(batch['text']))
            for metric in metrics:
                metrics_tracker.update(metric.name, metric(**batch), n=len(batch['text']))

            for i in tqdm(range(len(batch["text"]))):
                results.append({
                    "ground_trurh": batch["text"][i],
                    "pred_text_argmax": text_encoder.ctc_decode(batch["argmax"][i]),
                    "pred_text_beam_search": batch["beam_text"][i]
                })

    print(metrics_tracker.result())

    with Path(out_file).open('w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_TEST_CONFIG_PATH.absolute().resolve()),
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader"
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")

    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
        CustomArgs(
            ["-o", "--output"], type=str, target="data;val;output"
        )
    ]

    parsed_args = args.parse_args()

    config = ConfigParser.from_args(args, options)

    main(config, config['data']['val']['output'], parsed_args.jobs)
