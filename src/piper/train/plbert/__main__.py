"""Pretrain a text encoder: python -m piper.train.plbert fit --data.data_dir ..."""

import logging

import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from .dataset import PlBertDataModule
from .model import PlBertModel

_DEFAULT_CALLBACKS = [
    ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="step={step}-val_loss={val_loss:.4f}",
        auto_insert_metric_name=False,
    ),
    LearningRateMonitor(logging_interval="step"),
]


class PlBertLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # The prepared data decides the vocabularies and languages
        for name in ("languages", "num_symbols", "mask_id", "word_vocab_size"):
            parser.link_arguments(
                f"data.{name}", f"model.{name}", apply_on="instantiate"
            )


def main():
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    PlBertLightningCLI(
        PlBertModel,
        PlBertDataModule,
        trainer_defaults={"max_steps": 100000, "callbacks": _DEFAULT_CALLBACKS},
    )


if __name__ == "__main__":
    main()
