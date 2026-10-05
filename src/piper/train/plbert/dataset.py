"""Data for text encoder pretraining (see prepare.py for the shard format)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from piper.const import BOS, EOS, PAD

_LOGGER = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class PlBertBatch:
    phoneme_ids: torch.Tensor  # [b, t], masked input
    phoneme_lengths: torch.Tensor  # [b]
    language_ids: torch.Tensor  # [b, t]
    mlm_targets: torch.Tensor  # [b, t], IGNORE_INDEX except masked phonemes
    word_targets: torch.Tensor  # [b, t], IGNORE_INDEX where no known word
    word_index: torch.Tensor  # [b, t], word of each phoneme id, -1 if none/unaligned
    texts: List[str]  # plain text of each sample


class PlBertDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class PlBertCollate:
    """Pads a batch and applies whole-word masking (as in PL-BERT/BERT).

    A fraction of words is chosen; their phoneme ids become the mask id (80%),
    a random phoneme id (10%) or stay as they are (10%), and the model must
    predict the original ids. BOS/EOS/PAD, spaces and punctuation are never
    masked. Every phoneme of a known word also predicts that word.
    """

    def __init__(
        self,
        mask_id: int,
        replacement_ids: Sequence[int],
        mask_prob: float = 0.15,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.mask_id = mask_id
        self.replacement_ids = torch.tensor(sorted(replacement_ids), dtype=torch.long)
        self.mask_prob = mask_prob
        self.generator = generator

    def _rand(self, *size: int) -> torch.Tensor:
        return torch.rand(*size, generator=self.generator)

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> PlBertBatch:
        batch_size = len(samples)
        max_len = max(s["phoneme_ids"].size(0) for s in samples)

        phoneme_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        language_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        mlm_targets = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        word_targets = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        batch_word_index = torch.full((batch_size, max_len), -1, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, sample in enumerate(samples):
            ids = sample["phoneme_ids"].long()
            lids = sample["language_ids"].long()
            word_index = sample["word_index"].long()
            word_ids = sample["word_ids"].long()
            length = ids.size(0)
            lengths[i] = length
            language_ids[i, :length] = lids

            in_word = word_index >= 0
            if sample.get("aligned", False):
                batch_word_index[i, :length] = word_index
            if word_ids.numel() > 0:
                # Word target for every phoneme of a known word
                targets = torch.full_like(ids, IGNORE_INDEX)
                targets[in_word] = word_ids[word_index[in_word]]
                targets[targets < 0] = IGNORE_INDEX
                word_targets[i, :length] = targets

                # Whole-word masking
                num_words = word_ids.numel()
                masked_words = self._rand(num_words) < self.mask_prob
                if not bool(masked_words.any()):
                    masked_words[int(self._rand(1).item() * num_words)] = True

                is_masked = torch.zeros_like(in_word)
                is_masked[in_word] = masked_words[word_index[in_word]]
            else:
                is_masked = torch.zeros_like(in_word)

            mlm = torch.full_like(ids, IGNORE_INDEX)
            mlm[is_masked] = ids[is_masked]
            mlm_targets[i, :length] = mlm

            masked_ids = ids.clone()
            action = self._rand(length)
            to_mask = is_masked & (action < 0.8)
            to_random = is_masked & (action >= 0.8) & (action < 0.9)
            masked_ids[to_mask] = self.mask_id
            if bool(to_random.any()):
                random_idx = (
                    self._rand(int(to_random.sum())) * len(self.replacement_ids)
                ).long()
                masked_ids[to_random] = self.replacement_ids[random_idx]

            phoneme_ids[i, :length] = masked_ids

        return PlBertBatch(
            phoneme_ids=phoneme_ids,
            phoneme_lengths=lengths,
            language_ids=language_ids,
            mlm_targets=mlm_targets,
            word_targets=word_targets,
            word_index=batch_word_index,
            texts=[str(sample.get("text", "")) for sample in samples],
        )


class PlBertDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 64,
        validation_split: float = 0.01,
        mask_prob: float = 0.15,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.mask_prob = mask_prob
        self.num_workers = num_workers
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        with open(self.data_dir / "meta.json", "r", encoding="utf-8") as meta_file:
            self.meta = json.load(meta_file)

        # Read by the CLI (linked into the model on instantiation)
        self.languages: List[str] = self.meta["languages"]
        self.num_symbols: int = self.meta["num_symbols"]
        self.mask_id: int = self.meta["mask_id"]
        self.word_vocab_size: int = self.meta["word_vocab_size"]

        special = {PAD, BOS, EOS}
        self.replacement_ids = sorted(
            {
                phoneme_id
                for phoneme, ids in self.meta["phoneme_id_map"].items()
                if phoneme not in special
                for phoneme_id in ids
            }
        )

    def setup(self, stage: str) -> None:
        samples: List[Dict[str, Any]] = []
        for shard_path in sorted(self.data_dir.glob("shard_*.pt")):
            samples.extend(torch.load(shard_path))

        if not samples:
            raise ValueError(f"No samples in {self.data_dir}")

        num_val = max(1, int(len(samples) * self.validation_split))
        self.train_dataset, self.val_dataset = random_split(
            PlBertDataset(samples),
            [len(samples) - num_val, num_val],
            generator=torch.Generator().manual_seed(1234),
        )
        _LOGGER.info(
            "%s training / %s validation sample(s)",
            len(self.train_dataset),
            len(self.val_dataset),
        )

    def _dataloader(self, dataset, shuffle: bool) -> DataLoader:
        assert dataset is not None, "setup() was not called"
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle and (len(dataset) > self.batch_size),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=PlBertCollate(
                mask_id=self.mask_id,
                replacement_ids=self.replacement_ids,
                mask_prob=self.mask_prob,
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)
