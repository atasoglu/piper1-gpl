"""Text encoder pretraining module (phoneme-level BERT)."""

import logging
import math
from typing import Any, List, Optional, Tuple

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from piper.phonemize_multilingual import _WORD_PATTERN

from ..vits.models import TextEncoder
from .dataset import IGNORE_INDEX, PlBertBatch

_LOGGER = logging.getLogger(__name__)


def _head(hidden_channels: int, num_outputs: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_channels, hidden_channels),
        nn.GELU(),
        nn.LayerNorm(hidden_channels),
        nn.Linear(hidden_channels, num_outputs),
    )


class PlBertModel(L.LightningModule):
    """VITS TextEncoder + masked phoneme and phoneme-to-word heads.

    The encoder hyperparameters must be the voice's (the defaults are Piper's
    medium quality); they are checked when the checkpoint is loaded into a
    voice with --model.text_encoder_ckpt.

    word_target picks what every phoneme of a word learns to predict:
    - "vocab": the word's id in a fixed vocabulary (PL-BERT). Cheap, but for an
      agglutinative language like Turkish most word forms are rare, so most
      words fall outside any practical vocabulary.
    - "lm": the word's contextual vector from a frozen pretrained language
      model (lm_name, e.g. dbmdz/bert-base-turkish-cased or xlm-roberta-base),
      the mean of its subword vectors, with a cosine loss. Distills sentence
      context and meaning into the phoneme encoder. The language model only
      runs during pretraining (on the GPU) and is not saved in the checkpoint.
    """

    def __init__(
        self,
        num_symbols: int = 256,
        languages: Optional[List[str]] = None,
        word_vocab_size: int = 30000,
        mask_id: int = 255,
        # encoder (same names and defaults as VitsModel)
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        # training
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        word_loss_weight: float = 1.0,
        # word targets
        word_target: str = "vocab",
        lm_name: str = "xlm-roberta-base",
        lm_layer: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_languages = max(1, len(languages or []))
        self.word_target = word_target
        self.word_loss_weight = word_loss_weight
        self.lm_name = lm_name
        self.lm_layer = lm_layer

        # out_channels only sizes TextEncoder.proj, which pretraining never
        # uses and the voice does not load.
        self.encoder = TextEncoder(
            num_symbols,
            1,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            n_languages=self.num_languages,
        )
        self.mlm_head = _head(hidden_channels, num_symbols)

        # Frozen language model, kept out of the module tree so it is neither
        # trained nor saved. Loaded lazily on the training device.
        self._lm: Optional[Tuple[Any, Any]] = None
        if word_target == "vocab":
            self.word_head = _head(hidden_channels, word_vocab_size)
        elif word_target == "lm":
            from transformers import AutoConfig

            lm_hidden = AutoConfig.from_pretrained(lm_name).hidden_size
            self.word_head = _head(hidden_channels, lm_hidden)
        else:
            raise ValueError(f"word_target must be 'vocab' or 'lm', got {word_target}")

    def forward(self, batch: PlBertBatch):  # pylint: disable=arguments-differ
        lid = batch.language_ids if self.num_languages > 1 else None
        x, _m, _logs, _x_mask = self.encoder(
            batch.phoneme_ids, batch.phoneme_lengths, lid=lid
        )
        x = x.transpose(1, 2)  # [b, t, h]
        return self.mlm_head(x), self.word_head(x)

    def _step(self, batch: PlBertBatch, stage: str) -> torch.Tensor:
        mlm_logits, word_logits = self(batch)

        mlm_loss = F.cross_entropy(
            mlm_logits.float().transpose(1, 2),
            batch.mlm_targets,
            ignore_index=IGNORE_INDEX,
        )
        if self.word_target == "lm":
            word_loss = self._lm_word_loss(batch, word_logits)
        elif bool((batch.word_targets != IGNORE_INDEX).any()):
            word_loss = F.cross_entropy(
                word_logits.float().transpose(1, 2),
                batch.word_targets,
                ignore_index=IGNORE_INDEX,
            )
        else:
            word_loss = mlm_loss.new_zeros(())

        loss = mlm_loss + self.word_loss_weight * word_loss

        with torch.no_grad():
            masked = batch.mlm_targets != IGNORE_INDEX
            mlm_acc = (
                (mlm_logits.argmax(-1)[masked] == batch.mlm_targets[masked])
                .float()
                .mean()
            )

        batch_size = batch.phoneme_ids.size(0)
        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_mlm": mlm_loss,
                f"{stage}_word": word_loss,
                f"{stage}_mlm_acc": mlm_acc,
            },
            batch_size=batch_size,
            prog_bar=(stage == "val"),
            sync_dist=(stage == "val"),
        )
        return loss

    def _load_lm(self):
        if self._lm is None:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.lm_name)
            if not tokenizer.is_fast:
                raise ValueError(
                    f"{self.lm_name} needs a fast tokenizer (for character offsets)"
                )

            lm = AutoModel.from_pretrained(self.lm_name).eval()
            lm.requires_grad_(False)
            self._lm = (tokenizer, lm)

        tokenizer, lm = self._lm
        if next(lm.parameters()).device != self.device:
            lm.to(self.device)

        return tokenizer, lm

    @torch.no_grad()
    def _lm_word_vectors(self, texts: List[str]):
        """Mean subword vector of every word ([b, max_words, d]) and a mask."""
        tokenizer, lm = self._load_lm()
        encoded = tokenizer(
            texts,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        outputs = lm(**encoded.to(self.device), output_hidden_states=True)
        hidden = outputs.hidden_states[self.lm_layer].float()  # [b, s, d]

        word_spans = [
            [(m.start(), m.end()) for m in _WORD_PATTERN.finditer(text)]
            for text in texts
        ]
        max_words = max(1, max(len(spans) for spans in word_spans))
        # pooling[b, w, s] = 1/n for the n subwords of word w
        pooling = torch.zeros(len(texts), max_words, hidden.size(1))
        for i, spans in enumerate(word_spans):
            starts, ends = offsets[i, :, 0], offsets[i, :, 1]
            is_token = ends > starts  # special and padding tokens are (0, 0)
            for w, (start, end) in enumerate(spans):
                in_word = is_token & (starts < end) & (ends > start)
                if bool(in_word.any()):
                    pooling[i, w, in_word] = 1.0 / float(in_word.sum())

        pooling = pooling.to(self.device)
        vectors = torch.bmm(pooling, hidden)  # [b, w, d]
        has_vector = pooling.sum(-1) > 0  # [b, w]
        return vectors, has_vector

    def _lm_word_loss(self, batch: PlBertBatch, word_outputs: torch.Tensor):
        vectors, has_vector = self._lm_word_vectors(batch.texts)

        word_index = batch.word_index.clamp(min=0)
        word_index = word_index.clamp(max=vectors.size(1) - 1)
        targets = torch.gather(
            vectors, 1, word_index.unsqueeze(-1).expand(-1, -1, vectors.size(-1))
        )  # [b, t, d]
        valid = (batch.word_index >= 0) & torch.gather(has_vector, 1, word_index)
        if not bool(valid.any()):
            return word_outputs.new_zeros(()).float()

        similarity = F.cosine_similarity(  # pylint: disable=not-callable
            word_outputs.float()[valid], targets[valid], dim=-1
        )
        return (1.0 - similarity).mean()

    def training_step(  # pylint: disable=arguments-differ
        self, batch: PlBertBatch, batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: PlBertBatch, batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if (param.ndim < 2) or ("emb" in name):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
        )

        # Linear warmup, then cosine decay over the run
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = min(self.hparams.warmup_steps, total_steps // 10 + 1)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps

            progress = min(
                1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)
            )
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
