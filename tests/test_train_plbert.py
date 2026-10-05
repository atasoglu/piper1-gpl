"""Text encoder pretraining: prepare -> masking -> train -> load into a voice."""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

pytest.importorskip("piper.train.vits.lightning")

# pylint: disable=wrong-import-position,protected-access
import lightning as L  # noqa: E402

from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP  # noqa: E402
from piper.phonemize_multilingual import MultilingualPhonemizer  # noqa: E402
from piper.train.plbert.dataset import (  # noqa: E402
    IGNORE_INDEX,
    PlBertCollate,
    PlBertDataModule,
)
from piper.train.plbert.model import PlBertModel  # noqa: E402
from piper.train.plbert.prepare import make_sample  # noqa: E402
from piper.train.vits.lightning import VitsModel  # noqa: E402

LANGUAGES = ["tr", "en-us"]
TINY_ENCODER: Dict[str, Any] = {
    "hidden_channels": 16,
    "filter_channels": 32,
    "n_heads": 2,
    "n_layers": 2,
}


def _sample(text: str, language: str = "tr"):
    return make_sample(
        text,
        language,
        MultilingualPhonemizer(LANGUAGES, lexicon={"meeting": "en-us"}),
        languages=LANGUAGES,
        lexicon={},
        phoneme_id_map=DEFAULT_PHONEME_ID_MAP,
        language_id_map={"tr": 0, "en-us": 1},
        word_vocab={"bugün": 0, "meeting'e": 1, "geç": 2},
        max_phoneme_ids=512,
    )


def test_make_sample_aligns_words() -> None:
    sample = _sample("Bugün meeting'e geç kaldım.")
    assert sample is not None
    n = sample["phoneme_ids"].numel()
    assert sample["language_ids"].numel() == n
    assert sample["word_index"].numel() == n

    # 4 words; "kaldım" is not in the vocabulary
    assert sample["word_ids"].tolist() == [0, 1, 2, -1]
    word_index = sample["word_index"].tolist()
    assert sorted(set(word_index)) == [-1, 0, 1, 2, 3]

    # The English phonemes of "meeting" carry language id 1 and word 1
    lids = sample["language_ids"].tolist()
    assert any(lid == 1 and w == 1 for lid, w in zip(lids, word_index))


def test_collate_masks_whole_words_only() -> None:
    sample = _sample("Bugün meeting'e geç kaldım.")
    collate = PlBertCollate(
        mask_id=255,
        replacement_ids=[14, 15],
        mask_prob=0.5,
        generator=torch.Generator().manual_seed(0),
    )
    batch = collate([sample, sample])

    word_index = sample["word_index"].long()
    n = word_index.numel()
    for row in range(2):
        masked = batch.mlm_targets[row, :n] != IGNORE_INDEX
        assert masked.any()
        # Only phonemes inside words, never BOS/PAD/EOS/space/punctuation
        assert bool((word_index[masked] >= 0).all())
        # Whole words: every phoneme of a masked word is masked
        for word in word_index[masked].unique():
            assert bool(masked[word_index == word].all())

        # Targets are the original ids
        original = sample["phoneme_ids"].long()
        assert torch.equal(batch.mlm_targets[row, :n][masked], original[masked])

    # Word targets: phonemes of known words only
    assert (batch.word_targets[0, :n] == 1).any()


def _prepare(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    lines = [
        "tr|Bugün meeting'e geç kaldım.",
        "tr|Yarın hava çok güzel olacak.",
        "en-us|The weather will be nice tomorrow.",
        "en-us|I am late for the meeting.",
    ] * 4
    corpus.write_text("\n".join(lines), encoding="utf-8")
    lexicon = tmp_path / "lexicon.tsv"
    lexicon.write_text("meeting\ten-us\n", encoding="utf-8")

    data_dir = tmp_path / "plbert_data"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "piper.train.plbert.prepare",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(data_dir),
            "--languages",
            *LANGUAGES,
            "--lexicon",
            str(lexicon),
            "--num-workers",
            "2",
            "--shard-size",
            "10",
        ]
    )
    return data_dir


def test_pretrain_and_load_into_voice(tmp_path: Path) -> None:
    data_dir = _prepare(tmp_path)
    assert len(list(data_dir.glob("shard_*.pt"))) == 2

    data = PlBertDataModule(
        data_dir, batch_size=4, validation_split=0.25, num_workers=0
    )
    model = PlBertModel(
        num_symbols=data.num_symbols,
        languages=data.languages,
        word_vocab_size=data.word_vocab_size,
        mask_id=data.mask_id,
        warmup_steps=1,
        **TINY_ENCODER,
    )
    trainer = L.Trainer(
        max_steps=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=data)
    ckpt_path = tmp_path / "plbert.ckpt"
    trainer.save_checkpoint(ckpt_path)

    voice = VitsModel(
        languages=LANGUAGES,
        inter_channels=16,
        upsample_initial_channel=16,
        mos_metric=None,
        text_encoder_ckpt=str(ckpt_path),
        **TINY_ENCODER,
    )
    voice._load_text_encoder_from_ckpt(str(ckpt_path))
    for name, param in voice.model_g.enc_p.named_parameters():
        if name.startswith("proj."):
            continue

        assert torch.equal(param, model.encoder.state_dict()[name]), name

    # Mismatched encoder size or languages are refused
    with pytest.raises(ValueError):
        VitsModel(
            languages=LANGUAGES,
            inter_channels=16,
            hidden_channels=16,
            filter_channels=32,
            n_layers=3,
            upsample_initial_channel=16,
            mos_metric=None,
        )._load_text_encoder_from_ckpt(str(ckpt_path))

    with pytest.raises(ValueError):
        VitsModel(
            languages=["en-us", "tr"],
            inter_channels=16,
            upsample_initial_channel=16,
            mos_metric=None,
            **TINY_ENCODER,
        )._load_text_encoder_from_ckpt(str(ckpt_path))


TINY_LM = "hf-internal-testing/tiny-random-BertModel"


def _tiny_lm_available() -> bool:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(TINY_LM)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


@pytest.mark.skipif(
    not _tiny_lm_available(), reason="needs transformers and the tiny test LM"
)
def test_lm_word_target(tmp_path: Path) -> None:
    data_dir = _prepare(tmp_path)
    data = PlBertDataModule(
        data_dir, batch_size=4, validation_split=0.25, num_workers=0
    )
    model = PlBertModel(
        num_symbols=data.num_symbols,
        languages=data.languages,
        word_vocab_size=data.word_vocab_size,
        mask_id=data.mask_id,
        warmup_steps=1,
        word_target="lm",
        lm_name=TINY_LM,
        **TINY_ENCODER,
    )

    # Word vectors are the mean of each word's subword vectors
    texts = ["Bugün meeting'e geç kaldım.", "Hello there."]
    vectors, has_vector = model._lm_word_vectors(texts)
    assert vectors.shape[:2] == (2, 4)
    assert (
        has_vector[0].all() and has_vector[1, :2].all() and not has_vector[1, 2:].any()
    )

    tokenizer, lm = model._load_lm()
    encoded = tokenizer(
        ["Hello there."], return_offsets_mapping=True, return_tensors="pt"
    )
    offsets = encoded.pop("offset_mapping")[0]
    hidden = lm(**encoded).last_hidden_state[0]
    hello = [i for i, (s, e) in enumerate(offsets.tolist()) if e > s and s < 5]
    assert torch.allclose(vectors[1, 0], hidden[hello].mean(0), atol=1e-5)

    trainer = L.Trainer(
        max_steps=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=data)
    assert torch.isfinite(trainer.callback_metrics["val_word"])
    assert float(trainer.callback_metrics["val_word"]) > 0

    # The frozen LM is not part of the checkpoint
    ckpt_path = tmp_path / "plbert_lm.ckpt"
    trainer.save_checkpoint(ckpt_path)
    state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
    assert all(
        k.split(".")[0] in {"encoder", "mlm_head", "word_head"} for k in state_dict
    )
