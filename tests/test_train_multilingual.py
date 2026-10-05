"""End-to-end plumbing for multilingual voices: data -> train step -> ONNX -> PiperVoice.

These use a tiny random model on noise, so they prove the language ids flow
through every stage and line up with the phoneme ids, not that it sounds good.
"""

import csv
import json
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

pytest.importorskip("piper.train.vits.lightning")
pytest.importorskip("onnx")

# pylint: disable=wrong-import-position,protected-access
import lightning as L  # noqa: E402
import torch  # noqa: E402

from piper import PiperVoice  # noqa: E402
from piper.config import SynthesisConfig  # noqa: E402
from piper.train.vits.dataset import VitsDataModule  # noqa: E402
from piper.train.vits.lightning import VitsModel  # noqa: E402

SAMPLE_RATE = 16000
TINY: Dict[str, Any] = {
    "inter_channels": 16,
    "hidden_channels": 16,
    "filter_channels": 32,
    "n_heads": 2,
    "n_layers": 2,
    "upsample_initial_channel": 16,
    "mos_metric": None,
}


def _write_dataset(tmp_path: Path, rows: list[list[str]]) -> Path:
    audio_dir = tmp_path / "wav"
    audio_dir.mkdir()
    rng = np.random.default_rng(0)
    for row in rows:
        samples = (rng.standard_normal(SAMPLE_RATE) * 3000).astype(np.int16)
        with wave.Wave_write(str(audio_dir / f"{row[0]}.wav")) as wav_file:
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(samples.tobytes())

    csv_path = audio_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file, delimiter="|").writerows(rows)

    return csv_path


def _train_and_export(
    tmp_path: Path,
    languages,
    rows,
    lexicon=None,
    num_speakers: int = 1,
    warmstart_ckpt: Optional[Path] = None,
) -> Path:
    csv_path = _write_dataset(tmp_path, rows)
    lexicon_path = None
    if lexicon:
        lexicon_path = tmp_path / "lexicon.tsv"
        lexicon_path.write_text(
            "".join(f"{w}\t{lang}\n" for w, lang in lexicon.items()), encoding="utf-8"
        )

    config_path = tmp_path / "voice.onnx.json"
    data = VitsDataModule(
        csv_path=csv_path,
        cache_dir=tmp_path / "cache",
        espeak_voice="tr",
        config_path=config_path,
        voice_name="test",
        sample_rate=SAMPLE_RATE,
        batch_size=2,
        validation_split=0.0,
        num_test_examples=0,
        num_workers=0,
        trim_silence=False,
        languages=languages,
        lexicon_path=lexicon_path,
        num_speakers=num_speakers,
    )
    model = VitsModel(
        sample_rate=SAMPLE_RATE,
        languages=languages,
        batch_size=2,
        num_speakers=num_speakers,
        warmstart_ckpt=str(warmstart_ckpt) if warmstart_ckpt else None,
        **TINY,
    )
    trainer = L.Trainer(
        max_steps=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=data)

    ckpt_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt_path)

    onnx_path = tmp_path / "voice.onnx"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "piper.train.export_onnx",
            "--checkpoint",
            str(ckpt_path),
            "--output-file",
            str(onnx_path),
        ]
    )
    return onnx_path


def test_multilingual_train_export_synthesize(tmp_path: Path) -> None:
    languages = ["tr", "en-us"]
    rows = [
        ["utt1", "tr", "Bugün meeting'e geç kaldım."],
        ["utt2", "en-us", "Hello, how are you?"],
        ["utt3", "tr", "Yarın <en-us>deadline</en-us> var."],
        ["utt4", "en-us", "I love Istanbul."],
    ]
    onnx_path = _train_and_export(
        tmp_path, languages, rows, lexicon={"meeting": "en-us"}
    )

    config = json.loads((tmp_path / "voice.onnx.json").read_text(encoding="utf-8"))
    assert config["language_id_map"] == {"tr": 0, "en-us": 1}
    assert config["num_languages"] == 2
    assert config["lexicon"] == {"meeting": "en-us"}

    # Cached language ids line up with phoneme ids
    cache_dir = tmp_path / "cache"
    for langs_path in cache_dir.glob("*.langs.pt"):
        ids_path = Path(str(langs_path).replace(".langs.pt", ".phonemes.pt"))
        assert torch.load(langs_path).shape == torch.load(ids_path).shape

    voice = PiperVoice.load(onnx_path)
    assert [i.name for i in voice.session.get_inputs()] == [
        "input",
        "input_lengths",
        "scales",
        "lid",
    ]

    chunks = list(voice.synthesize("Bugün meeting'e geç kaldım."))
    assert len(chunks) == 1
    assert chunks[0].audio_float_array.size > 0

    # Whole text in the other language
    chunks = list(
        voice.synthesize("Hello there.", syn_config=SynthesisConfig(language="en-us"))
    )
    assert chunks and chunks[0].audio_float_array.size > 0


def test_monolingual_export_has_no_language_input(tmp_path: Path) -> None:
    rows = [["utt1", "Merhaba dünya."], ["utt2", "Nasılsın?"]]
    onnx_path = _train_and_export(tmp_path, None, rows)

    config = json.loads((tmp_path / "voice.onnx.json").read_text(encoding="utf-8"))
    assert "language_id_map" not in config

    voice = PiperVoice.load(onnx_path)
    assert [i.name for i in voice.session.get_inputs()] == [
        "input",
        "input_lengths",
        "scales",
    ]
    assert list(voice.synthesize("Merhaba."))


def test_multilingual_requires_known_row_language(tmp_path: Path) -> None:
    rows = [["utt1", "de", "Hallo."], ["utt2", "tr", "Merhaba."]]
    with pytest.raises(ValueError):
        _train_and_export(tmp_path, ["tr", "en-us"], rows)


def test_multispeaker_multilingual_warmstart_from_monolingual(tmp_path: Path) -> None:
    # The overfit recipe: a single-speaker monolingual voice warmstarts a
    # multi-speaker multilingual one (new speaker/language layers start fresh).
    mono_dir = tmp_path / "mono"
    mono_dir.mkdir()
    _train_and_export(mono_dir, None, [["a", "Merhaba dünya."], ["b", "Nasılsın?"]])

    multi_dir = tmp_path / "multi"
    multi_dir.mkdir()
    rows = [
        ["utt1", "sila", "tr", "Bugün meeting'e geç kaldım."],
        ["utt2", "lj", "en-us", "Hello, how are you?"],
        ["utt3", "sila", "tr", "Yarın toplantı var."],
        ["utt4", "lj", "en-us", "I love Istanbul."],
    ]
    onnx_path = _train_and_export(
        multi_dir,
        ["tr", "en-us"],
        rows,
        lexicon={"meeting": "en-us"},
        num_speakers=2,
        warmstart_ckpt=mono_dir / "model.ckpt",
    )

    config = json.loads((multi_dir / "voice.onnx.json").read_text(encoding="utf-8"))
    assert config["num_speakers"] == 2
    assert config["language_id_map"] == {"tr": 0, "en-us": 1}

    voice = PiperVoice.load(onnx_path)
    assert [i.name for i in voice.session.get_inputs()] == [
        "input",
        "input_lengths",
        "scales",
        "sid",
        "lid",
    ]

    # Every speaker in every language, including the pairing never seen in data
    sila = config["speaker_id_map"]["sila"]
    for language in ("tr", "en-us"):
        chunks = list(
            voice.synthesize(
                "Hello there.",
                syn_config=SynthesisConfig(speaker_id=sila, language=language),
            )
        )
        assert chunks and chunks[0].audio_float_array.size > 0
