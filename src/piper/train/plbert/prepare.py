"""Phonemize a text corpus for text encoder pretraining.

Input: a text file with one sentence (or paragraph) per line, either
"language|text" or plain text with --default-language. <lang>...</lang> markup
and the lexicon work exactly as they do for a voice.

Output directory:
- meta.json: languages, phoneme id map, mask id and word vocabulary size
- word_vocab.json: word -> id, the most frequent --vocab-size words
- shard_*.pt: lists of samples with, per phoneme id, its language id and the
  index of the word it belongs to (-1 for BOS/PAD/EOS, spaces and punctuation),
  plus the plain text (for --model.word_target lm, which runs a frozen language
  model on it during pretraining)

Phoneme ids are produced by the same phonemizer and phonemes_to_ids_with_languages
(BOS, PAD after every phoneme, EOS) as voice training, so the encoder is
pretrained on exactly the input distribution it will see in a voice.

espeak-ng keeps one global voice per process, so phonemization runs in
separate processes, not threads.
"""

import argparse
import collections
import json
import logging
import unicodedata
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from piper.const import BOS, EOS, PAD
from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP
from piper.phonemize_multilingual import (
    _WORD_PATTERN,
    MultilingualPhonemizer,
    normalize_word,
    phonemes_to_ids_with_languages,
    strip_language_markup,
)

_LOGGER = logging.getLogger("piper.train.plbert.prepare")

DEFAULT_NUM_SYMBOLS = 256
UNKNOWN_WORD = -1

# Set in each worker process
_PHONEMIZER: Optional[MultilingualPhonemizer] = None
_SETTINGS: Dict = {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--corpus", required=True, nargs="+", help="Text file(s)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--languages",
        required=True,
        nargs="+",
        help="espeak-ng voices in language id order (must match the voice's --data.languages)",
    )
    parser.add_argument(
        "--default-language",
        help="Language of lines without a 'language|' prefix",
    )
    parser.add_argument("--lexicon", help="Foreign words: word<TAB>language per line")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--num-symbols", type=int, default=DEFAULT_NUM_SYMBOLS)
    parser.add_argument(
        "--max-phoneme-ids",
        type=int,
        default=512,
        help="Skip samples longer than this (in phoneme ids)",
    )
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-lines", type=int, help="Stop after this many lines (for quick tests)"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    languages: List[str] = args.languages
    if args.default_language and (args.default_language not in languages):
        parser.error(f"--default-language must be one of {languages}")

    lexicon: Dict[str, str] = {}
    if args.lexicon:
        from piper.train.vits.dataset import load_lexicon

        lexicon = load_lexicon(args.lexicon)

    phoneme_id_map = DEFAULT_PHONEME_ID_MAP
    max_phoneme_id = max(max(ids) for ids in phoneme_id_map.values())
    mask_id = args.num_symbols - 1
    if mask_id <= max_phoneme_id:
        parser.error(
            f"No free id for the mask token: --num-symbols ({args.num_symbols}) must be "
            f"greater than max phoneme id + 1 ({max_phoneme_id + 1})"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: word vocabulary from the raw text (fast, no phonemization)
    _LOGGER.info("Counting words")
    word_counts: collections.Counter = collections.Counter()
    for _language, text in _read_corpus(
        args.corpus, args.default_language, args.max_lines
    ):
        word_counts.update(_words(text))

    word_vocab = {
        word: word_id
        for word_id, (word, _count) in enumerate(
            word_counts.most_common(args.vocab_size)
        )
    }
    with open(output_dir / "word_vocab.json", "w", encoding="utf-8") as vocab_file:
        json.dump(word_vocab, vocab_file, ensure_ascii=False)

    _LOGGER.info(
        "Word vocabulary: %s of %s distinct words", len(word_vocab), len(word_counts)
    )

    meta = {
        "languages": languages,
        "lexicon": lexicon,
        "num_symbols": args.num_symbols,
        "mask_id": mask_id,
        "phoneme_id_map": phoneme_id_map,
        "word_vocab_size": len(word_vocab),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False, indent=2)

    # Pass 2: phonemize in worker processes
    settings = {
        "languages": languages,
        "lexicon": lexicon,
        "phoneme_id_map": phoneme_id_map,
        "language_id_map": {lang: i for i, lang in enumerate(languages)},
        "word_vocab": word_vocab,
        "max_phoneme_ids": args.max_phoneme_ids,
    }

    num_samples = 0
    num_aligned = 0
    num_shards = 0
    shard: List[Dict[str, Any]] = []

    def write_shard() -> None:
        nonlocal num_shards, shard
        shard_path = output_dir / f"shard_{num_shards:05d}.pt"
        torch.save(shard, shard_path)
        _LOGGER.info("Wrote %s sample(s) to %s", len(shard), shard_path)
        num_shards += 1
        shard = []

    lines = _read_corpus(args.corpus, args.default_language, args.max_lines)
    with Pool(args.num_workers, initializer=_init_worker, initargs=(settings,)) as pool:
        for sample in pool.imap(_process_line, lines, chunksize=64):
            if sample is None:
                continue

            num_samples += 1
            if bool((sample["word_ids"] >= 0).any()):
                num_aligned += 1

            shard.append(sample)
            if len(shard) >= args.shard_size:
                write_shard()

    if shard:
        write_shard()

    _LOGGER.info(
        "Done: %s sample(s) in %s shard(s); %s with word targets",
        num_samples,
        num_shards,
        num_aligned,
    )


def _read_corpus(
    paths: Iterable[str], default_language: Optional[str], max_lines: Optional[int]
) -> Iterable[Tuple[str, str]]:
    num_lines = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as corpus_file:
            for line in corpus_file:
                line = line.strip()
                if not line:
                    continue

                language, sep, text = line.partition("|")
                if not sep:
                    if default_language is None:
                        raise ValueError(
                            f"Line without 'language|' and no --default-language: {line}"
                        )

                    language, text = default_language, line

                yield language.strip(), text.strip()

                num_lines += 1
                if (max_lines is not None) and (num_lines >= max_lines):
                    return


def _words(text: str) -> List[str]:
    return [
        normalize_word(w) for w in _WORD_PATTERN.findall(strip_language_markup(text))
    ]


def _init_worker(settings: Dict) -> None:
    global _PHONEMIZER, _SETTINGS
    _SETTINGS = settings
    _PHONEMIZER = MultilingualPhonemizer(
        settings["languages"], lexicon=settings["lexicon"]
    )


def _process_line(language_text: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    language, text = language_text
    try:
        return make_sample(text, language, _PHONEMIZER, **_SETTINGS)
    except Exception:  # pylint: disable=broad-exception-caught
        _LOGGER.exception("Failed to process: %s", text)
        return None


def make_sample(
    text: str,
    language: str,
    phonemizer: Optional[MultilingualPhonemizer],
    languages: List[str],
    lexicon: Dict[str, str],
    phoneme_id_map: Dict[str, List[int]],
    language_id_map: Dict[str, int],
    word_vocab: Dict[str, int],
    max_phoneme_ids: int,
) -> Optional[Dict[str, Any]]:
    """Phonemize one line into phoneme ids, language ids and word targets."""
    del languages, lexicon  # used by the phonemizer
    assert phonemizer is not None

    if language not in language_id_map:
        return None

    sentences = phonemizer.phonemize(text, language)
    if not sentences:
        return None

    ids: List[int] = []
    lids: List[int] = []
    word_index: List[int] = []
    num_word_groups = 0

    for sentence in sentences:
        sentence_ids, sentence_lids = phonemes_to_ids_with_languages(
            sentence.phonemes,
            sentence.languages,
            id_map=phoneme_id_map,
            language_id_map=language_id_map,
            sentence_language=language,
        )

        # Word index per phoneme id, built with the same BOS/PAD/EOS layout as
        # phonemes_to_ids_with_languages. Words are runs of phonemes between
        # spaces; punctuation inside a run belongs to no word.
        sentence_word_index: List[int] = []
        sentence_word_index.extend([-1] * len(phoneme_id_map[BOS]))
        sentence_word_index.extend([-1] * len(phoneme_id_map[PAD]))
        in_word = False
        for phoneme in sentence.phonemes:
            is_space = phoneme.isspace()
            if is_space:
                in_word = False
            elif not in_word:
                in_word = True
                num_word_groups += 1

            if phoneme not in phoneme_id_map:
                continue

            is_punctuation = unicodedata.category(phoneme[0]).startswith("P")
            phoneme_word = -1
            if (not is_space) and (not is_punctuation):
                phoneme_word = num_word_groups - 1

            sentence_word_index.extend([phoneme_word] * len(phoneme_id_map[phoneme]))
            sentence_word_index.extend([-1] * len(phoneme_id_map[PAD]))

        sentence_word_index.extend([-1] * len(phoneme_id_map[EOS]))
        assert len(sentence_word_index) == len(sentence_ids)

        ids.extend(sentence_ids)
        lids.extend(sentence_lids)
        word_index.extend(sentence_word_index)

    if len(ids) > max_phoneme_ids:
        return None

    # Word targets only when espeak-ng's word groups line up with the text's
    # words (numbers, abbreviations etc. expand into several groups).
    words = _words(text)
    aligned = len(words) == num_word_groups
    if aligned:
        word_ids = [word_vocab.get(word, UNKNOWN_WORD) for word in words]
    else:
        word_ids = [UNKNOWN_WORD] * num_word_groups

    return {
        "phoneme_ids": torch.tensor(ids, dtype=torch.int16),
        "language_ids": torch.tensor(lids, dtype=torch.int8),
        "word_index": torch.tensor(word_index, dtype=torch.int16),
        "word_ids": torch.tensor(word_ids, dtype=torch.int32),
        "aligned": aligned,
        "text": strip_language_markup(text),
    }


if __name__ == "__main__":
    main()
