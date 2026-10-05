"""Tests for multilingual (code-switching) phonemization."""

import pytest

from piper.const import BOS, EOS, PAD
from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP, phonemes_to_ids
from piper.phonemize_multilingual import (
    MultilingualPhonemizer,
    phonemes_to_ids_with_languages,
    split_language_segments,
    strip_language_markup,
)

LANGUAGES = ["tr", "en-us"]
LANGUAGE_ID_MAP = {"tr": 0, "en-us": 1}


def test_split_markup() -> None:
    segments = split_language_segments(
        "Yarın <en-us>Hello world</en-us> diyeceğim.", "tr", LANGUAGES
    )
    assert [(s.text, s.language) for s in segments] == [
        ("Yarın ", "tr"),
        ("Hello world", "en-us"),
        (" diyeceğim.", "tr"),
    ]


def test_split_markup_short_code() -> None:
    # <en> resolves to the voice's en-us
    segments = split_language_segments("<en>hi</en>", "tr", LANGUAGES)
    assert [(s.text, s.language) for s in segments] == [("hi", "en-us")]


def test_split_unknown_tag() -> None:
    with pytest.raises(ValueError):
        split_language_segments("<de>hallo</de>", "tr", LANGUAGES)


def test_split_lexicon_keeps_suffix_in_host_language() -> None:
    segments = split_language_segments(
        "Bugün Meeting'e geç kaldım.", "tr", LANGUAGES, {"meeting": "en-us"}
    )
    assert [(s.text, s.language) for s in segments] == [
        ("Bugün ", "tr"),
        ("Meeting", "en-us"),
        ("'e geç kaldım.", "tr"),
    ]


def test_lexicon_matches_turkish_capital_i() -> None:
    # "İ".lower() is "i" + U+0307, which would never match the lexicon key
    segments = split_language_segments(
        "İnternet yavaş.", "tr", LANGUAGES, {"internet": "en-us"}
    )
    assert [(s.text, s.language) for s in segments] == [
        ("İnternet", "en-us"),
        (" yavaş.", "tr"),
    ]


def test_strip_markup() -> None:
    assert strip_language_markup("a <en-us>b c</en-us> d") == "a b c d"


def test_phonemize_tags_languages() -> None:
    phonemizer = MultilingualPhonemizer(LANGUAGES, lexicon={"meeting": "en-us"})
    sentences = phonemizer.phonemize("Bugün meeting'e geç kaldım.", "tr")

    # The foreign segment must not split the sentence
    assert len(sentences) == 1
    sentence = sentences[0]
    assert len(sentence.phonemes) == len(sentence.languages)

    text = "".join(sentence.phonemes)
    english = "".join(
        p for p, lang in zip(sentence.phonemes, sentence.languages) if lang == "en-us"
    )
    # English reading of "meeting", not Turkish letter-by-letter (meetɪnɡ)
    assert "iː" in english
    assert "meet" not in text
    assert text.endswith(".")


def test_phonemize_real_sentence_breaks() -> None:
    phonemizer = MultilingualPhonemizer(LANGUAGES)
    sentences = phonemizer.phonemize(
        "Merhaba. <en-us>How are you?</en-us> İyiyim.", "tr"
    )
    assert len(sentences) == 3
    assert set(sentences[1].languages) >= {"en-us"}


def test_ids_and_languages_line_up() -> None:
    phonemes = ["b", "a", " ", "ʔ", "x", "."]
    languages = ["tr", "tr", "en-us", "en-us", "en-us", "tr"]
    # "x" is a valid id; make one phoneme missing from the map
    id_map = dict(DEFAULT_PHONEME_ID_MAP)
    id_map.pop("ʔ", None)

    ids, lids = phonemes_to_ids_with_languages(
        phonemes, languages, id_map, LANGUAGE_ID_MAP, sentence_language="tr"
    )

    # Same ids as the monolingual function
    assert ids == phonemes_to_ids(phonemes, id_map)
    assert len(ids) == len(lids)

    # BOS/PAD/EOS, space and punctuation take the sentence language; a phoneme
    # and its PAD take the phoneme's language.
    assert lids[0] == 0  # BOS
    x_idx = ids.index(id_map["x"][0])
    assert lids[x_idx] == 1
    assert lids[x_idx + 1] == 1  # PAD after x
    space_idx = ids.index(id_map[" "][0])
    assert lids[space_idx] == 0
    assert ids[-1] == id_map[EOS][0] and lids[-1] == 0
    assert id_map[BOS][0] == ids[0] and id_map[PAD][0] == ids[1]
