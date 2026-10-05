"""Multilingual (code-switching) phonemization with espeak-ng.

A multilingual voice is trained on a shared IPA phoneme inventory plus one
language id per phoneme id, so a single sentence can mix languages word by
word ("Bugün <en-us>meeting</en-us>'e geç kaldım.").

espeak-ng cannot be relied on to find the foreign words by itself: its
Turkish voice reads "meeting" letter by letter with Turkish rules (meetɪnɡ)
and never emits a language switch flag for it. So the text is first split into
language segments, and each segment is phonemized with its own espeak-ng
voice:

1. explicit markup: <en-us>text</en-us>, where the tag is one of the voice's
   languages;
2. a lexicon of foreign words (word -> language), applied to the untagged text.
   A Turkish suffix after an apostrophe stays in the surrounding language:
   "meeting'e" -> "meeting" (en-us) + "e" (tr);
3. language switch flags that espeak-ng does emit, e.g. (en)...(tr), are kept
   when they name one of the voice's languages instead of being thrown away.

Sentence boundaries come only from real sentence terminators. espeak-ng ends
every input with an end-of-sentence clause, so phonemizing a lone segment like
"meeting" would otherwise split the host sentence in two.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from .const import BOS, EOS, PAD
from .phonemize_espeak import ESPEAK_DATA_DIR, ESPEAK_LOCK, EspeakPhonemizer

_LANG_TAG_PATTERN = re.compile(r"<([A-Za-z]{2,3}(?:-[A-Za-z0-9]+)*)>(.*?)</\1>", re.S)
_LANG_FLAG_PATTERN = re.compile(r"\(([^)]+)\)")
_WORD_PATTERN = re.compile(r"\w+(?:['’]\w+)?")
_SENTENCE_TERMINATORS = {".", "!", "?"}
_CLAUSE_TERMINATORS = {",", ":", ";"}


@dataclass
class LanguageSegment:
    text: str
    language: str


@dataclass
class MultilingualSentence:
    phonemes: List[str]
    languages: List[str]
    """Language of each phoneme (same length as phonemes)."""


def normalize_word(word: str) -> str:
    """Lexicon key for a word.

    str.lower() turns the Turkish capital İ into i + a combining dot above, so
    a sentence-initial "İnternet" would never match "internet".
    """
    return word.replace("İ", "i").lower()


def resolve_language(code: str, languages: Sequence[str]) -> Optional[str]:
    """Match an espeak-ng language code (en) against a voice language (en-us)."""
    code = code.lower()
    for language in languages:
        if language.lower() == code:
            return language

    for language in languages:
        if language.lower().split("-")[0] == code.split("-")[0]:
            return language

    return None


def split_language_segments(
    text: str,
    default_language: str,
    languages: Sequence[str],
    lexicon: Optional[Mapping[str, str]] = None,
) -> List[LanguageSegment]:
    """Split text into runs of a single language (markup, then lexicon)."""
    segments: List[LanguageSegment] = []

    def add(segment_text: str, language: str) -> None:
        if not segment_text:
            return

        if segments and (segments[-1].language == language):
            segments[-1].text += segment_text
        else:
            segments.append(LanguageSegment(segment_text, language))

    def add_untagged(untagged_text: str) -> None:
        if not lexicon:
            add(untagged_text, default_language)
            return

        last_end = 0
        for match in _WORD_PATTERN.finditer(untagged_text):
            word = match.group(0)
            word_language = lexicon.get(normalize_word(word))
            stem, suffix = word, ""
            if word_language is None:
                # meeting'e -> meeting + 'e
                apostrophe_match = re.search(r"['’]", word)
                if apostrophe_match is not None:
                    stem = word[: apostrophe_match.start()]
                    suffix = word[apostrophe_match.start() :]
                    word_language = lexicon.get(normalize_word(stem))

            if (word_language is None) or (word_language == default_language):
                continue

            add(untagged_text[last_end : match.start()], default_language)
            add(stem, word_language)
            # Suffix after the apostrophe stays in the host language
            add(suffix, default_language)
            last_end = match.end()

        add(untagged_text[last_end:], default_language)

    last_end = 0
    for match in _LANG_TAG_PATTERN.finditer(text):
        add_untagged(text[last_end : match.start()])
        language = resolve_language(match.group(1), languages)
        if language is None:
            raise ValueError(
                f"Unknown language tag <{match.group(1)}>; expected one of {list(languages)}"
            )

        add(match.group(2), language)
        last_end = match.end()

    add_untagged(text[last_end:])

    return segments


def strip_language_markup(text: str) -> str:
    """Remove <lang>...</lang> tags, keeping their text."""
    return _LANG_TAG_PATTERN.sub(lambda m: m.group(2), text)


class MultilingualPhonemizer:
    """Phonemizer that tags every phoneme with its language."""

    def __init__(
        self,
        languages: Sequence[str],
        lexicon: Optional[Mapping[str, str]] = None,
        espeak_data_dir: Union[str, Path] = ESPEAK_DATA_DIR,
    ) -> None:
        if not languages:
            raise ValueError("At least one language is required")

        self.languages = list(languages)
        self.lexicon = {
            normalize_word(word): lang for word, lang in (lexicon or {}).items()
        }
        for word, lang in self.lexicon.items():
            if lang not in self.languages:
                raise ValueError(
                    f"Lexicon word '{word}' has language '{lang}', which is not one of {self.languages}"
                )

        with ESPEAK_LOCK:
            EspeakPhonemizer(espeak_data_dir)

    def phonemize(
        self,
        text: str,
        language: str,
        vowel_clusters: Optional[Set[Tuple[str, ...]]] = None,
    ) -> List[MultilingualSentence]:
        """Text to phonemes (with languages) grouped by sentence."""
        from . import espeakbridge  # avoid circular import

        if language not in self.languages:
            raise ValueError(
                f"Unknown language '{language}': expected one of {self.languages}"
            )

        segments = split_language_segments(text, language, self.languages, self.lexicon)

        sentences: List[MultilingualSentence] = []
        current = MultilingualSentence([], [])

        def append_text(phonemes_str: str, phoneme_language: str) -> None:
            for codepoint in unicodedata.normalize("NFD", phonemes_str):
                current.phonemes.append(codepoint)
                current.languages.append(phoneme_language)

        with ESPEAK_LOCK:
            for segment_idx, segment in enumerate(segments):
                is_last_segment = segment_idx == (len(segments) - 1)
                segment_text = segment.text
                if (segment_idx > 0) and segment_text[:1] in ("'", "’"):
                    # Apostrophe before a suffix ("meeting'e") is not spoken
                    segment_text = segment_text[1:]

                if (
                    current.phonemes
                    and (
                        segment.text[:1].isspace()
                        or segments[segment_idx - 1].text[-1:].isspace()
                    )
                    and (current.phonemes[-1] != " ")
                ):
                    append_text(" ", language)

                espeakbridge.set_voice(segment.language)
                clauses = espeakbridge.get_phonemes(segment_text)
                for clause_idx, (
                    phonemes_str,
                    terminator_str,
                    end_of_sentence,
                ) in enumerate(clauses):
                    is_last_clause = clause_idx == (len(clauses) - 1)

                    # Language switch flags: keep the ones we know
                    clause_language = segment.language
                    for part_idx, part in enumerate(
                        _LANG_FLAG_PATTERN.split(phonemes_str)
                    ):
                        if (part_idx % 2) == 1:
                            clause_language = (
                                resolve_language(part, self.languages)
                                or segment.language
                            )
                            continue

                        append_text(part, clause_language)

                    # Punctuation is not technically a phoneme; it belongs to
                    # the sentence's language.
                    append_text(terminator_str, language)
                    if terminator_str in _CLAUSE_TERMINATORS:
                        append_text(" ", language)

                    # espeak-ng ends every input with an end-of-sentence clause,
                    # even a mid-sentence segment. Only real terminators (or the
                    # very end of the text) end a sentence.
                    if end_of_sentence and (
                        (terminator_str in _SENTENCE_TERMINATORS)
                        or (is_last_segment and is_last_clause)
                    ):
                        sentences.append(current)
                        current = MultilingualSentence([], [])

        if current.phonemes:
            sentences.append(current)

        for sentence in sentences:
            # Trailing spaces from segment joins
            while sentence.phonemes and sentence.phonemes[-1] == " ":
                sentence.phonemes.pop()
                sentence.languages.pop()

            if vowel_clusters:
                sentence.phonemes, sentence.languages = (
                    merge_vowel_clusters_with_languages(
                        sentence.phonemes, sentence.languages, vowel_clusters
                    )
                )

        return [sentence for sentence in sentences if sentence.phonemes]


def merge_vowel_clusters_with_languages(
    phonemes: Sequence[str],
    languages: Sequence[str],
    clusters: Set[Tuple[str, ...]],
) -> Tuple[List[str], List[str]]:
    """Merge adjacent recognized vowel clusters, keeping languages aligned."""
    max_len = max(len(k) for k in clusters)
    out_phonemes: List[str] = []
    out_languages: List[str] = []
    i = 0

    while i < len(phonemes):
        match_len = 1
        for n in range(min(max_len, len(phonemes) - i), 1, -1):
            if tuple(phonemes[i : i + n]) in clusters:
                match_len = n
                break

        out_phonemes.append("".join(phonemes[i : i + match_len]))
        out_languages.append(languages[i])
        i += match_len

    return out_phonemes, out_languages


def phonemes_to_ids_with_languages(
    phonemes: Sequence[str],
    languages: Sequence[str],
    id_map: Mapping[str, Sequence[int]],
    language_id_map: Mapping[str, int],
    sentence_language: str,
) -> Tuple[List[int], List[int]]:
    """Phonemes to ids, with one language id per phoneme id.

    Mirrors phoneme_ids.phonemes_to_ids (BOS, PAD after every phoneme, EOS).
    PAD takes the language of the phoneme before it; BOS, EOS, spaces and
    punctuation take the sentence's language.
    """
    if len(phonemes) != len(languages):
        raise ValueError("Phonemes and languages must have the same length")

    sentence_lid = language_id_map[sentence_language]
    ids: List[int] = []
    lids: List[int] = []

    def extend(new_ids: Sequence[int], lid: int) -> None:
        ids.extend(new_ids)
        lids.extend([lid] * len(new_ids))

    extend(id_map[BOS], sentence_lid)
    extend(id_map[PAD], sentence_lid)

    for phoneme, phoneme_language in zip(phonemes, languages):
        if phoneme not in id_map:
            # Same as phonemes_to_ids: skip (it logs the warning)
            continue

        if phoneme.isspace() or unicodedata.category(phoneme[0]).startswith("P"):
            lid = sentence_lid
        else:
            lid = language_id_map[phoneme_language]

        extend(id_map[phoneme], lid)
        extend(id_map[PAD], lid)

    extend(id_map[EOS], sentence_lid)

    return ids, lids
