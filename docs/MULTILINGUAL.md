# 🌍 Multilingual voices and text encoder pretraining

Two independent, opt-in training features. Neither adds inference cost beyond
a few embedding rows: a voice trained with them has the same architecture,
parameter count (± one `num_languages × hidden_channels` table) and speed.

A Colab notebook that runs the whole pipeline (corpus → pretraining → voice →
ONNX) is in [`notebooks/colab_multilingual_plbert.ipynb`](../notebooks/colab_multilingual_plbert.ipynb).

## Multilingual voices (code-switching)

Every phoneme id carries a language id, which is embedded and added to the
phoneme embedding in the text encoder. A sentence can therefore switch
language word by word:

``` text
Bugün <en-us>meeting</en-us>'e geç kaldım.
```

espeak-ng's own language detection is not enough for this: its Turkish voice
reads `meeting` with Turkish letter rules (`meetɪnɡ`) and never flags it as
English. Text is split into language segments first, and each segment is
phonemized with its own espeak-ng voice:

1. `<lang>…</lang>` markup, where `lang` is one of the voice's languages
   (`<en>` resolves to `en-us`);
2. a **lexicon** of foreign words (`word<TAB>language` per line). A suffix after
   an apostrophe stays in the host language: `meeting'e` → `meeting` (en-us) +
   `e` (tr);
3. language switch flags that espeak-ng does emit, when they name one of the
   voice's languages.

### Training

The CSV gets a language column right before the text:

``` csv
utt1|tr|Bugün meeting'e geç kaldım.
utt2|en-us|I'm running late for the meeting.
```

(`utt|speaker|language|text` for multi-speaker voices.)

``` sh
python3 -m piper.train fit \
  --data.espeak_voice tr \
  --data.languages '[tr, en-us]' \
  --data.lexicon_path /path/to/lexicon.tsv \
  ...
```

* `--data.languages` lists espeak-ng voices in **language id order**; keep the
  order fixed for the lifetime of the voice. `--data.espeak_voice` must be one
  of them and is the voice's default language.
* Only espeak-ng phonemes and text datasets are supported.
* The language list and lexicon are written to the voice config
  (`language_id_map`, `lexicon`) and used at synthesis time.
* Fine-tuning an existing monolingual checkpoint: use
  `--model.warmstart_ckpt` (non-strict), not `--ckpt_path`, which fails on the
  new language embedding.

The data the voice is trained on decides how well it speaks each language: a
speaker who only ever speaks Turkish in the dataset will still have a Turkish
accent in English words.

### Synthesis

``` python
from piper import PiperVoice
from piper.config import SynthesisConfig

voice = PiperVoice.load("voice.onnx")
voice.synthesize("Bugün meeting'e geç kaldım.")                 # default language
voice.synthesize("Hello there.", SynthesisConfig(language="en-us"))  # whole text
```

The exported model has an extra `lid` input (one language id per phoneme id).
libpiper runs multilingual voices but does not split text by language: every
phoneme gets the voice's default language. `piper.train.infer_torch` does not
support multilingual voices.

## Text encoder pretraining (phoneme-level BERT)

Piper's text encoder sees only phonemes, so it has little sense of sentence
context, which is a large part of why prosody sounds read rather than spoken.
`piper.train.plbert` pretrains the voice's own text encoder (`enc_p`) on a
large phonemized text corpus, without audio, as in PL-BERT
([Li et al. 2023](https://arxiv.org/abs/2301.08810)):

* masked phoneme prediction with whole-word masking, plus
* a per-phoneme word target, either
  * `--model.word_target vocab`: the word's id in a vocabulary (PL-BERT), or
  * `--model.word_target lm`: the word's contextual vector from a frozen
    pretrained language model (`--model.lm_name`, e.g. `xlm-roberta-base` or
    `dbmdz/bert-base-turkish-cased`), mean of its subwords, cosine loss.
    Requires `pip install -e '.[train,plbert]'`. For agglutinative languages
    most word forms fall outside any practical vocabulary, so this target is
    expected to do better there; this has not been measured yet, so compare
    both. The language model runs only during pretraining and is not saved in
    the checkpoint. Watch that `val_word` stays above 0: it is 0 when no word
    could be matched to the language model's tokens.

The prediction heads are discarded; only the encoder is loaded into the voice.

``` sh
# 1. Phonemize a corpus ("language|text" per line), in parallel processes
python3 -m piper.train.plbert.prepare \
  --corpus corpus.txt \
  --output-dir plbert_data \
  --languages tr en-us \
  --lexicon lexicon.tsv \
  --num-workers 8

# 2. Pretrain (encoder size defaults match Piper's medium voices)
python3 -m piper.train.plbert fit \
  --data.data_dir plbert_data \
  --model.word_target lm \
  --model.lm_name xlm-roberta-base \
  --trainer.max_steps 100000

# 3. Train a voice starting from the pretrained encoder
python3 -m piper.train fit ... \
  --data.languages '[tr, en-us]' \
  --model.text_encoder_ckpt /path/to/plbert/last.ckpt
```

Loading checks that the encoder size (`hidden_channels`, `filter_channels`,
`n_heads`, `n_layers`, `kernel_size`), `num_symbols` and the language list
match the voice, and refuses the checkpoint otherwise. Phoneme ids come from
the default id map with the same BOS/PAD/EOS layout as voice training; the
mask token uses the last id (`num_symbols - 1`), which no phoneme uses. Like the
other warmstarts, the encoder is not reloaded when resuming with `--ckpt_path`.

Pretraining is not a guaranteed improvement. Compare against a voice trained
without it on the same data (the notebook sets up this A/B).
