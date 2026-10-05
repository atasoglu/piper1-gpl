"""Generates colab_multilingual_plbert.ipynb. Edit this file, not the notebook."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(True),
        }
    )


md(r"""
# Piper: çok dilli (code-switching) ses + ön eğitimli text encoder

Bu notebook iki şeyi birlikte eğitir:

1. **Çok dilli ses:** her fonem kendi dil ID'sini taşır, böylece tek cümlede dil değiştirilebilir:
   `Bugün <en-us>meeting</en-us>'e geç kaldım.` Yabancı kelimeler `<dil>…</dil>` etiketiyle ya da bir **lexicon** dosyasıyla bulunur.
2. **Ön eğitimli text encoder:** VITS'in `enc_p` encoder'ı, ses verisi olmadan büyük bir fonem korpusu üzerinde ön eğitilir. Ses modeli bu encoder'dan başlar.
   * `word_target=vocab`: PL-BERT (maskeli fonem + kelime ID'si tahmini).
   * `word_target=lm`: maskeli fonem + **dondurulmuş bir dil modelinin** (XLM-R / BERTurk) kelime vektörlerini tahmin etme. Türkçe gibi eklemeli dillerde kelime sözlüğü seyrek kaldığı için daha iyi olması beklenir, ancak bu henüz ölçülmedi: A/B ile doğrulayın.

Inference'ta **ek parametre ve ek gecikme yoktur.** Ön eğitimdeki tahmin başlıkları ve dil modeli export'a girmez; dil embedding'i dil sayısı × 192 parametredir.

**Akış:** kurulum → korpus → fonemleştirme → encoder ön eğitimi → ses eğitimi (A/B: encoder'lı / encoder'sız) → ONNX export → dinleme.

> **Hücreleri sırayla çalıştırın.** Overfit testi için 1. Ayarlar, 2. Kurulum ve Overfit testi bölümleri yeterlidir; 3–8. adımlar kendi ses verinizle tam eğitim içindir.
> Tüm dosyalar Colab'ın geçici diskine (`/content`) yazılır. Oturum kapanınca silinir: saklamak istediğiniz checkpoint ve `.onnx` dosyalarını sol paneldeki dosya tarayıcısından indirin. Aynı oturumda bir eğitim hücresini yeniden çalıştırırsanız kaldığı yerden devam eder.
""")

code(r"""
import subprocess
if subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
    raise RuntimeError("GPU bulunamadı. Runtime → Change runtime type → T4 GPU seçip bu hücreyi yeniden çalıştırın.")
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("## 1. Ayarlar")

code(r"""
import os
from pathlib import Path

# --- Değiştirin ---------------------------------------------------------------
REPO_URL = "https://github.com/atasoglu/piper1-gpl.git"
BRANCH = "multilingual-plbert"

WORK = Path("/content/piper_multilingual")  # checkpoint'ler, hazır veri
LOCAL = Path("/content/work")                # cache

LANGUAGES = ["tr", "en-us"]   # espeak-ng ses adları; SIRASI dil ID'lerini belirler, sonradan değiştirmeyin
PRIMARY_LANGUAGE = "tr"       # sesin varsayılan dili

# Kendi ses veriniz (3–8. adımlar için; Colab'a yükleyin): wav dosyaları + metadata.csv
#   tek konuşmacı:  dosya_adı|dil|metin      örn. 0001|tr|Bugün meeting'e geç kaldım.
#   çok konuşmacı:  dosya_adı|konuşmacı|dil|metin
TTS_AUDIO_DIR = WORK / "dataset" / "wav"
TTS_CSV = TTS_AUDIO_DIR / "metadata.csv"
SAMPLE_RATE = 22050
# ------------------------------------------------------------------------------

for d in (WORK, LOCAL):
    d.mkdir(parents=True, exist_ok=True)
""")

md("## 2. Kurulum\n\nColab'ın kendi PyTorch'u kullanılır, yeniden kurulmaz. espeak-ng derlemesi birkaç dakika sürer.")

code(r"""
%cd /content
if not Path("/content/piper1-gpl").exists():
    !git clone --branch {BRANCH} {REPO_URL} piper1-gpl
%cd /content/piper1-gpl
!git pull --ff-only

import sys
!pip install -q uv
!uv pip install --python {sys.executable} -q scikit-build cmake ninja cython
!uv pip install --python {sys.executable} -q -e ".[train,plbert]" datasets
!{sys.executable} setup.py build_ext --inplace > /content/build_ext.log 2>&1 && echo "espeakbridge OK"
!bash build_monotonic_align.sh && echo "monotonic_align OK"

# Düzenlenebilir kurulum, çalışan oturumda ancak yeniden başlatınca görünür.
# Yeniden başlatmaya gerek kalmasın diye kaynak klasörü doğrudan yola eklenir
# (PYTHONPATH, "!python -m ..." komutlarının da aynı kodu görmesini sağlar).
SRC = "/content/piper1-gpl/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")
import piper
print("piper:", piper.__file__)
""")

code(r"""
import torch
from piper.phonemize_multilingual import MultilingualPhonemizer

print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
# is_bf16_supported() T4'te de True döner (yazılımla taklit, çok yavaş).
# Gerçek bf16 desteği Ampere (compute capability 8.0) ve sonrasında var.
BF16 = torch.cuda.get_device_capability(0)[0] >= 8
# VITS (GAN) eğitimi fp16'da kararsız olabilir: bf16 yoksa (T4) fp32 kullan.
TTS_PRECISION = "bf16-mixed" if BF16 else "32-true"
# Encoder ön eğitimi fp16 ile sorunsuz çalışır.
PLBERT_PRECISION = "bf16-mixed" if BF16 else "16-mixed"
print("TTS precision:", TTS_PRECISION, "| PL-BERT precision:", PLBERT_PRECISION)

# Hızlı kontrol: code-switching fonemleştirmesi
p = MultilingualPhonemizer(LANGUAGES, lexicon={"meeting": "en-us"})
for s in p.phonemize("Bugün meeting'e geç kaldım.", PRIMARY_LANGUAGE):
    print("".join(s.phonemes))
    print("".join("E" if lang != PRIMARY_LANGUAGE else "·" for lang in s.languages))
""")

md(r"""
## Overfit testi (asıl eğitimden önce)

Bu bölüm 3–8. adımlardan bağımsızdır. Yaklaşık 100 klip üzerinde kısa bir eğitim yapıp şunları kontrol eder:

* GPU'da eğitim, warmstart, export ve sentez zinciri uçtan uca çalışıyor mu?
* Kayıplar (`train_mel`, `train_dur`) düşüyor mu, yani model öğreniyor mu?
* Eğitim cümleleri anlaşılır biçimde tekrar üretilebiliyor mu?
* Dil ID'si (`lid`) çıktıyı gerçekten etkiliyor mu? (aynı fonemler, yalnızca `lid` farklı)

**Ne ölçmez:** PL-BERT'in faydasını. Overfit, ön eğitimin hedeflediği genelleme yeteneğini tam olarak devre dışı bırakır. PL-BERT için ayrı bir duman testi aşağıda var; kalite karşılaştırması tam koşuda yapılır.

**Veri:** Aynı kişinin hem Türkçe hem İngilizce konuştuğu açık bir TTS veriseti Hugging Face'te yok. Bu yüzden iki farklı konuşmacı kullanılıyor:
* Türkçe: [`Anilosan15/Turkish_TTS_Data`](https://huggingface.co/datasets/Anilosan15/Turkish_TTS_Data), tek kadın konuşmacı (`sila`), 48 kHz, sesli kitap. **Lisans etiketi yok:** yalnızca test için kullanın.
* İngilizce: [`MikhailT/lj-speech`](https://huggingface.co/datasets/MikhailT/lj-speech), LJSpeech (public domain).

Her biri için yalnızca **bir parquet dosyası** indirilir (~500 MB). Model, Türkçe `tr_TR-dfki-medium` checkpoint'inden warmstart ile başlar, böylece birkaç bin adımda anlaşılır ses beklenir.

> **Bilinen sınır:** Tüm Türkçe klipler `sila`, tüm İngilizce klipler `lj`. Konuşmacı ve dil veride tamamen örtüşüyor, dolayısıyla "`sila` İngilizce konuşuyor" durumu için hiçbir eğitim sinyali yok. Bu çapraz sentez çalışır ama `lj`'nin tınısı sızabilir. Gerçek çözüm aynı sesten iki dilde veri (kayıt ya da teacher modelle sentetik).
""")

code(r"""
import io, csv, json
import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OVERFIT_N = 50     # konuşmacı başına eğitim klibi
OVERFIT_HELD = 3   # konuşmacı başına eğitimde görülmeyen cümle (yalnızca dinleme için)

OVERFIT_DIR = LOCAL / "overfit"
OVERFIT_WAV = OVERFIT_DIR / "wav"
OVERFIT_CSV = OVERFIT_WAV / "metadata.csv"
OVERFIT_LEXICON = OVERFIT_DIR / "lexicon.tsv"
OVERFIT_HELD_JSON = OVERFIT_DIR / "held_out.json"

# (konuşmacı, dil, HF repo, parquet dosyası, metin sütunu)
OVERFIT_SOURCES = [
    ("sila", "tr", "Anilosan15/Turkish_TTS_Data", "data/train-00000-of-00041.parquet", "text"),
    ("lj", "en-us", "MikhailT/lj-speech", "data/full-00000-of-00008-0cc4e31b6b21f574.parquet", "normalized_text"),
]

def pick_clips(repo, filename, text_col, n, min_sec=2.0, max_sec=10.0):
    path = hf_hub_download(repo, filename, repo_type="dataset")
    clips = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["audio", text_col]):
        for row in batch.to_pylist():
            text = " ".join(row[text_col].split())
            if len(text.split()) < 5 or "|" in text or text[-1] not in ".!?":
                continue
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if not (min_sec <= len(audio) / sr <= max_sec):
                continue
            clips.append((audio, sr, text))
            if len(clips) >= n:
                return clips
    raise RuntimeError(f"{repo}: yalnızca {len(clips)} uygun klip bulundu")

if not OVERFIT_CSV.exists():
    OVERFIT_WAV.mkdir(parents=True, exist_ok=True)
    rows, held = [], {}
    for speaker, language, repo, filename, text_col in OVERFIT_SOURCES:
        clips = pick_clips(repo, filename, text_col, OVERFIT_N + OVERFIT_HELD)
        held[speaker] = {"language": language, "texts": [t for _, _, t in clips[OVERFIT_N:]]}
        for i, (audio, sr, text) in enumerate(clips[:OVERFIT_N]):
            utt = f"{speaker}_{i:03d}"
            # Örnekleme hızı olduğu gibi yazılır; eğitim 22050 Hz'e kendisi çevirir.
            sf.write(OVERFIT_WAV / f"{utt}.wav", audio, sr)
            rows.append([utt, speaker, language, text])
    with open(OVERFIT_CSV, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|").writerows(rows)
    OVERFIT_HELD_JSON.write_text(json.dumps(held, ensure_ascii=False, indent=1), encoding="utf-8")
    OVERFIT_LEXICON.write_text("".join(f"{w}\ten-us\n" for w in ["meeting", "deadline", "email", "online"]), encoding="utf-8")

!wc -l {OVERFIT_CSV}
!head -n 3 {OVERFIT_CSV}
!grep -m 3 "|lj|" {OVERFIT_CSV}
""")

code(r"""
OVERFIT_EPOCHS = 50    # 100 klip / batch 16 = 6 adım/epoch → 300 adım (T4 fp32 ≈ 0.43 it/s → ~12 dk)
OVERFIT_RUN = LOCAL / "overfit_run"
OVERFIT_CONFIG = OVERFIT_RUN / "voice.onnx.json"

# Türkçe medium ses: tek konuşmacı, tek dil. Yeni katmanlar (konuşmacı, dil, MRD) sıfırdan başlar.
DFKI_CKPT = hf_hub_download("rhasspy/piper-checkpoints",
                            "tr/tr_TR/dfki/medium/epoch=5679-step=1489110.ckpt",
                            repo_type="dataset")

languages_arg = "[" + ", ".join(LANGUAGES) + "]"
overfit_resume = sorted(OVERFIT_RUN.glob("lightning_logs/version_*/checkpoints/last.ckpt"),
                        key=lambda p: p.stat().st_mtime)
overfit_resume_arg = f"--ckpt_path {overfit_resume[-1]}" if overfit_resume else ""

!python -m piper.train fit \
    --data.voice_name overfit \
    --data.csv_path {OVERFIT_CSV} \
    --data.audio_dir {OVERFIT_WAV} \
    --data.cache_dir {OVERFIT_DIR / "cache"} \
    --data.config_path {OVERFIT_CONFIG} \
    --data.espeak_voice tr \
    --data.languages "{languages_arg}" \
    --data.lexicon_path {OVERFIT_LEXICON} \
    --data.batch_size 16 \
    --data.num_workers 2 \
    --data.validation_split 0 \
    --data.num_test_examples 0 \
    --model.num_speakers 2 \
    --model.sample_rate 22050 \
    --model.use_mrd true \
    --model.mos_metric none \
    --model.lr_final_ratio 1.0 \
    --model.warmstart_ckpt {DFKI_CKPT} \
    --trainer.precision {TTS_PRECISION} \
    --trainer.max_epochs {OVERFIT_EPOCHS} \
    --trainer.log_every_n_steps 5 \
    --trainer.default_root_dir {OVERFIT_RUN} \
    {overfit_resume_arg}
""")

md(r"""
### Overfit sonuçları

1. **Kayıplar:** `train_mel` belirgin şekilde düşmeli (warmstart'lı bir medium seste tipik olarak ~0.5'in altına iner). Düşmüyorsa veri ya da fonem–ses hizası bozuktur.
2. **Eğitim cümleleri:** iki konuşmacıda da anlaşılır olmalı. Olmuyorsa test başarısızdır.
3. **Görülmemiş cümleler:** 100 klipte bozuk çıkması normaldir; zayıf bir genelleme sinyalidir, karar ölçütü değildir.
4. **Çapraz konuşmacı ve code-switching:** yukarıdaki "bilinen sınır" nedeniyle en zayıf kanıttır.
5. **`lid` ablasyonu:** aynı fonem ID'leri, yalnızca `lid` farklı. İki ses birbirinden duyulur şekilde farklıysa ve sayısal fark sıfırdan büyükse dil embedding'i modele ulaşıyor ve kullanılıyor demektir.
""")

code(r"""
import csv, json
import numpy as np
import soundfile as sf
from IPython.display import Audio, display
from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.phonemize_multilingual import phonemes_to_ids_with_languages

# 1. Kayıplar
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(sorted(OVERFIT_RUN.glob("lightning_logs/version_*"))[-1]))
    acc.Reload()
    for tag in ("train_mel", "train_dur", "train_kl"):
        values = [s.value for s in acc.Scalars(tag)]
        print(f"{tag:10s} ilk: {np.mean(values[:3]):.3f}   son: {np.mean(values[-3:]):.3f}   ({len(values)} kayıt)")
except Exception as err:
    print("Kayıplar okunamadı:", err)

# Export
overfit_ckpts = sorted(OVERFIT_RUN.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
if not overfit_ckpts:
    print("Bulunan dosyalar:", sorted(str(p) for p in OVERFIT_RUN.rglob("*"))[:30])
    raise RuntimeError(f"{OVERFIT_RUN} altında checkpoint yok: eğitim hücresi en az bir epoch "
                       "tamamlamadan durdu ya da hiç çalışmadı. Eğitim hücresini yeniden çalıştırın.")
overfit_ckpt = overfit_ckpts[-1]
print("checkpoint:", overfit_ckpt)
OVERFIT_ONNX = OVERFIT_RUN / "overfit.onnx"
!python -m piper.train.export_onnx --checkpoint {overfit_ckpt} --output-file {OVERFIT_ONNX}
!cp {OVERFIT_CONFIG} {OVERFIT_ONNX}.json

ov = PiperVoice.load(OVERFIT_ONNX)
print("inputs:", [i.name for i in ov.session.get_inputs()], "| speakers:", ov.config.speaker_id_map)

# Sesler WAV olarak da kaydedilir: Colab oynatıcısı ses vermezse en sonda zip olarak iner.
SAMPLES = OVERFIT_RUN / "samples"
SAMPLES.mkdir(exist_ok=True)
sample_idx = 0

def ov_say(label, text, speaker, language=None):
    # normalize_audio=False: ham çıktıya bakılır (NaN ya da sessizlik gizlenmesin)
    cfg = SynthesisConfig(speaker_id=ov.config.speaker_id_map[speaker], language=language,
                          normalize_audio=False)
    audio = np.concatenate([c.audio_float_array for c in ov.synthesize(text, syn_config=cfg)])
    nan = int(np.isnan(audio).sum())
    audio = np.nan_to_num(audio)
    global sample_idx
    sample_idx += 1
    path = SAMPLES / f"{sample_idx:02d}_{label}_{speaker}_{language or ov.config.espeak_voice}.wav"
    sf.write(path, audio / max(np.abs(audio).max(), 1e-8), ov.config.sample_rate, subtype="PCM_16")
    print(f"[{label}] {speaker}/{language or ov.config.espeak_voice}: {text}")
    print(f"    {path.name} | {len(audio) / ov.config.sample_rate:.2f} s | tepe {np.abs(audio).max():.3f} | "
          f"rms {np.sqrt((audio ** 2).mean()):.4f} | NaN {nan}")
    display(Audio(filename=str(path)))

train_rows = list(csv.reader(open(OVERFIT_CSV, encoding="utf-8"), delimiter="|"))
held = json.loads(OVERFIT_HELD_JSON.read_text(encoding="utf-8"))

# 2. Eğitim cümleleri
for speaker in ("sila", "lj"):
    for utt, spk, language, text in [r for r in train_rows if r[1] == speaker][:2]:
        ov_say("eğitim", text, spk, language)
        display(Audio(str(OVERFIT_WAV / f"{utt}.wav")))  # orijinal kayıt

# 3. Görülmemiş cümleler
for speaker, h in held.items():
    ov_say("görülmemiş", h["texts"][0], speaker, h["language"])

# 4. Çapraz konuşmacı ve code-switching
en_train = next(r for r in train_rows if r[1] == "lj")[3]
ov_say("çapraz", en_train, "sila", "en-us")
ov_say("code-switch", "Bugün meeting'e geç kaldım, deadline yarın.", "sila")
ov_say("code-switch", "Yarın <en-us>online</en-us> bir toplantı var.", "sila")

# 5. lid ablasyonu: aynı fonem ID'leri, yalnızca dil ID'si farklı
text = "Hello, how are you today?"
phonemes, langs = ov.phonemize_with_languages(text, "en-us")[0]
ids, _ = phonemes_to_ids_with_languages(phonemes, langs, ov.config.phoneme_id_map,
                                        ov.config.language_id_map, sentence_language="en-us")
cfg = SynthesisConfig(speaker_id=ov.config.speaker_id_map["lj"], noise_scale=0.0, noise_w_scale=0.0)
outs = {}
for language, lid in ov.config.language_id_map.items():
    outs[language] = ov.phoneme_ids_to_audio(ids, cfg, language_ids=[lid] * len(ids))
    print(f"[lid ablasyonu] lid={lid} ({language}), {len(outs[language]) / ov.config.sample_rate:.2f} s")
    ablation_path = SAMPLES / f"lid_{lid}_{language}.wav"
    sf.write(ablation_path, outs[language], ov.config.sample_rate, subtype="PCM_16")
    display(Audio(filename=str(ablation_path)))
a, b = outs["tr"], outs["en-us"]
n = min(len(a), len(b))
print(f"süre farkı: {abs(len(a) - len(b)) / ov.config.sample_rate:.3f} s | "
      f"ortak kısımda ortalama mutlak fark: {np.abs(a[:n] - b[:n]).mean():.4f}")

# Tüm sesleri indir (Colab oynatıcısında ses gelmezse bunları dinleyin)
!cd {OVERFIT_RUN} && rm -f samples.zip && zip -qr samples.zip samples
from google.colab import files
files.download(str(OVERFIT_RUN / "samples.zip"))
""")

md(r"""
### PL-BERT duman testi (isteğe bağlı, ~10 dk)

Yalnızca "ön eğitim öğreniyor mu?" sorusuna bakar: dil başına 5.000 Wikipedia cümlesi ve 1.500 adım. Beklenen:
* `val_mlm_acc` belirgin şekilde artar (rastgele tahmin ≈ 1/150'dir).
* `lm` modunda `val_word` 0'dan büyüktür ve düşer.

Faydası bu testle ölçülemez; o, 6–7. adımlardaki A/B'nin işidir.
""")

code(r"""
from datasets import load_dataset
import re

SMOKE = LOCAL / "plbert_smoke"
SMOKE.mkdir(parents=True, exist_ok=True)
smoke_corpus = SMOKE / "corpus.txt"
_split = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])")

if not smoke_corpus.exists():
    with open(smoke_corpus, "w", encoding="utf-8") as out:
        for language, config in (("tr", "20231101.tr"), ("en-us", "20231101.en")):
            n = 0
            for article in load_dataset("wikimedia/wikipedia", config, split="train", streaming=True):
                for sentence in _split.split(" ".join(article["text"].split())):
                    if 20 <= len(sentence) <= 250 and "|" not in sentence:
                        out.write(f"{language}|{sentence}\n")
                        n += 1
                if n >= 5000:
                    break

!python -m piper.train.plbert.prepare --corpus {smoke_corpus} --output-dir {SMOKE / "data"} \
    --languages {" ".join(LANGUAGES)} --num-workers {os.cpu_count()}
!python -m piper.train.plbert fit \
    --data.data_dir {SMOKE / "data"} \
    --data.batch_size 64 \
    --model.word_target lm \
    --model.lm_name xlm-roberta-base \
    --model.warmup_steps 200 \
    --trainer.max_steps 1500 \
    --trainer.val_check_interval 250 \
    --trainer.precision {PLBERT_PRECISION} \
    --trainer.default_root_dir {SMOKE / "run"}
""")


md(r"""
## 3. Lexicon (yabancı kelimeler)

espeak-ng'nin Türkçe sesi İngilizce kelimeleri **tanımaz** ve Türkçe harf kurallarıyla okur (`meeting` → `meetɪnɡ`). Bunları ya metinde `<en-us>…</en-us>` ile işaretleyin ya da bu dosyaya ekleyin. Satır biçimi: `kelime<TAB>dil`.
Kesme işaretinden sonraki ek ev sahibi dilde kalır: `meeting'e` → `meeting` (en-us) + `e` (tr).

Lexicon config'e yazılır ve runtime'da da kullanılır. Ön eğitim ile ses eğitiminde **aynı dosyayı** kullanın.
""")

code(r"""
LEXICON = WORK / "lexicon.tsv"
if not LEXICON.exists():
    starter = "".join(f"{w}\ten-us\n" for w in ['meeting', 'deadline', 'email', 'online', 'software', 'update', 'feedback', 'weekend', 'startup', 'download'])
    LEXICON.write_text(starter, encoding="utf-8")
print(LEXICON.read_text(encoding="utf-8"))
""")

md(r"""
## 4. Ön eğitim korpusu

Varsayılan olarak Wikipedia'dan dil başına `SENTENCES_PER_LANGUAGE` cümle alınır. Kendi metninizi de kullanabilirsiniz: satır başına `dil|metin`.
Konuşma diline yakın metin (altyazı, forum, diyalog) prozodi için Wikipedia'dan daha faydalıdır. Mümkünse ekleyin.
""")

code(r"""
import re
from datasets import load_dataset

SENTENCES_PER_LANGUAGE = 300_000
WIKI_CONFIGS = {"tr": "20231101.tr", "en-us": "20231101.en"}

CORPUS = WORK / "corpus.txt"
_sentence_split = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])")

if not CORPUS.exists():
    with open(CORPUS, "w", encoding="utf-8") as out:
        for language in LANGUAGES:
            wiki = load_dataset("wikimedia/wikipedia", WIKI_CONFIGS[language], split="train", streaming=True)
            n = 0
            for article in wiki:
                for paragraph in article["text"].split("\n"):
                    for sentence in _sentence_split.split(paragraph.strip()):
                        sentence = " ".join(sentence.split())
                        if not (20 <= len(sentence) <= 250) or "|" in sentence:
                            continue
                        out.write(f"{language}|{sentence}\n")
                        n += 1
                if n >= SENTENCES_PER_LANGUAGE:
                    break
            print(language, n)

!wc -l {CORPUS}
!shuf -n 5 {CORPUS}
""")

md("## 5. Fonemleştirme (CPU, bir kez)\n\nespeak-ng sesi süreç başına global olduğu için paralellik süreçlerle yapılır.")

code(r"""
PLBERT_DATA = WORK / "plbert_data"
if not (PLBERT_DATA / "meta.json").exists() or not list(PLBERT_DATA.glob("shard_*.pt")):
    !python -m piper.train.plbert.prepare \
        --corpus {CORPUS} \
        --output-dir {PLBERT_DATA} \
        --languages {" ".join(LANGUAGES)} \
        --lexicon {LEXICON} \
        --num-workers {os.cpu_count()}
!ls -la {PLBERT_DATA} | head
""")

md(r"""
## 6. Text encoder ön eğitimi

`WORD_TARGET`:
* `"lm"` (beklenen kazanan, ölçülmedi): dondurulmuş dil modelinin kelime vektörleri. Çok dilli için `xlm-roberta-base`, yalnızca Türkçe için `dbmdz/bert-base-turkish-cased`. Dil modeli yalnızca burada çalışır ve checkpoint'e girmez.
* `"vocab"`: klasik PL-BERT (kelime ID'si).

Encoder boyutları ses modelininkiyle **aynı** olmalıdır (varsayılanlar Piper medium: 192/768/2 head/6 katman). Uyuşmazlık, ses eğitimi başında hata olarak yakalanır.
İlerlemeyi `val_mlm_acc` (maskeli fonem doğruluğu) ile izleyin. `lm` modunda `val_word` 0'dan büyük olmalı; 0 kalıyorsa kelimeler dil modelinin token'larıyla eşleşmiyor demektir ve ön eğitim yalnızca MLM'e düşmüştür.
""")

code(r"""
WORD_TARGET = "lm"            # "lm" veya "vocab"
LM_NAME = "xlm-roberta-base"  # WORD_TARGET="lm" için
PLBERT_STEPS = 100_000
PLBERT_BATCH = 64

PLBERT_RUN = WORK / f"plbert_{WORD_TARGET}"

def last_ckpt(run_dir):
    # En son çalıştırmanın last.ckpt'si (aynı oturumda devam etmek için)
    ckpts = sorted(Path(run_dir).glob("lightning_logs/version_*/checkpoints/last.ckpt"),
                   key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None

resume = last_ckpt(PLBERT_RUN)
resume_arg = f"--ckpt_path {resume}" if resume else ""
print("resume:", resume)

!python -m piper.train.plbert fit \
    --data.data_dir {PLBERT_DATA} \
    --data.batch_size {PLBERT_BATCH} \
    --data.num_workers 2 \
    --model.word_target {WORD_TARGET} \
    --model.lm_name {LM_NAME} \
    --trainer.max_steps {PLBERT_STEPS} \
    --trainer.precision {PLBERT_PRECISION} \
    --trainer.val_check_interval 2000 \
    --trainer.default_root_dir {PLBERT_RUN} \
    {resume_arg}

TEXT_ENCODER_CKPT = last_ckpt(PLBERT_RUN)
print("text encoder:", TEXT_ENCODER_CKPT)
""")

md(r"""
## 7. Ses eğitimi

Aynı veriyle iki model eğitip karşılaştırın:
* `RUN_NAME="plbert"`: ön eğitimli encoder ile (`--model.text_encoder_ckpt`)
* `RUN_NAME="baseline"`: encoder'sız

İsteğe bağlı olarak var olan bir Piper checkpoint'inden başlayabilirsiniz (`WARMSTART_CKPT`, örn. Türkçe bir medium ses). Bu durumda `--ckpt_path` değil `--model.warmstart_ckpt` kullanılır, çünkü yeni dil embedding'i katı yüklemeyi bozar. Sıralama: önce warmstart, sonra encoder ön eğitimi `enc_p`'nin üzerine yazılır.

MRD (çok çözünürlüklü STFT discriminator) açıktır. Yalnızca eğitimde çalışır, inference maliyeti yoktur.
""")

code(r"""
RUN_NAME = "plbert"   # "plbert" veya "baseline"
MAX_EPOCHS = 300
TTS_BATCH = 16        # T4 (16 GB) için güvenli; L4/A100'de 32
WARMSTART_CKPT = None # örn. WORK / "base" / "tr_medium.ckpt"

TTS_RUN = WORK / f"tts_{RUN_NAME}"
TTS_CONFIG = TTS_RUN / "voice.onnx.json"
CACHE = LOCAL / f"cache_{RUN_NAME}"

languages_arg = "[" + ", ".join(LANGUAGES) + "]"
extra = []
if RUN_NAME == "plbert":
    assert TEXT_ENCODER_CKPT, "Önce 6. adımı çalıştırın"
    extra.append(f"--model.text_encoder_ckpt {TEXT_ENCODER_CKPT}")
if WARMSTART_CKPT:
    extra.append(f"--model.warmstart_ckpt {WARMSTART_CKPT}")
resume = last_ckpt(TTS_RUN)
if resume:
    extra.append(f"--ckpt_path {resume}")  # devam ederken warmstart/encoder yüklemesi atlanır
print("resume:", resume)
extra_args = " ".join(extra)

!python -m piper.train fit \
    --data.voice_name {RUN_NAME} \
    --data.csv_path {TTS_CSV} \
    --data.audio_dir {TTS_AUDIO_DIR} \
    --data.cache_dir {CACHE} \
    --data.config_path {TTS_CONFIG} \
    --data.espeak_voice {PRIMARY_LANGUAGE} \
    --data.languages "{languages_arg}" \
    --data.lexicon_path {LEXICON} \
    --data.batch_size {TTS_BATCH} \
    --data.num_workers 2 \
    --model.sample_rate {SAMPLE_RATE} \
    --model.use_mrd true \
    --trainer.precision {TTS_PRECISION} \
    --trainer.max_epochs {MAX_EPOCHS} \
    --trainer.default_root_dir {TTS_RUN} \
    {extra_args}
""")

md("## 8. ONNX export ve dinleme")

code(r"""
from IPython.display import Audio, display
from piper import PiperVoice
from piper.config import SynthesisConfig
import numpy as np

ckpt = last_ckpt(TTS_RUN)
ONNX = TTS_RUN / f"{RUN_NAME}.onnx"
!python -m piper.train.export_onnx --checkpoint {ckpt} --output-file {ONNX}
!cp {TTS_CONFIG} {ONNX}.json

voice = PiperVoice.load(ONNX)
print("inputs:", [i.name for i in voice.session.get_inputs()])

def say(text, language=None):
    chunks = list(voice.synthesize(text, syn_config=SynthesisConfig(language=language)))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    print(text)
    display(Audio(audio, rate=voice.config.sample_rate))

say("Bugün meeting'e geç kaldım, deadline yarın.")
say("Yarın <en-us>machine learning</en-us> sunumu var, hazır mısın?")
say("This sentence is entirely in English.", language="en-us")
""")

md(r"""
## Notlar

* **libpiper (C++):** çok dilli modeli yükler ve çalıştırır, ancak metni dillere ayırmaz. Tüm fonemler sesin varsayılan dilini alır, yani code-switching yalnızca Python runtime'ında (`PiperVoice`) çalışır.
* **A/B:** iki koşuyu aynı cümlelerle dinleyin. TensorBoard'da `val_mel` ve (açıksa) `val_mos` karşılaştırılabilir: `%load_ext tensorboard` ve `%tensorboard --logdir {WORK}`.
* **Dil sırası:** `LANGUAGES` sırası ön eğitim, ses eğitimi ve export boyunca aynı kalmalıdır. Farklı sıra, encoder yüklenirken hata verir.
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
with open(Path(__file__).with_name("colab_multilingual_plbert.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(len(cells), "cells")
