"""Generates colab_multilingual.ipynb. Edit this file, not the notebook."""

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
# Piper TR+EN pilot eğitimi (~2 saat, T4)

Amaç: hazır veri setlerinden küçük, dengeli bir karışımla çok dilli (code-switching) ve **PL-BERT ile ön eğitilmiş text encoder'lı** bir Piper sesini eğitip **bir gelişme olup olmadığını ölçmek**.

Model kriterleri (hepsi bu notebook'ta):
1. **Çok dilli / code-switching:** her fonem kendi dil ID'sini taşır; yabancı kelimeler lexicon ve `<en-us>…</en-us>` ile bulunur.
2. **PL-BERT:** text encoder (`enc_p`), ses verisi olmadan Türkçe + İngilizce metin üzerinde ön eğitilir (maskeli fonem + dondurulmuş XLM-R kelime vektörleri), ses modeli bu encoder'la başlar.
3. **MRD** discriminator açık.
4. **Inference'ta ek parametre/gecikme yok:** ön eğitim başlıkları ve dil modeli export'a girmez.

| Dil | Konuşmacı | Kaynak | Süre |
|---|---|---|---|
| TR | `antalia` (hedef ses) | [cloud0day3/antalia-voice-corpus](https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus) (CC BY 4.0) | 20 dk |
| TR | `zeynep`, `ali` | [Anilosan15/Synthetic_Turkish_TTS_Data](https://huggingface.co/datasets/Anilosan15/Synthetic_Turkish_TTS_Data) (CC BY 4.0, 16 kHz sentetik) | 2 × 20 dk |
| EN | `en_1116`, `en_696`, `en_6367` | LibriTTS-R train-clean-100 (CC BY 4.0) | ~53 dk |

* Klipler 1.5–15 s, sabit seed. Toplam ~1000 klip, ~2 saat ses.
* Akış: kurulum → veri → **PL-BERT ön eğitimi (en fazla 25 dk)** → ses eğitimi (30 epoch, **en fazla 1 sa 30 dk**, `tr_TR-dfki-medium`'dan warmstart + ön eğitimli encoder) → export → ölçüm. Süre dolarsa her aşama kendiliğinden durur.
* Değerlendirme: 20 code-switching + 10 TR + 10 EN cümle; karşılaştırma tabanı hazır `tr_TR-dfki-medium`. Whisper ile İngilizce kelime isabeti / hata oranları, WavLM ile konuşmacı benzerliği.

> **Hücreleri sırayla çalıştırın.** Ayar yapmanız gerekmez. Son hücre sesleri ve sonuçları zip olarak indirir; çıktıları bana yapıştırın.
> Antalia şartları: bu verilerle eğitilen ses konuşmacıyı taklit etmek için kullanılamaz ve sentetik olduğu belirtilmelidir. Atıf: "Antalia (Patientdesk.ai)".
""")

code(r"""
import subprocess
if subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
    raise RuntimeError("GPU bulunamadı. Runtime → Change runtime type → T4 GPU seçip bu hücreyi yeniden çalıştırın.")
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("## 1. Kurulum (~5 dk)")

code(r"""
import os, sys
from pathlib import Path

REPO_URL = "https://github.com/atasoglu/piper1-gpl.git"
BRANCH = "multilingual-plbert"
WORK = Path("/content/pilot")
WORK.mkdir(parents=True, exist_ok=True)

%cd /content
if not Path("/content/piper1-gpl").exists():
    !git clone --branch {BRANCH} {REPO_URL} piper1-gpl
%cd /content/piper1-gpl
!git pull --ff-only

!pip install -q uv
!uv pip install --python {sys.executable} -q scikit-build cmake ninja cython
!uv pip install --python {sys.executable} -q -e ".[train,plbert]" datasets
!{sys.executable} setup.py build_ext --inplace > /content/build_ext.log 2>&1 && echo "espeakbridge OK"
!bash build_monotonic_align.sh && echo "monotonic_align OK"

# Düzenlenebilir kurulum çalışan oturumda görünmez: kaynak klasörü yola eklenir.
SRC = "/content/piper1-gpl/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch, piper
print("piper:", piper.__file__)
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
# is_bf16_supported() T4'te de True döner ama yazılımla taklit edilir: compute capability'ye bak.
TTS_PRECISION = "bf16-mixed" if torch.cuda.get_device_capability(0)[0] >= 8 else "32-true"
print("precision:", TTS_PRECISION)
""")

md(r"""
## 2. Veri hazırlığı (~5–10 dk)

İndirilenler: Antalia'dan yalnızca seçilen WAV'lar, Synthetic_Turkish_TTS_Data'dan 1 parquet (`ali` + `zeynep`, ~510 MB), LibriTTS-R'den 1 parquet (~490 MB).

**Lexicon:** değerlendirme cümlelerindeki İngilizce kelimeler + Antalia'daki `Bluetooth`, `spam`. Lexicon config'e yazılır ve sentezde de kullanılır.
""")

code(r"""
import io, csv, json, random, re
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

SEED = 0
MIN_SEC, MAX_SEC = 1.5, 15.0
MINUTES_PER_SPEAKER = 20

DATA = WORK / "data"
WAV = DATA / "wav"
REF = DATA / "antalia_ref"      # eğitimde kullanılmayan Antalia klipleri (konuşmacı benzerliği referansı)
CSV_PATH = DATA / "metadata.csv"
LEXICON = DATA / "lexicon.tsv"

# --- Değerlendirme cümleleri ---------------------------------------------------
# (cümle, içindeki İngilizce kelimeler)
EVAL_CS = [
    ("Bugün meeting'e geç kaldım, kusura bakmayın.", ["meeting"]),
    ("Deadline yarın sabah, bu akşam bitirmemiz lazım.", ["deadline"]),
    ("Sunum hakkında feedback verirsen çok sevinirim.", ["feedback"]),
    ("Telefonuma yeni bir update geldi ama henüz yüklemedim.", ["update"]),
    ("Bu weekend ne yapıyorsun, bir yere gidiyor musun?", ["weekend"]),
    ("Dosyayı download ettim, şimdi açmaya çalışıyorum.", ["download"]),
    ("Arkadaşım geçen yıl kendi startup'ını kurdu.", ["startup"]),
    ("Laptop'ım çok ısınıyor, servise götürmem gerekecek.", ["laptop"]),
    ("Password'ünü kimseyle paylaşma, lütfen.", ["password"]),
    ("Satış rakamlarını dashboard'da takip edebilirsin.", ["dashboard"]),
    ("Bu sprint'te üç yeni özellik çıkarmayı planlıyoruz.", ["sprint"]),
    ("Kodunu review ettim, birkaç küçük önerim var.", ["review"]),
    ("Yarınki brainstorming toplantısına herkes katılsın.", ["brainstorming"]),
    ("Projenin timeline'ını tekrar gözden geçirelim.", ["timeline"]),
    ("Online eğitim programına kayıt oldum.", ["online"]),
    ("Bana bir email atarsan detayları gönderirim.", ["email"]),
    ("Bu feature müşteriler tarafından çok beğenildi.", ["feature"]),
    ("Takımın performance değerlendirmesi gelecek hafta yapılacak.", ["performance"]),
    ("Cloud'a yedek almayı unutma, yoksa her şey kaybolur.", ["cloud"]),
    ("Yeni release'i cuma günü yayınlamayı düşünüyoruz.", ["release"]),
]
EVAL_TR = [
    "Yarın sabah erkenden yola çıkmamız gerekiyor.",
    "Kahvaltıda peynir, zeytin ve taze ekmek vardı.",
    "Bu kitabı okumayı herkese tavsiye ederim.",
    "Hava kararmadan eve dönmeye çalışalım.",
    "Toplantı saat üçte başlayacak, lütfen geç kalmayın.",
    "Çocuklar bahçede top oynuyor, sesleri buraya kadar geliyor.",
    "Geçen hafta İstanbul'da çok güzel bir konsere gittik.",
    "Siparişiniz hazırlanıyor ve en kısa sürede kargoya verilecek.",
    "Öğretmenimiz sınav sonuçlarını cuma günü açıklayacak.",
    "Doğru bildiğin yoldan şaşma, sonunda kazanırsın.",
]
EVAL_EN = [
    "The meeting has been moved to Thursday afternoon.",
    "Please send me the final report by the end of the day.",
    "I think we should take a short break before we continue.",
    "The weather was perfect for a long walk by the river.",
    "She downloaded the latest update on her new laptop.",
    "Could you review the changes before the release?",
    "We are planning a small dinner party this weekend.",
    "The train was delayed because of the heavy snow.",
    "Thank you for your feedback, it was really helpful.",
    "Learning a new language takes patience and practice.",
]
LEXICON_WORDS = sorted({w for _, ws in EVAL_CS for w in ws} | {"bluetooth", "spam"})

def clean(text):
    text = " ".join(text.replace("|", " ").replace("[", "").replace("]", "").split())
    return text

def take_minutes(items, minutes, rng):
    # items: (süre, ...) listesi; karıştırıp hedef süreye kadar al
    items = list(items)
    rng.shuffle(items)
    out, total = [], 0.0
    for it in items:
        if total >= minutes * 60:
            break
        out.append(it)
        total += it[0]
    return out

if not CSV_PATH.exists():
    WAV.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)
    rows = []  # (utt, speaker, language, text)

    # --- Antalia ---------------------------------------------------------------
    repo = "cloud0day3/antalia-voice-corpus"
    meta = [json.loads(l) for l in open(hf_hub_download(repo, "metadata.jsonl", repo_type="dataset"), encoding="utf-8")]
    short = [m for m in meta if MIN_SEC <= m["duration_seconds"] <= MAX_SEC]
    rng = random.Random(SEED)
    # Önce yabancı terim içeren kategori, sonra rastgele diğerleri
    first = [m for m in short if m["campaign_category"] == "foreign_tech_terms"]
    rest = [m for m in short if m["campaign_category"] != "foreign_tech_terms"]
    rng.shuffle(rest)
    chosen, total = [], 0.0
    for m in first + rest:
        if total >= MINUTES_PER_SPEAKER * 60:
            break
        chosen.append(m)
        total += m["duration_seconds"]
    chosen_ids = {m["clip_id"] for m in chosen}
    refs = random.Random(SEED + 1).sample([m for m in meta if m["clip_id"] not in chosen_ids], 10)
    local = Path(snapshot_download(repo, repo_type="dataset",
                                   allow_patterns=[m["file_name"] for m in chosen + refs]))
    for i, m in enumerate(chosen):
        audio, sr = sf.read(local / m["file_name"], dtype="float32")
        utt = f"antalia_{i:04d}"
        sf.write(WAV / f"{utt}.wav", audio, sr)
        rows.append((utt, "antalia", "tr", clean(m["normalized_transcript"])))
    for i, m in enumerate(refs):
        audio, sr = sf.read(local / m["file_name"], dtype="float32")
        sf.write(REF / f"ref_{i:02d}.wav", audio, sr)

    # --- Synthetic_Turkish_TTS_Data: ali + zeynep (aynı parquet'te) ------------
    path = hf_hub_download("Anilosan15/Synthetic_Turkish_TTS_Data", "data/train-00001-of-00006.parquet", repo_type="dataset")
    by_spk = {"zeynep": [], "ali": []}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["audio", "text", "speaker"]):
        for r in batch.to_pylist():
            if r["speaker"] not in by_spk:
                continue
            info = sf.info(io.BytesIO(r["audio"]["bytes"]))
            if MIN_SEC <= info.duration <= MAX_SEC:
                by_spk[r["speaker"]].append((info.duration, r["audio"]["bytes"], clean(r["text"])))
    for speaker in ("zeynep", "ali"):
        for i, (_, data, text) in enumerate(take_minutes(by_spk[speaker], MINUTES_PER_SPEAKER, random.Random(SEED))):
            utt = f"{speaker}_{i:04d}"
            (WAV / f"{utt}.wav").write_bytes(data)
            rows.append((utt, speaker, "tr", text))
    del by_spk

    # --- LibriTTS-R: 3 konuşmacı (1116, 696: kadın sesi; 6367: erkek sesi) ----
    path = hf_hub_download("mythicinfinity/libritts_r", "data/train.clean.100/train.clean.100-00000-of-00018.parquet", repo_type="dataset")
    en_speakers = ["1116", "696", "6367"]
    by_spk = {s: [] for s in en_speakers}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["audio", "text_normalized", "speaker_id"]):
        for r in batch.to_pylist():
            if r["speaker_id"] not in by_spk:
                continue
            info = sf.info(io.BytesIO(r["audio"]["bytes"]))
            text = clean(r["text_normalized"])
            if MIN_SEC <= info.duration <= MAX_SEC and len(text.split()) >= 2:
                by_spk[r["speaker_id"]].append((info.duration, r["audio"]["bytes"], text))
    for speaker in en_speakers:
        for i, (_, data, text) in enumerate(take_minutes(by_spk[speaker], MINUTES_PER_SPEAKER, random.Random(SEED))):
            utt = f"en_{speaker}_{i:04d}"
            (WAV / f"{utt}.wav").write_bytes(data)
            rows.append((utt, f"en_{speaker}", "en-us", text))
    del by_spk

    random.Random(SEED).shuffle(rows)
    # Konuşmacı ID'leri CSV'de ilk görülme sırasına göre verilir: antalia hep 0 olsun
    rows.sort(key=lambda r: r[1] != "antalia")
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|").writerows(rows)
    LEXICON.write_text("".join(f"{w}\ten-us\n" for w in LEXICON_WORDS), encoding="utf-8")

# Özet
rows = list(csv.reader(open(CSV_PATH, encoding="utf-8"), delimiter="|"))
NUM_SPEAKERS = len({r[1] for r in rows})
stats = {}
for utt, speaker, language, text in rows:
    s = stats.setdefault(speaker, [language, 0, 0.0])
    s[1] += 1
    s[2] += sf.info(WAV / f"{utt}.wav").duration
total_sec = sum(s[2] for s in stats.values())
for speaker, (language, n, sec) in stats.items():
    print(f"{speaker:10s} {language:6s} {n:5d} klip  {sec / 60:5.1f} dk  ort {sec / n:4.1f} s")
print(f"TOPLAM: {len(rows)} klip, {total_sec / 3600:.2f} saat, {NUM_SPEAKERS} konuşmacı")
lex = set(LEXICON.read_text(encoding="utf-8").split()[::2])
hits = [r for r in rows if r[2] == "tr" and any(re.sub(r"['’].*", "", w).lower() in lex for w in r[3].split())]
print(f"Lexicon kelimesi içeren TR eğitim klibi: {len(hits)}")
for r in hits[:5]:
    print("   ", r[1], "|", r[3][:100])
""")

md(r"""
## 3. PL-BERT: text encoder ön eğitimi (≤ 25 dk)

Encoder yalnızca fonemleri görür, cümle bağlamını bilmez; bu da "okunan" prozodinin bir nedeni. Burada `enc_p`, ses olmadan metin üzerinde ön eğitilir:
* **Korpus:** Wikipedia (TR + EN, dil başına 40.000 cümle) + pilotun kendi konuşma metinleri (konuşma diline yakın).
* **Hedef:** maskeli fonem (tüm kelime maskeleme) + `lm`: her fonemin ait olduğu kelimenin, dondurulmuş `xlm-roberta-base` ile üretilmiş bağlamsal vektörü (kosinüs kaybı). Türkçe gibi eklemeli dillerde kelime sözlüğü seyrek kaldığı için `vocab` hedefinden daha iyi olması beklenir; bu henüz ölçülmedi.
* Dil modeli yalnızca burada çalışır; checkpoint'e ve export'a girmez. Tahmin başlıkları atılır, yalnızca encoder ses modeline yüklenir.
* Sağlık kontrolü: `val_mlm_acc` artmalı; `val_word` **0'dan büyük** kalmalı (0 ise kelimeler dil modelinin token'larıyla eşleşmiyor demektir).
""")

code(r"""
import re
from datasets import load_dataset

SENTENCES_PER_LANGUAGE = 40_000
WIKI_CONFIGS = {"tr": "20231101.tr", "en-us": "20231101.en"}
CORPUS = WORK / "plbert_corpus.txt"
_sentence_split = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])")

if not CORPUS.exists():
    with open(CORPUS, "w", encoding="utf-8") as out:
        for language, config in WIKI_CONFIGS.items():
            n = 0
            for article in load_dataset("wikimedia/wikipedia", config, split="train", streaming=True):
                for paragraph in article["text"].split("\n"):
                    for sentence in _sentence_split.split(paragraph.strip()):
                        sentence = " ".join(sentence.split())
                        if 20 <= len(sentence) <= 250 and "|" not in sentence:
                            out.write(f"{language}|{sentence}\n")
                            n += 1
                if n >= SENTENCES_PER_LANGUAGE:
                    break
            print(language, n)
        # Pilotun konuşma metinleri (+ içindeki yabancı kelimeler lexicon'dan işaretlenir)
        for utt, speaker, language, text in rows:
            out.write(f"{language}|{text}\n")

PLBERT_DATA = WORK / "plbert_data"
# prepare meta.json'yi en son yazar; "num_samples" yoksa önceki çalışma yarıda kalmıştır
_meta = PLBERT_DATA / "meta.json"
if not _meta.exists() or "num_samples" not in json.loads(_meta.read_text()):
    !python -m piper.train.plbert.prepare \
        --corpus {CORPUS} \
        --output-dir {PLBERT_DATA} \
        --languages tr en-us \
        --lexicon {LEXICON} \
        --num-workers {os.cpu_count()}
!wc -l {CORPUS}
!ls {PLBERT_DATA} | head -5
""")

code(r"""
PLBERT_RUN = WORK / "plbert_run"
PLBERT_PRECISION = "bf16-mixed" if TTS_PRECISION == "bf16-mixed" else "16-mixed"

def last_ckpt_in(run_dir):
    ckpts = sorted(Path(run_dir).glob("lightning_logs/version_*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None

resume = last_ckpt_in(PLBERT_RUN)
resume_arg = f"--ckpt_path {resume}" if resume else ""
!python -m piper.train.plbert fit \
    --data.data_dir {PLBERT_DATA} \
    --data.batch_size 64 \
    --data.num_workers 2 \
    --model.word_target lm \
    --model.lm_name xlm-roberta-base \
    --model.warmup_steps 200 \
    --trainer.max_steps 100000 \
    --trainer.max_time 00:00:25:00 \
    --trainer.val_check_interval 500 \
    --trainer.precision {PLBERT_PRECISION} \
    --trainer.default_root_dir {PLBERT_RUN} \
    {resume_arg}

TEXT_ENCODER_CKPT = last_ckpt_in(PLBERT_RUN)
assert TEXT_ENCODER_CKPT, "PL-BERT checkpoint'i yok: ön eğitim hücresi hata verdi"
print("text encoder:", TEXT_ENCODER_CKPT)

# Sağlık kontrolü
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(sorted(PLBERT_RUN.glob("lightning_logs/version_*"))[-1]))
    acc.Reload()
    for tag in ("train_mlm_acc", "val_mlm_acc", "val_word"):
        if tag in acc.Tags()["scalars"]:
            v = [s.value for s in acc.Scalars(tag)]
            print(f"{tag:14s} ilk {v[0]:.3f}  son {v[-1]:.3f}  ({len(v)} kayıt)")
except Exception as err:
    print("PL-BERT kayıpları okunamadı:", err)
""")

md(r"""
## 4. Ses eğitimi (≤ 1 sa 30 dk)

* `tr_TR-dfki-medium`'dan warmstart; konuşmacı, dil ve MRD katmanları sıfırdan başlar. **Sonra** PL-BERT encoder'ı `enc_p`'nin üzerine yüklenir (`--model.text_encoder_ckpt`): ses modelinin geri kalanı dfki'den, text encoder PL-BERT'ten gelir.
* 30 epoch; öğrenme oranı bu süreye göre azalır. Süre 1 sa 30 dk'yı aşarsa eğitim durur ve son checkpoint kullanılır.
* Batch 12 (15 s'lik kliplerle T4 belleği için). GPU belleği yetmezse hücre kendiliğinden batch 8 ile devam eder.
* İlerleme çubuğundaki `it/s` değerine bakın: T4 fp32'de ~0.4–0.5 bekleniyor.
""")

code(r"""
MAX_EPOCHS = 30
MAX_TIME = "00:01:30:00"
RUN = WORK / "run"
CONFIG = RUN / "voice.onnx.json"
LOG = WORK / "train.log"

DFKI_CKPT = hf_hub_download("rhasspy/piper-checkpoints",
                            "tr/tr_TR/dfki/medium/epoch=5679-step=1489110.ckpt",
                            repo_type="dataset")

def last_ckpt():
    ckpts = sorted(RUN.glob("lightning_logs/version_*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None

def train(batch_size):
    resume = last_ckpt()
    # Devam ederken warmstart atlanır
    start_arg = (f"--ckpt_path {resume}" if resume else
                 f"--model.warmstart_ckpt {DFKI_CKPT} --model.text_encoder_ckpt {TEXT_ENCODER_CKPT}")
    print("batch:", batch_size, "| başlangıç:", resume or "dfki warmstart + PL-BERT encoder")
    !python -m piper.train fit \
        --data.voice_name pilot \
        --data.csv_path {CSV_PATH} \
        --data.audio_dir {WAV} \
        --data.cache_dir {WORK / "cache"} \
        --data.config_path {CONFIG} \
        --data.espeak_voice tr \
        --data.languages "[tr, en-us]" \
        --data.lexicon_path {LEXICON} \
        --data.batch_size {batch_size} \
        --data.num_workers 2 \
        --data.validation_split 0.05 \
        --data.num_test_examples 0 \
        --model.num_speakers {NUM_SPEAKERS} \
        --model.sample_rate 22050 \
        --model.use_mrd true \
        --model.mos_metric none \
        --trainer.precision {TTS_PRECISION} \
        --trainer.max_epochs {MAX_EPOCHS} \
        --trainer.max_time {MAX_TIME} \
        --trainer.log_every_n_steps 10 \
        --trainer.default_root_dir {RUN} \
        {start_arg} 2>&1 | tee {LOG}

train(12)
if "out of memory" in LOG.read_text(errors="ignore").lower():
    print("\n>>> GPU belleği yetmedi, batch 8 ile devam ediliyor\n")
    train(8)
print("son checkpoint:", last_ckpt())
""")

md(r"""
## 5. Export ve sentez

Pilot ses: `antalia` (hedef), `ali` (diğer TR konuşmacı) ve `en_1116` (EN konuşmacı) ile; taban: hazır `tr_TR-dfki-medium` (tek dilli, yabancı kelimeleri Türkçe kurallarla okur).
""")

code(r"""
import numpy as np
from IPython.display import Audio, display
from piper import PiperVoice
from piper.config import SynthesisConfig

# Kayıp eğrileri
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(sorted(RUN.glob("lightning_logs/version_*"))[-1]))
    acc.Reload()
    for tag in ("train_mel", "train_dur", "train_kl", "val_mel"):
        if tag in acc.Tags()["scalars"]:
            v = [s.value for s in acc.Scalars(tag)]
            print(f"{tag:10s} ilk {np.mean(v[:3]):.3f}  son {np.mean(v[-3:]):.3f}  ({len(v)} kayıt)")
except Exception as err:
    print("Kayıplar okunamadı:", err)

ckpt = last_ckpt()
print("checkpoint:", ckpt)
ONNX = RUN / "pilot.onnx"
!python -m piper.train.export_onnx --checkpoint {ckpt} --output-file {ONNX}
!cp {CONFIG} {ONNX}.json

pilot = PiperVoice.load(ONNX)
print("inputs:", [i.name for i in pilot.session.get_inputs()], "| konuşmacılar:", pilot.config.speaker_id_map)
dfki_onnx = hf_hub_download("rhasspy/piper-voices", "tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx")
hf_hub_download("rhasspy/piper-voices", "tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx.json")
dfki = PiperVoice.load(dfki_onnx)

SAMPLES = WORK / "samples"
SAMPLES.mkdir(exist_ok=True)
CLIPS = []  # her sentez için bir kayıt

def synth(system, voice, speaker, kind, idx, text, language=None, en_words=()):
    cfg = SynthesisConfig(language=language, normalize_audio=False,
                          speaker_id=voice.config.speaker_id_map[speaker] if voice.config.speaker_id_map else None)
    audio = np.concatenate([c.audio_float_array for c in voice.synthesize(text, syn_config=cfg)])
    audio = np.nan_to_num(audio)
    path = SAMPLES / f"{system}_{speaker}_{kind}_{idx:02d}.wav"
    sf.write(path, audio / max(np.abs(audio).max(), 1e-8) * 0.9, voice.config.sample_rate, subtype="PCM_16")
    CLIPS.append(dict(system=system, speaker=speaker, kind=kind, idx=idx, text=text,
                      en_words=list(en_words), path=str(path)))

for system, voice, speakers in (("pilot", pilot, ["antalia", "ali", "en_1116"]), ("dfki", dfki, ["dfki"])):
    for speaker in speakers:
        for i, (text, words) in enumerate(EVAL_CS):
            synth(system, voice, speaker, "cs", i, text, en_words=words)
        for i, text in enumerate(EVAL_TR):
            synth(system, voice, speaker, "tr", i, text)
        for i, text in enumerate(EVAL_EN):
            synth(system, voice, speaker, "en", i, text, language="en-us" if system == "pilot" else None)
print(len(CLIPS), "ses üretildi")

for c in CLIPS:
    if c["idx"] < 2 and c["speaker"] in ("antalia", "dfki"):
        print(c["system"], c["speaker"], c["kind"], "|", c["text"])
        display(Audio(filename=c["path"]))
""")

md(r"""
## 6. Ölçüm ve sonuçlar

* **Whisper (large-v3-turbo):** code-switching ve TR cümleleri Türkçe, EN cümleleri İngilizce modda yazıya dökülür.
  * `cs_en_hit`: code-switching cümlelerindeki İngilizce kelimenin transkriptte geçme oranı (yüksek = İngilizce okunmuş).
  * `tr_cer`: TR cümlelerinde karakter hata oranı (düşük = iyi). `en_wer`: EN cümlelerinde kelime hata oranı.
* **WavLM konuşmacı benzerliği:** üretilen sesin, eğitimde kullanılmayan 10 gerçek Antalia klibine kosinüs benzerliği. `antalia`'nın TR, EN ve code-switching cümlelerinde benzer kalması tınının dil değişince kaymadığını gösterir; `ali` ve `dfki` farklı ses için alt referanstır.

Whisper Türkçe telaffuz edilmiş bir kelimeyi de İngilizce yazabilir: `cs_en_hit` kaba bir ölçüdür, dinlemeyle birlikte değerlendirin.
""")

code(r"""
import torch, unicodedata
from scipy.signal import resample_poly
from transformers import pipeline, AutoFeatureExtractor, WavLMForXVector

def load16k(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_poly(audio, 16000, sr).astype(np.float32) if sr != 16000 else audio

asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               torch_dtype=torch.float16, device=0)
for c in CLIPS:
    lang = "english" if c["kind"] == "en" else "turkish"
    c["asr"] = asr({"raw": load16k(c["path"]), "sampling_rate": 16000},
                   generate_kwargs={"language": lang, "task": "transcribe"})["text"].strip()
del asr
torch.cuda.empty_cache()

def norm(text):
    text = unicodedata.normalize("NFC", text.replace("İ", "i").replace("I", "ı").lower())
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())

def norm_en(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

for c in CLIPS:
    if c["kind"] == "cs":
        hyp = c["asr"].lower()
        c["score"] = float(np.mean([w in hyp for w in c["en_words"]]))
    elif c["kind"] == "tr":
        ref, hyp = norm(c["text"]), norm(c["asr"])
        c["score"] = edit_distance(ref, hyp) / max(len(ref), 1)
    else:
        ref, hyp = norm_en(c["text"]).split(), norm_en(c["asr"]).split()
        c["score"] = edit_distance(ref, hyp) / max(len(ref), 1)

fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()

@torch.no_grad()
def embed(path):
    inputs = fe(load16k(path), sampling_rate=16000, return_tensors="pt").to("cuda")
    e = sv(**inputs).embeddings[0]
    return torch.nn.functional.normalize(e, dim=-1)

ref_emb = torch.nn.functional.normalize(torch.stack([embed(p) for p in sorted(REF.glob("*.wav"))]).mean(0), dim=-1)
for c in CLIPS:
    c["sim_antalia"] = float(embed(c["path"]) @ ref_emb)

# Tablo
import pandas as pd
df = pd.DataFrame(CLIPS)
summary = df.pivot_table(index=["system", "speaker"], columns="kind", values="score", aggfunc="mean")
summary = summary.rename(columns={"cs": "cs_en_hit", "tr": "tr_cer", "en": "en_wer"})
sims = df.pivot_table(index=["system", "speaker"], columns="kind", values="sim_antalia", aggfunc="mean")
sims.columns = [f"sim_{k}" for k in sims.columns]
table = summary.join(sims).round(3)
print(table.to_string())

print("\nCode-switching transkriptleri (pilot/antalia ve dfki):")
for i, (text, words) in enumerate(EVAL_CS):
    p = df[(df.speaker == "antalia") & (df.kind == "cs") & (df.idx == i)].iloc[0]
    d = df[(df.speaker == "dfki") & (df.kind == "cs") & (df.idx == i)].iloc[0]
    print(f"{i:2d} {words[0]:14s} pilot[{int(p.score)}]: {p.asr}")
    print(f"   {'':14s} dfki [{int(d.score)}]: {d.asr}")

df.drop(columns=["path"]).to_csv(WORK / "results.csv", index=False)
table.to_csv(WORK / "summary.csv")

!cd {WORK} && rm -f pilot_results.zip && zip -qr pilot_results.zip samples results.csv summary.csv train.log run/voice.onnx.json plbert_run/lightning_logs
from google.colab import files
files.download(str(WORK / "pilot_results.zip"))
""")

md(r"""
## Notlar

* **PL-BERT'in faydası bu turda izole ölçülmüyor:** karşılaştırma tabanı tek dilli dfki'dir. PL-BERT'li / PL-BERT'siz A/B, pilot yön gösterirse büyük koşuda yapılır.
* **Bu turda ne ölçülüyor:** (1) Antalia sesi İngilizce kelimeleri dfki'den daha İngilizce okuyor mu (`cs_en_hit`), (2) TR kalitesi korunuyor mu (`tr_cer`), (3) tını dil değişince Antalia'ya yakın kalıyor mu (`sim_*`).
* **Sınırlar:** Antalia'nın ≤15 s kliplerinde yabancı kelime çok az (yalnızca "Bluetooth", "spam"); gerçek code-switching sinyali neredeyse yok. `zeynep`/`ali` 16 kHz sentetik ses. 1–2 saatlik eğitim kalite tavanını değil, yönü gösterir.
* Checkpoint'ler `/content/pilot/run` altında; oturum kapanınca silinir. Saklamak isterseniz `run/lightning_logs/.../checkpoints/last.ckpt` ve `run/pilot.onnx` dosyalarını sol panelden indirin.
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
with open(Path(__file__).with_name("colab_multilingual.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(len(cells), "cells")
