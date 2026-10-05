# Çok dilli Piper + text encoder ön eğitimi: çalışma raporu

Tarih: 2026-10-05 · Dal: `multilingual-plbert` (fork: `atasoglu/piper1-gpl`)

Bu rapor yapılanları, ölçülenleri, ölçülemeyenleri, yolda öğrenilenleri ve
sonraki adımları bir arada tutar. Kullanım kılavuzu için
[MULTILINGUAL.md](MULTILINGUAL.md)'ye bakın.

## 1. Hedef ve kısıtlar

* **Çok dillilik / code-switching:** tek model Türkçe ve İngilizceyi konuşsun,
  Türkçe cümle içindeki İngilizce kelimeleri (`meeting'e`, `deadline`) İngilizce
  okusun.
* **Doğallık:** daha az "okunan", daha insansı prozodi.
* **Kısıt:** inference'ta parametre sayısı ve gecikme artmayacak. Yalnızca
  eğitim sırasında çalışan eklemeler ve ön eğitim serbest.

## 2. Ne yapıldı

| Parça | Ne | Inference maliyeti |
|---|---|---|
| Dil embedding'i | Her fonem ID'sine bir dil ID'si; text encoder'da fonem embedding'ine eklenir. Sıfırla başlatılır. | `dil sayısı × 192` parametre (2 dil: 384) |
| Segmentasyon | `<en-us>…</en-us>` etiketi, yabancı kelime lexicon'u (`kelime<TAB>dil`), kesme işaretinden sonraki ekin Türkçe kalması | Yok (CPU'da metin işleme) |
| Eğitim verisi | CSV'de dil sütunu (`utt\|dil\|metin`, `utt\|konuşmacı\|dil\|metin`) | — |
| Export / runtime | ONNX'e `lid` girdisi; `PiperVoice` code-switching yapar; libpiper çok dilli modeli çalıştırır ama metni bölmez | Yok |
| Text encoder ön eğitimi | `piper.train.plbert`: maskeli fonem + kelime hedefi (`vocab` ya da dondurulmuş XLM-R/BERTurk vektörleri, `lm`) | Yok (başlıklar ve dil modeli atılır) |
| MRD | Var olan çok çözünürlüklü STFT discriminator'ı notebook'ta açık | Yok (yalnızca eğitim) |
| Colab notebook | Kurulum, overfit testi, PL-BERT duman testi, tam akış (korpus → ön eğitim → A/B ses eğitimi → export) | — |

Notebook `notebooks/make_colab_notebook.py` ile üretilir; değişiklikler o
dosyada yapılmalıdır.

## 3. Ölçülenler

### CPU (bu makine)

* Birim ve uçtan uca testler: **70 geçti**, 2 atlandı (Litvanya testleri
  hariç tutuldu, bkz. §5).
* Yeni test: tek konuşmacılı, tek dilli bir checkpoint'ten çok konuşmacılı ve
  çok dilli modele warmstart → export (`sid` + `lid`) → sentez.
* Hazır Türkçe ses `tr_TR-dfki-medium` ile uyumluluk: **784 tensör kopyalandı,
  0 boyut uyuşmazlığı, 0 eksik.** Sıfırdan başlayan modüller beklenenler:
  konuşmacı koşullama katmanları, `emb_g`, `emb_lang`, MRD. dfki fonem
  haritası bu reponun haritasının alt kümesi ve ortak tüm girdilerin ID'leri
  aynı.
* Overfit hücreleri gerçek veriyle küçük ölçekte (6+6 klip, 2 adım) CPU'da
  baştan sona çalıştırıldı.

### Colab T4 overfit testi

Veri: 50 Türkçe klip (`Anilosan15/Turkish_TTS_Data`, konuşmacı "sıla") ve 50
İngilizce klip (LJSpeech), dfki'den warmstart, MRD açık, fp32, batch 16,
**50 epoch = 300 adım**, sabit öğrenme oranı.

| Metrik | İlk | Son |
|---|---|---|
| `train_mel` | 0.741 | 0.614 |
| `train_dur` | 1.838 | 1.659 |
| `train_kl` | 11.788 | 2.048 |

* Hız: **0.43 it/s** (fp32). Yanlışlıkla seçilen bf16'da 0.19 it/s idi.
* Export girdileri: `input, input_lengths, scales, sid, lid`.
* "Merhaba, bugün hava çok güzel." için ham çıktı: sıla 2.52 s (tepe 0.23,
  NaN 0), lj 2.41 s (tepe 0.45, NaN 0).
* Dinleme: indirilen sesler **"gayet anlaşılır"** (kullanıcı).

**Sonuç:** Eğitim, warmstart, export ve sentez zinciri GPU'da çalışıyor ve
model öğreniyor.

## 4. Ölçülemeyenler

* **Code-switching kalitesi:** `1_sila_tr` dosyasında "meeting/deadline"
  kelimelerinin İngilizce okunup okunmadığı ayrıca dinlenmedi.
* **`lid` ablasyonunun Colab sonucu:** Aynı fonemleri farklı dil ID'leriyle
  karşılaştıran test. Değeri raporlanmadı; yalnızca CPU'da 2 adımdan sonra
  sıfırdan farklı olduğu görüldü.
* **PL-BERT'in faydası:** Overfit testi bunu ölçemez, çünkü genellemeyi
  devre dışı bırakır. A/B yapılmadı.
* **XLM-R hedefi:** Yalnızca tokenizer düzeyinde ve küçük bir test modeliyle
  denendi; gerçek ön eğitim koşusu yapılmadı.
* **Doğallık:** MOS/UTMOS gibi bir kalite ölçümü yok.
* **Diğer:** Tam veriyle eğitim ve libpiper'da code-switching yapılmadı.

## 5. Öğrenilen dersler

### Teknik

1. **Türkçe espeak İngilizce kelimeyi tanımaz.** `meeting` → `meetɪnɡ`;
   espeak dil değişimini işaretlemez. Açık segmentasyon (etiket + lexicon)
   şart.
2. **Türkçe büyük İ.** `"İ".lower()` sonucu `i` + U+0307 olur ve lexicon
   eşleşmez. `normalize_word` ile düzeltildi; testi var.
3. **torch ≥ 2.9 ONNX export'u varsayılan olarak dynamo'ya geçti** ve
   `onnxscript` istiyor. Export eski exporter'a (`dynamo=False`) sabitlendi.
   Bu tüm modelleri etkiliyor.
4. **Dil embedding'ini sıfırla başlatmak.** Rastgele başlatma, warmstart edilen
   encoder'ın girdisine embedding'ler kadar büyük bir gürültü ekliyordu.
   Sıfırla başlatınca model ilk adımda hazır sesle birebir aynı davranıyor ve
   gradyan yine de akıyor.
5. **`torch.cuda.is_bf16_supported()` T4'te `True` döner.** Oysa bf16
   yazılımla taklit edilir ve çok yavaştır (0.19 vs 0.43 it/s). Doğru kontrol:
   compute capability ≥ 8.
6. **Colab'da `pip install -e`, çalışan kernel'de görünmez.** `.pth` dosyası
   yalnızca yeni başlayan Python oturumunda işlenir. Çözüm: `sys.path` ve
   `PYTHONPATH`'e `src` eklemek, kurulumu `--python sys.executable` ile
   yapmak.
7. **Colab'ın satır içi oynatıcısı float dizisini sessiz çaldı.** Aynı sesler
   WAV olarak indirildiğinde anlaşılırdı. Kök neden bulunamadı. Notebook artık
   her sesi WAV'a yazıyor ve zip olarak indiriyor.
8. **Dal adında `/` olması Colab GitHub linkini bozabiliyor.** Dal
   `multilingual-plbert` olarak yeniden adlandırıldı.
9. **Colab diski geçici.** Linki yeniden açmak yeni bir oturum başlatabiliyor
   ve önceki checkpoint'ler kayboluyor. "Checkpoint yazılmıyor" diye bir
   şüphe doğdu; CPU'da tekrar üretilerek yazıldığı doğrulandı.
10. **Aynı kişiden Türkçe + İngilizce TTS verisi HF'te yok.** İki farklı
    konuşmacı kullanılınca konuşmacı ve dil veride tamamen örtüşüyor; "sıla
    İngilizce konuşuyor" durumu için eğitim sinyali yok.

### Süreç

1. **Notebook'u CPU'da gerçek veriyle küçük ölçekte çalıştırmak** veri
   hazırlığı, warmstart ve export hatalarını Colab'a gitmeden yakaladı. Ama
   Colab'a özgü sorunları (Python 3.13, bf16, kernel yolu, oynatıcı)
   yakalamadı. Colab'da da önce kısa bir duman koşusu gerekiyor.
2. **Notebook repo dışındaki bir betikten üretiliyordu.** Kullanıcının
   notebook'ta yaptığı değişiklik üretimde ezildi. Üretici artık repoda.
3. **Kullanıcıdan istenen elle adımlar en aza inmeli.** Ayar seçtirmek
   yerine kodda doğru varsayılanlar kullanılmalı. Hücreler adım adım
   çalıştırılır; durumu teşhis için tek, kopyala-yapıştır bir hücre yeterli.
4. **Süre tahminleri ölçüme dayanmalı.** İlk tahmin (~2000 adım, 30–60 dk)
   gerçek hızla (0.43 it/s) uyuşmadı. Ölçülen hızdan sonra 50 epoch (~12 dk)
   seçildi ve 300 adım anlaşılır ses için yetti.
5. **Açıklanamayan sorunlar açıkça not edilmeli:**
   * Litvanya testlerindeki segfault, yalnızca yeni test dosyaları tüm
     paketle birlikte toplandığında çıkıyor; nedeni bulunamadı.
   * `script/lint`, değiştirilmemiş `tests/test_lithuanian_phonemizer.py`
     içinde iki `protected-access` uyarısıyla başarısız oluyor. Önceki
     koşuda temizdi; arada `.venv`'e paket kuruldu. Nedeni araştırılmadı.

## 6. Gelecekte daha iyi olabilecekler

### Veri (en büyük kaldıraç)

* **Aynı sesten iki dilde veri.** Ya kayıt, ya da çok dilli bir teacher TTS
  ile sentetik veri. Teacher'ın lisansı kontrol edilmeli.
* **Dil başına birden fazla konuşmacı,** ki konuşmacı ve dil birbirinden
  ayrılsın. İngilizce için LibriTTS-R (`script/libritts_r_to_csv`).
* **Code-switching içeren gerçek cümleler** ve geniş bir lexicon.
* `Anilosan15/Turkish_TTS_Data`'nın lisansı yok; yalnızca test için.

### Ölçüm

* **Ayrılmış bir test seti ve nesnel metrikler:**
  * `val_mel`, UTMOS.
  * ASR (ör. Whisper) ile karakter hata oranı. Code-switching cümleleri için
    ayrıca, İngilizce kelimelerin doğru okunup okunmadığı.
* **PL-BERT A/B:** Aynı veri ve aynı adım sayısıyla `baseline` /
  `plbert-vocab` / `plbert-lm`.
* **`lid` ablasyonu** her koşuda otomatik raporlansın.

### Model ve eğitim

* Ertelenenler: SLM (WavLM) discriminator, iSTFT decoder, teacher-TTS
  distilasyonu.
* libpiper'da (C++) code-switching segmentasyonu.
* Tam koşu için hazır Türkçe checkpoint'ten warmstart. Notebook'ta
  `WARMSTART_CKPT` şu an boş; dfki uyumluluğu doğrulandı.

### Altyapı

* **Daha hızlı GPU:** T4'te 0.43 it/s ile tam eğitim pratik değil. L4/A100
  (bf16) ya da kiralık 4090/3090 kullanılmalı; hızları ölçülmedi.
* **Colab'da eğitimi arka planda çalıştırmak,** böylece eğitim sürerken
  ara checkpoint'ler dinlenebilsin.
* **Checkpoint'leri kalıcı bir yere yazmak** (HF Hub ya da kiralık disk).
* **Overfit testine küçük bir doğrulama seti** eklenerek genelleme sinyali
  alınabilir.
* **Temizlik:** Fork'taki eski `claude/multilingual-plbert` dalı artık
  kullanılmıyor.
