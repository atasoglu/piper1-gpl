"""Phoneme-level BERT pretraining of Piper's text encoder.

Pretrains the VITS text encoder (enc_p: phoneme embedding, optional language
embedding and transformer) on phonemized text alone, as in PL-BERT (Li et al.
2023, arXiv:2301.08810): masked phoneme prediction plus predicting the word
each phoneme belongs to. The pretrained encoder is then loaded into a voice
with --model.text_encoder_ckpt. The prediction heads are thrown away, so the
voice has exactly as many parameters and runs exactly as fast as without it.
"""
