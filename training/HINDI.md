# Hindi Paradee

This fork keeps Paradee's architecture and training recipe but targets Hindi. The model is
still distilled from Kokoro-82M; Kokoro's Hindi path uses **espeak-ng** (`lang_code='h'`), not
misaki's English lexicon, so the English `web/misaki.js` conversion is not used here.

Everything language-specific lives in [`scripts/lang.py`](scripts/lang.py):

```python
LANG_CODE = "h"          # Kokoro language code
VOICE = "hf_alpha"       # hf_beta, hm_omega, hm_psi also exist
RAW_SENTS_PATH = "data/hindi_sents.json"
SENTS_PATH = "data/hindi_sents_segments.json"
MAX_PHONEMES = 400
WHISPER_LANG = "hi"
```

## Corpus

`data/hindi_sents.json` contains 12,000 unique Hindi sentences from
[`subhamgiri9748/kokoros-distill`](https://huggingface.co/datasets/subhamgiri9748/kokoros-distill),
a corpus of Kokoro-82M `hf_alpha` audio at 24 kHz (source text: `shivam/hindi_pib_processed`).
Every row uses `hf_alpha`; the released audio is 51.2 hours, about twice the English
23.9-hour Paradee corpus. The loader is
[`scripts/fetch_hindi_sents.py`](scripts/fetch_hindi_sents.py); it caches pages under
`data/.hindi_pages/`, works around the dataset server's rate limit, and shuffles with seed 0.

Paradee trains on Kokoro's **intermediates** (durations, F0, N, `t_en`, `pred_dur`), not just
the released waveform, so the teacher has to be run locally. `gen_teacher.py` skips text over
400 phonemes, which is 1,640 of the 12,000 raw rows (13.7%) and includes the 698 rows (5.8%)
that are also over the model's 510-token limit. [`scripts/prepare_hindi.py`](scripts/prepare_hindi.py)
therefore rewrites the corpus into `data/hindi_sents_segments.json`: sentences that fit are
unchanged, and longer ones become `ceil(total / 400)` roughly equal word-boundary pieces.
That gives **13,654 unique segments** (median 232 phonemes, p95 358, max 400) from the 12,000
rows, with no row mixed into another and no segment over the limit. Only two phonemes in the
whole corpus fall outside Kokoro's vocabulary (a rare combining mark), so teacher generation
drops almost nothing.

The first 200 segments are held out by every training script. Because segments are generated
in the shuffled raw order, the split is still fixed and disjoint.

Check the result before spending hours on teacher generation:

```bash
uv run python scripts/fetch_hindi_sents.py   # only needed once; resumes from cache
uv run python scripts/prepare_hindi.py
uv run python scripts/check_hindi.py         # duplicates, lengths, dropped characters
```

The released dataset audio is useful as a sanity check: `gen_teacher.py` should produce the
same voice. Check the source dataset's licence before redistributing the text; Kokoro's
weights are Apache 2.0.

## Train

Run from this folder, with `uv sync` done once. The device is auto-detected (MPS, CUDA, then
CPU); override it with `DEVICE=cpu` for `gen_teacher.py` or `--device` for the trainers.

```bash
cd training
uv sync

# 1. teacher corpus: 13,654 segments (28 shards of 500)
uv run python scripts/gen_teacher.py
uv run python scripts/regen_audio.py

# 2. text side (4.23M)
uv run python scripts/train_text.py s+mlp --steps 8000

# 3. decoder, v1.0 generator
uv run python scripts/train_decoder.py A
uv run python scripts/train_decoder.py A --gan --mel-weight 10 \
  --init-from checkpoints/dec_A/last.pt --steps 5000 --bs 8 --tag _gan10
uv run python scripts/train_decoder.py A --gan --mel-weight 3 \
  --init-from checkpoints/dec_A_gan10/last.pt --steps 5000 --bs 8 --shards 8 --tag _gan3

# 4. v1.1 frame-rate head
X=(--head v2 --fine-stft --harm-weight 2 --mod-weight 5 --shards 999 --utmos)
uv run python scripts/train_vhead.py $X --steps 40000 --eval-every 10000 --tag 2_full
uv run python scripts/train_vhead.py $X --gan --mel-weight 3 --bs 8 \
  --init-from checkpoints/vh2_full/last.pt --steps 15000 --tag 2_fullgan

# 5. one ONNX graph, fp32 + int8
uv run python scripts/export_paradee_v1_1.py checkpoints/vh2_fullgan/last.pt
```

All trainers resume from `checkpoints/.../last.pt`. Start with a few thousand steps on a
small shard count before committing to the full run.

## Evaluate

```bash
uv run python scripts/eval_full.py s+mlp A --dec-tag _gan3
uv run python scripts/rtf.py s+mlp A --dec-tag _gan3
uv run python scripts/utmos.py out/full_s+mlp_A_gan3
uv run python scripts/wer.py out/full_s+mlp_A_gan3
uv run python scripts/onnx_eval.py 200 v1.1=models/paradee_v1.1/paradee_int8.onnx
```

`wer.py`, `onnx_eval.py`, and `vh_ab.py` use `WHISPER_LANG='hi'` and keep Devanagari when
normalising. `WHISPER_MODEL='base'` is the default; `large-v3` gives a much fairer Hindi WER.
UTMOS was trained on English and is only a rough relative signal for Hindi — use native
listener A/B against the teacher.

## Inference

The model repository needs `config.json` with Kokoro's vocabulary and the two ONNX files
under `onnx/paradee.onnx` and `onnx/paradee_int8.onnx`, which is the layout `Paradee`
downloads. Copy the config from Kokoro (the Hindi model uses the same 178-token vocab):

```bash
mkdir -p models/paradee_hi
cp models/paradee_v1.1/paradee*.onnx models/paradee_hi/
curl -L https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json \
  -o models/paradee_hi/config.json
```

Then:

```python
from paradee import Paradee, SAMPLE_RATE

tts = Paradee(language="hi",
              model_path="models/paradee_hi/paradee_int8.onnx",
              config_path="models/paradee_hi/config.json")
audio = tts("नमस्ते, यह हिंदी में बोलने वाला एक छोटा मॉडल है।")
```

or, after uploading the files to the Hub:

```bash
python -m paradee "नमस्ते।" --language hi --repo-id YOUR_USER/Paradee-8M-hi -o hi.wav
```

## What to watch

- **Teacher ceiling.** Kokoro's Hindi voices are grade C in the official voice table, with
  tens of minutes of target data each. The student will inherit their accent, pacing, and
  mispronunciations. Expect a usable demo, not English-Paradee quality.
- **G2P quirks.** espeak-ng inserts syllable dots inside words and handles some nukta
  consonants loosely. The teacher saw the same phonemes, so the student imitates them; a
  better Hindi G2P would require a different teacher or real recordings.
- **Male voices.** The v1.1 head uses `K=64` harmonics and an 11 kHz cutoff. That covers a
  female F0 (~200 Hz) fully; for a ~100 Hz male voice, raise `K` in `vhead.py` (or lower the
  cutoff) before training.
- **v1.1 from scratch.** The released English head was warm-started from experiments that are
  not reproducible; the from-scratch commands are the documented but untested path. Getting
  v1.0 working first is the lower-risk route.
- **Browser.** Use an espeak-ng Hindi phonemizer and the same `config.json`; do not apply
  `web/misaki.js`, which converts English eSpeak output to misaki's spelling.
