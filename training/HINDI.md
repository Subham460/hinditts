# Hindi Paradee

This fork keeps Paradee's architecture and training recipe but targets Hindi. The model is
still distilled from Kokoro-82M; Kokoro's Hindi path uses **espeak-ng** (`lang_code='h'`), not
misaki's English lexicon, so the English `web/misaki.js` conversion is not used here.

Everything language-specific lives in [`scripts/lang.py`](scripts/lang.py):

```python
LANG_CODE = "h"          # Kokoro language code
VOICE = "hf_alpha"       # hf_beta, hm_omega, hm_psi also exist
MANIFEST_PATH = "data/hindi_segments.jsonl"   # canonical segment manifest
TEACHER_DIR = "data/teacher"                  # canonical cache (shard_XXXX.pt)
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
`data/.hindi_pages/`, works around the dataset server's rate limit, and writes both the
legacy text list and an id-preserving `data/hindi_sents.jsonl`.

Paradee trains on Kokoro's **intermediates** (durations, F0, N, `t_en`, `pred_dur`), not just
the released waveform, so the teacher has to be run once. The canonical training input is
now the segment manifest written by [`scripts/prepare_hindi.py`](scripts/prepare_hindi.py):

```text
data/hindi_segments.jsonl     one line per segment: id, parent_id, source_row_index,
                              segment_index, n_segments, text, phonemes, text_hash,
                              split (train/val/test)
```

On the 12,000-row corpus it produces **13,655 segments**: sentences that fit 400 phonemes
are unchanged, longer ones become `ceil(total / 400)` roughly equal word-boundary pieces.
There are 13,654 unique segment texts; the one duplicate is deliberately kept with a unique
id so that no source row loses a segment. Median length is 232 phonemes, p95 358, max 400.
Only two phonemes in the whole corpus (`̩`) fall outside Kokoro's 178-token vocabulary, so
teacher generation drops almost nothing.

The split is assigned per **parent** row by a stable hash (`sha256(seed:parent_id)`), so all
segments of one source sentence stay together. At the default 1%/1% it gives about 13,400
train / 140 val / 115 test segments. The old "first 200 rows of shard 0" convention is still
used automatically for legacy shards that have no `split` field.

Build the manifest and check it before spending hours on teacher generation:

```bash
uv run python scripts/fetch_hindi_sents.py   # only if data/hindi_sents.jsonl is absent
uv run python scripts/prepare_hindi.py --in data/hindi_sents.jsonl
uv run python scripts/check_hindi.py --in data/hindi_segments.jsonl
```

The released dataset audio is useful as a sanity check: the regenerated teacher cache should
use the same voice. Check the source dataset's licence before redistributing the text;
Kokoro's weights are Apache 2.0. The full schema, generation commands, and Kaggle runbook are
in [DATASET.md](DATASET.md).

## Build the teacher cache

Run Kokoro **once** and store the waveform plus every intermediate a student may need. The
one-shot Kaggle notebook is
[`kaggle_generate_teacher_cache.ipynb`](kaggle_generate_teacher_cache.ipynb): it installs the
dependencies, finds or downloads the source, prepares the manifest, runs a smoke test, uses
two T4s for disjoint shard ranges, validates, and publishes a Kaggle Dataset.

Locally (or in any GPU session):

```bash
uv sync

# two GPUs, disjoint shards; rerun either command to resume
CUDA_VISIBLE_DEVICES=0 uv run python scripts/gen_teacher_dataset.py \
  --manifest data/hindi_segments.jsonl --out data/teacher --start 0 --stride 2
CUDA_VISIBLE_DEVICES=1 uv run python scripts/gen_teacher_dataset.py \
  --manifest data/hindi_segments.jsonl --out data/teacher --start 1 --stride 2

# finalize: checks every row and writes meta/splits/schema/README
uv run python scripts/validate_teacher_dataset.py \
  --root data/teacher --manifest data/hindi_segments.jsonl
```

`data/teacher/` is 15–16 GB (51.2 h int16 audio plus float16 `t_en`/`d`). Mount it read-only
on Kaggle and symlink it to `data/teacher`; do **not** run `regen_audio.py` against the mount.
`gen_teacher.py` and `regen_audio.py` are the legacy generator path and are not used for the
reusable cache.

## Train

Run from this folder. Build or mount the cache first (above); the trainers read
`data/teacher/shard_*.pt` and use the dataset's `split` field for held-out data.

```bash
cd training

# text side (4.23M)
uv run python scripts/train_text.py s+mlp --steps 8000

# decoder, v1.0 generator
uv run python scripts/train_decoder.py A
uv run python scripts/train_decoder.py A --gan --mel-weight 10 \
  --init-from checkpoints/dec_A/last.pt --steps 5000 --bs 8 --tag _gan10
uv run python scripts/train_decoder.py A --gan --mel-weight 3 \
  --init-from checkpoints/dec_A_gan10/last.pt --steps 5000 --bs 8 --shards 8 --tag _gan3

# v1.1 frame-rate head
X=(--head v2 --fine-stft --harm-weight 2 --mod-weight 5 --shards 999 --utmos)
uv run python scripts/train_vhead.py $X --steps 40000 --eval-every 10000 --tag 2_full
uv run python scripts/train_vhead.py $X --gan --mel-weight 3 --bs 8 \
  --init-from checkpoints/vh2_full/last.pt --steps 15000 --tag 2_fullgan

# one ONNX graph, fp32 + int8
uv run python scripts/export_paradee_v1_1.py checkpoints/vh2_fullgan/last.pt
```

In a Kaggle training notebook, use the system environment (it already has torch/torchaudio)
and mount the published dataset:

```bash
pip install -q --no-cache-dir kokoro==0.9.4 "misaki[en]==0.9.4"
rm -rf data/teacher && ln -s /kaggle/input/kokoro-hindi-teacher data/teacher
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
