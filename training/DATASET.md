# Reusable Kokoro-Hindi teacher cache

This document describes the **data-generation** half of the Hindi fork. The goal is to run
Kokoro-Hindi once, save everything a student could need, and then spend only student-training
time afterwards.

The Word "teacher cache" means **audio plus intermediates**, not audio-only:

```text
text / phonemes / token ids
style vector + style index
teacher durations (dur, pred_dur) and alignment
teacher pitch and loudness (F0, N)
teacher text features (t_en, d)
teacher waveform (int16, 24 kHz)
source ids, parent ids, split, teacher revision, schema version
```

Audio-only datasets (including the released `subhamgiri9748/kokoros-distill`) cannot replace
this: they contain no `t_en`, `d`, `F0`, `N`, `dur`, or `pred_dur`. They are useful as a
listening/reference target, but every future distillation model would have to re-run Kokoro
to get the training targets.

## Files

```text
training/data/
├── hindi_sents.jsonl              # optional source records with ids (fetch_hindi_sents.py)
├── hindi_segments.jsonl           # canonical segment manifest (prepare_hindi.py)
├── hindi_segments_report.json     # counts, hashes, duplicates, split sizes
└── teacher/                       # canonical cache
    ├── shard_0000.pt
    ├── shard_0000.meta.json
    ├── ...
    ├── generation_start0_stride2.json
    ├── meta.json                  # counts, dtypes, teacher revision, per-shard sha256
    ├── splits.json                # stable train / val / test id lists
    ├── validate.json              # full pass/fail report and statistics
    ├── schema.json                # machine-readable row schema
    └── README.md                  # dataset card
```

The `.pt` shards are the training format. A portable export (FLAC + Parquet/`.npy` +
`manifest.jsonl`) can be derived later; do not maintain two copies during generation.

## 1. Get the source records with ids

The preferred input is the Kaggle selection file:

```text
/kaggle/working/kokoro_distill_selection/selected_texts.jsonl
```

Each line is `{"id": ..., "text": ...}`. If you do not have it, download the public dataset;
the downloader now writes an id-preserving JSONL as well as the legacy text list:

```bash
cd training
uv run python scripts/fetch_hindi_sents.py
# data/hindi_sents.json       legacy list of 12,000 texts
# data/hindi_sents.jsonl      {"id", "text", "source_row_index", ...}
```

The legacy `data/hindi_sents.json` (already in this repo) still works, but because it contains
only text, its parent ids are text hashes and `source_row_index` is the index in that list,
not the original dataset row. Prefer the JSONL when you want exact source joins.

## 2. Build the segment manifest

```bash
uv run python scripts/prepare_hindi.py \
  --in /kaggle/working/kokoro_distill_selection/selected_texts.jsonl \
  --out data/hindi_segments.jsonl \
  --texts-out data/hindi_sents_segments.json \
  --report data/hindi_segments_report.json
```

What it does:

- normalises the text (drops editorial `[]`, collapses whitespace);
- keeps `id`, `parent_id`, `source_row_index`, `segment_index`, `n_segments`;
- splits only rows longer than 400 phonemes, at word boundaries, into roughly equal parts;
- computes the espeak-ng phonemes once and stores them;
- assigns `train` / `val` / `test` by a stable parent-level hash, so all segments of one
  source row stay in the same split;
- writes the legacy text-only list for older scripts;
- records counts, sha256 hashes, duplicates and phoneme quantiles in the report.

The expected result on the 12,000-row corpus is **13,655 segments**:

```text
records 12000, parents 12000
segments 13655, unique texts 13654, duplicates 1 (kept)
phonemes min 11, p50 232, p95 358, max 400, >400 0
splits train 13,400 / val 140 / test 115
```

`val` and `test` are written first so the existing evaluation scripts, which read the first
rows of shard 0, evaluate on held-out data. The training loaders now filter on `split`, so the
remaining held-out rows are not leaked into training. The exact val/test counts move by a few
rows when the parent ids are source ids instead of text hashes, because the split hash is over
the id; the notebook asserts both splits are non-empty rather than their exact size.

Check before generating:

```bash
uv run python scripts/check_hindi.py --in data/hindi_segments.jsonl
```

## 3. Generate the teacher cache

One process per GPU, disjoint shard ranges:

```bash
# GPU 0: shards 0,2,4,...
CUDA_VISIBLE_DEVICES=0 uv run python scripts/gen_teacher_dataset.py \
  --manifest data/hindi_segments.jsonl --out data/teacher \
  --start 0 --stride 2

# GPU 1: shards 1,3,5,...
CUDA_VISIBLE_DEVICES=1 uv run python scripts/gen_teacher_dataset.py \
  --manifest data/hindi_segments.jsonl --out data/teacher \
  --start 1 --stride 2
```

Single GPU: omit `CUDA_VISIBLE_DEVICES` and use `--start 0 --stride 1`.

Options:

| Flag | Meaning |
|---|---|
| `--shard-size 500` | rows per `.pt` shard |
| `--limit N` | only the first N manifest rows (smoke test) |
| `--start`, `--end`, `--stride` | shard range; `--end` is exclusive |
| `--device cuda:0` | override the auto-detected device |
| `--teacher-revision REV` | pin the Kokoro commit and voice pack |
| `--overwrite` | regenerate existing shards instead of skipping |
| `--list` | print the shard plan and exit |

Each shard is written atomically (`shard_XXXX.pt.tmp.<pid>` then `os.replace`). A sidecar
`shard_XXXX.meta.json` records the row range, first/last id, audio seconds, frames, bytes and
sha256. Rerunning the same command skips complete shards, so a killed session resumes.

The generator refuses to overwrite a legacy `gen_teacher.py` shard (no `schema_version`)
unless you pass `--overwrite`.

### What one row contains

Shapes use `T = len(ids)` and `F = sum(pred_dur)`.

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `id` | str | | `<parent_id>__s<segment_index>` |
| `parent_id` | str | | source row id |
| `segment_index`, `n_segments` | int | | position inside the source row |
| `source_row_index`, `source_row_id`, `source_index_kind` | int/str | | join back to the source |
| `source_dataset` | str | | `subhamgiri9748/kokoros-distill` |
| `text`, `phonemes`, `ps` | str | | text-side input (`ps` is an alias for the trainers) |
| `text_hash` | str | | sha256 of `text` |
| `language`, `voice` | str | | `hi`, `hf_alpha` |
| `split` | str | | `train` / `val` / `test` |
| `style_index` | int | | index into the 510-entry voice pack |
| `style` | float16 | `[256]` | voice/style vector |
| `ids`, `token_ids` | int64 | `[T]` | Kokoro ids with BOS/EOS 0 at both ends |
| `pred_dur` | int64 | `[T]` | frames per token, every value >= 1 |
| `dur` | float32 | `[T]` | pre-rounding duration prediction |
| `t_en` | float16 | `[512, T]` | teacher text-encoder features, feature-major |
| `d` | float16 | `[T, 640]` | teacher predictor text features, token-major (512 hidden + 128 style) |
| `F0` | float32 | `[2F]` | pitch, two values per frame |
| `N` | float32 | `[2F]` | loudness/noise, two values per frame |
| `audio` | int16 | `[600F]` | 24 kHz teacher waveform, exact frame grid |
| `duration_seconds`, `audio_seconds` | float | | `len(audio) / 24000` |
| `frames` | int | | `F` |
| `teacher_model`, `teacher_revision` | str | | provenance |
| `audio_device` | str | | device that rendered the waveform |
| `audio_seed` | int | | per-row torch seed for Kokoro's stochastic source |
| `schema_version` | int | | 1 |

Invariants checked for every row:

- `ids[0] == ids[-1] == 0`
- `len(pred_dur) == len(dur) == T`
- `len(F0) == len(N) == 2F`
- `len(audio) == 600F` (the generator pads or trims the iSTFT edge)
- no NaN/Inf in `style`, `t_en`, `d`, `F0`, `N`
- `text_hash == sha256(text)`
- unique `id`
- `split` is one of `train`, `val`, `test`

## 4. Validate and finalize

```bash
uv run python scripts/validate_teacher_dataset.py \
  --root data/teacher \
  --manifest data/hindi_segments.jsonl
```

- If you are validating a subset, add `--allow-partial`; missing manifest rows are warnings.
- The script writes `validate.json` always; it writes `meta.json`, `splits.json`,
  `schema.json` and `README.md` only when validation passes (or with `--force`).
- `--no-checksums` skips the per-shard sha256 (faster, but not publishable).
- The exit code is 0 only when there are no errors.

It checks every row, cross-checks the manifest (ids, text, hashes, parent, split), verifies
sidecar row counts and sha256, and writes dataset-level audio/phoneme/token statistics.

Expected result:

```text
shards      28
examples    13,655
audio       ~51.2 h
splits      train 13,400  val 140  test 115
errors      0
complete    true
```

## 5. Kaggle and Hugging Face

The cache is 15-16 GB (about 8.8 GB audio + ~6.5 GB `t_en`/`d` + small feature arrays). It is
designed to be uploaded once and mounted read-only.

Kaggle Dataset:

```bash
# from a notebook with the kaggle CLI configured
kaggle datasets create -p data/teacher --dir-mode skip
# later updates
kaggle datasets version -p data/teacher -m "update" --dir-mode skip
```

Add `dataset-metadata.json` first:

```json
{
  "title": "Kokoro Hindi Teacher Cache",
  "id": "YOUR_KAGGLE_USER/kokoro-hindi-teacher",
  "licenses": [{"name": "other"}]
}
```

Hugging Face:

```python
from huggingface_hub import HfApi
api = HfApi(token="...")
api.create_repo("YOUR_USER/kokoro-hindi-teacher", repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path="data/teacher",
    repo_id="YOUR_USER/kokoro-hindi-teacher",
    repo_type="dataset",
)
```

The ready-to-run notebook is
[`kaggle_generate_teacher_cache.ipynb`](kaggle_generate_teacher_cache.ipynb). It installs
dependencies, finds or downloads the source, prepares the manifest, runs a smoke test, runs
two GPUs with disjoint shards, validates, and publishes.

## 6. Training from the cache

Mount the published dataset and point the trainers at it:

```bash
# a Kaggle training notebook
git clone https://github.com/Subham460/hinditts.git
cd hinditts/training
rm -rf data/teacher
ln -s /kaggle/input/kokoro-hindi-teacher data/teacher

# text side and decoder can run at the same time on two T4s
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_text.py s+mlp --steps 8000
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_decoder.py A --steps 50000
```

The trainers read `data/teacher/shard_*.pt` directly. They use the `split` field for held-out
data; legacy shards without it keep the old first-200-of-shard-0 behaviour.

Do **not** run `regen_audio.py` against the mounted Kaggle dataset: it rewrites shards and the
mount is read-only. The cache already contains the teacher waveform. If you need CPU-canonical
audio for an exact reproduction, run `regen_audio.py` on a writable copy before publishing;
it is CPU-only and takes hours.

## 7. Reuse for another student

The cache is architecture-agnostic. Depending on the student:

| Student part | Fields |
|---|---|
| text encoder / duration model | `text`, `phonemes`, `ids`, `dur`, `pred_dur`, `t_en`, `d`, `style` |
| prosody model | `pred_dur`, `dur`, `F0`, `N`, `style` |
| decoder / vocoder | `t_en`, `d`, `F0`, `N`, `pred_dur`, `audio`, `style` |
| multi-voice model | `voice`, `style`, `style_index` |
| evaluation / alignment | `pred_dur`, `frames`, `audio`, `text` |

`t_en` (`[512, T]`) and `d` (`[T, 640]`) are float16, the same layouts as the old
`gen_teacher.py` shards, so existing trainer code reads them unchanged. `d` is the input to
the teacher's `F0Ntrain`; its 640 channels are the 512 predictor hidden units plus the 128
predictor style channels. Cast to float32 if your model is sensitive. `audio` is int16;
divide by 32767 for float.

## 8. Things to watch

- **Teacher quality.** Kokoro's Hindi voices are grade C; the student inherits their accent,
  pacing and mispronunciations.
- **G2P.** espeak-ng inserts syllable dots and handles some nukta consonants loosely. The
  teacher saw the same phonemes, so the student imitates them.
- **OOV phonemes.** A tiny number of combining marks are not in Kokoro's 178-token vocabulary;
  the generator records `dropped_phonemes` and uses the raw phoneme count for `style_index`,
  matching `KPipeline.infer`.
- **Audio length.** Kokoro's iSTFT can return a few samples off `600F`. The generator pads
  or trims each waveform to exactly `600F`, so every trainer can slice by frames safely.
- **Audio reproducibility.** Kokoro's harmonic source draws a random initial phase and
  additive noise, so two unseeded forwards differ. The generator seeds torch per segment
  (`sha256(seed:id)`) and stores that value in `audio_seed`; rerunning the same manifest
  produces byte-identical audio. `train_decoder.py --matched-source` reuses the stored seed.
- **Licence.** Check the source text licence before redistributing. Kokoro's weights are
  Apache 2.0.
