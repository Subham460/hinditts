"""Validate a teacher cache and write the dataset-level files.

The shards are the training format.  This script is the reproducibility layer: it checks
every row against ``teacher_schema.py``, cross-checks the manifest (ids, text hashes,
splits), verifies shard sidecars/checksums, and writes:

  meta.json       counts, teacher revision, dtypes, audio hours, per-shard sha256
  splits.json     stable train/val/test id lists
  validate.json   full pass/fail report and statistics
  schema.json     machine-readable row schema
  README.md       dataset card

usage:
  validate_teacher_dataset.py --root data/teacher --manifest data/hindi_segments.jsonl
  validate_teacher_dataset.py --root data/teacher --manifest ... --allow-partial

Exit code is 0 only when there are no errors.  Warnings are recorded but do not fail.
"""
import argparse, glob, hashlib, json, os, re, statistics, subprocess, sys, time
from datetime import datetime, timezone

sys.path.insert(0, "scripts")
import torch

from lang import (
    SAMPLE_RATE, FRAME_HOP, SAMPLES_PER_FRAME, FEATURE_DIM, STYLE_DIM, N_TOKEN,
    MAX_PHONEMES, SCHEMA_VERSION, TEACHER_DIR, MANIFEST_PATH, SPLIT_SEED,
    VAL_BPS, TEST_BPS, SOURCE_DATASET,
)
from teacher_schema import check_row, schema_json, text_hash

SHARD_RE = re.compile(r"shard_(\d+)\.pt$")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: {e}") from e
            rows.append(row)
    return rows


def q(values, p):
    values = sorted(values)
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[p - 1]


def describe(values):
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0, "mean": 0}
    return {
        "min": min(values),
        "p50": q(values, 50),
        "p95": q(values, 95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def code_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        ).stdout.strip())
        return commit, dirty
    except Exception:
        return "", False


def pkg_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return ""


def espeak_version():
    try:
        out = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True, timeout=10)
        line = (out.stdout or out.stderr or "").strip().splitlines()
        return line[0] if line else ""
    except Exception:
        return ""


def dataset_card(meta):
    splits = meta["splits"]
    audio = meta["audio"]
    teacher = meta["teacher"]
    expected = meta["counts"].get("expected_examples")
    expected_s = f"{expected:,}" if isinstance(expected, int) else "unknown"
    parents = meta["counts"].get("parents")
    parents_s = f"{parents:,}" if isinstance(parents, int) else "unknown"
    return f"""# Kokoro Hindi teacher cache

A reusable distillation cache for **{teacher['language']}** / **{teacher['voice']}**,
generated with `{teacher['model']}`{f" at `{teacher['revision']}`" if teacher.get('revision') else ""}.
Every row is one phoneme-aligned teacher forward pass: the waveform plus the intermediate
values a student needs, so training never has to run Kokoro again.

## Contents

| File | What it is |
|---|---|
| `shard_XXXX.pt` | list of row dicts; the canonical training format |
| `shard_XXXX.meta.json` | row count, id range, audio seconds, sha256 |
| `meta.json` | dataset-level counts, dtypes, teacher revision, checksums |
| `splits.json` | stable train / val / test id lists |
| `validate.json` | full validation report |
| `schema.json` | machine-readable row schema |
| `README.md` | this card |

## Counts

| | |
|---|---:|
| Examples | {meta['counts']['examples']:,} |
| Expected from manifest | {expected_s} |
| Complete | {str(meta['counts']['complete']).lower()} |
| Shards | {meta['counts']['shards']:,} |
| Parents / source rows | {parents_s} |
| Audio | {audio['hours']:.2f} h |
| Sample rate | {audio['sample_rate']} Hz |
| Frame hop | {audio['frame_hop']} |
| Splits | train {splits['counts']['train']:,} / val {splits['counts']['val']:,} / test {splits['counts']['test']:,} |

## Row schema

The full description is in `schema.json`.  Shapes use `T = len(ids)` and
`F = sum(pred_dur)`.

| Field | Type | Shape | Use |
|---|---|---|---|
| `id`, `parent_id` | str | | stable joins and parent-level splits |
| `text`, `phonemes`, `ps` | str | | text-side input |
| `ids`, `token_ids` | int64 | `[T]` | Kokoro token ids (BOS/EOS included) |
| `pred_dur` | int64 | `[T]` | teacher frames per token |
| `dur` | float32 | `[T]` | pre-rounding duration prediction |
| `t_en` | float16 | `[512, T]` | teacher text-encoder features, feature-major |
| `d` | float16 | `[T, 640]` | teacher predictor text features, token-major (512 hidden + 128 style) |
| `F0`, `N` | float32 | `[2F]` | pitch and loudness, two values per frame |
| `style`, `style_index` | float16 / int | `[256]` | voice/style vector and its index |
| `audio` | int16 | `[600F]` | 24 kHz teacher waveform, exact frame grid |
| `split` | str | | `train`, `val`, `test` |
| `teacher_model`, `teacher_revision` | str | | provenance |

Invariants:

- `ids[0] == ids[-1] == 0`
- `len(pred_dur) == len(dur) == T`
- `len(F0) == len(N) == 2F`
- `len(audio) == 600F`
- no NaN/Inf in any float feature
- `text_hash == sha256(text)`
- every `id` is unique

## Loading

```python
import glob, torch

rows = []
for path in sorted(glob.glob("shard_*.pt")):
    rows += torch.load(path, map_location="cpu", mmap=True)

train = [r for r in rows if r["split"] == "train"]
val = [r for r in rows if r["split"] == "val"]
```

The Paradee trainers read `data/teacher/shard_*.pt` directly.  Keep the directory
read-only when it is mounted as a Kaggle Dataset and do **not** run `regen_audio.py`
against it; that script rewrites shards.

## Provenance

- Source text: `{meta.get('source_dataset') or SOURCE_DATASET}`
- Teacher: `{teacher['model']}` / `{teacher['voice']}` / language `{teacher['language']}`
- G2P: espeak-ng (`{teacher.get('g2p', 'hi')}`)
- Audio device: `{teacher.get('audio_device', 'unknown')}`
- Created: {meta['created_utc']}
- Code: `{meta.get('code_commit') or 'unknown'}`{ ' (dirty)' if meta.get('code_dirty') else '' }

Check the source text licence before redistributing.  Kokoro's weights are Apache 2.0.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=TEACHER_DIR)
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--report", default="data/hindi_segments_report.json")
    ap.add_argument("--allow-partial", action="store_true",
                    help="missing manifest rows are warnings, not errors")
    ap.add_argument("--no-checksums", action="store_true", help="skip shard sha256")
    ap.add_argument("--force", action="store_true", help="write meta/splits/card even if invalid")
    ap.add_argument("--max-errors", type=int, default=200)
    ap.add_argument("--expected-language", default="hi")
    ap.add_argument("--expected-voice", default="hf_alpha")
    a = ap.parse_args()

    root = a.root
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(root, "shard_*.pt")))
    if not paths:
        raise SystemExit(f"no shard_*.pt under {root}")

    manifest = None
    manifest_by_id = {}
    if a.manifest and os.path.isfile(a.manifest):
        manifest = read_manifest(a.manifest)
        for row in manifest:
            if row["id"] in manifest_by_id:
                raise SystemExit(f"manifest has duplicate id: {row['id']}")
            manifest_by_id[row["id"]] = row
        manifest_sha = sha256_file(a.manifest) if not a.no_checksums else ""
    elif a.manifest:
        print(f"warning: manifest not found: {a.manifest}", flush=True)
        manifest_sha = ""
    else:
        manifest_sha = ""

    report = read_json(a.report, {}) or {}
    seed = report.get("seed", SPLIT_SEED)
    val_bps = report.get("val_bps", VAL_BPS)
    test_bps = report.get("test_bps", TEST_BPS)

    errors, warnings = [], []
    error_count = warning_count = 0

    def err(msg):
        nonlocal error_count
        error_count += 1
        if len(errors) < a.max_errors:
            errors.append(msg)

    def warn(msg):
        nonlocal warning_count
        warning_count += 1
        if len(warnings) < a.max_errors:
            warnings.append(msg)

    seen_ids = set()
    split_ids = {"train": [], "val": [], "test": []}
    split_seen = {"train": 0, "val": 0, "test": 0}
    phoneme_counts, token_counts, frame_counts, duration_values = [], [], [], []
    audio_samples = audio_seconds = 0.0
    frame_total = sample_total = 0
    peak = 0
    clipped = 0
    dc_values = []
    silence_values = []
    length_deltas = []
    dropped_total = 0
    teacher_models = set()
    teacher_revisions = set()
    voices = set()
    languages = set()
    schema_versions = set()
    audio_devices = set()
    shard_stats = []

    for path in paths:
        name = os.path.basename(path)
        match = SHARD_RE.search(name)
        index = int(match.group(1)) if match else -1
        side_path = os.path.splitext(path)[0] + ".meta.json"
        side = read_json(side_path, None)
        try:
            rows = torch.load(path, map_location="cpu", mmap=True)
        except Exception as e:
            err(f"{name}: cannot load: {e}")
            continue
        if not isinstance(rows, list):
            err(f"{name}: expected a list, got {type(rows).__name__}")
            continue

        shard_seconds = 0.0
        shard_frames = 0
        shard_splits = {"train": 0, "val": 0, "test": 0}
        for j, row in enumerate(rows):
            rid = row.get("id", f"{name}[{j}]") if isinstance(row, dict) else f"{name}[{j}]"
            if not isinstance(row, dict):
                err(f"{name}[{j}]: row is not a dict")
                continue
            e, w, stats = check_row(
                row, expected_language=a.expected_language, expected_voice=a.expected_voice
            )
            for msg in e:
                err(f"{rid}: {msg}")
            for msg in w:
                warn(f"{rid}: {msg}")

            if rid in seen_ids:
                err(f"{rid}: duplicate id")
            seen_ids.add(rid)

            if manifest is not None:
                source = manifest_by_id.get(rid)
                if source is None:
                    err(f"{rid}: id not in manifest")
                else:
                    if row.get("text_hash") != source.get("text_hash"):
                        err(f"{rid}: text_hash differs from manifest")
                    if row.get("text") != source.get("text"):
                        err(f"{rid}: text differs from manifest")
                    if row.get("parent_id") != source.get("parent_id"):
                        err(f"{rid}: parent_id differs from manifest")
                    if row.get("segment_index") != source.get("segment_index"):
                        err(f"{rid}: segment_index differs from manifest")
                    if row.get("split") != source.get("split"):
                        err(f"{rid}: split differs from manifest")

            split = row.get("split")
            if split in split_seen:
                split_seen[split] += 1
                split_ids[split].append(rid)
                shard_splits[split] += 1
            else:
                err(f"{rid}: invalid split {split!r}")

            phoneme_counts.append(int(row.get("phoneme_count", 0) or 0))
            token_counts.append(int(row.get("token_count", len(row.get("ids", []))) or 0))
            frame_counts.append(int(row.get("frames", 0) or 0))
            duration_values.append(float(row.get("duration_seconds", 0) or 0))
            frame_total += int(row.get("frames", 0) or 0)
            sample_total += int(stats.get("samples", 0))
            audio_seconds += float(stats.get("audio_seconds", 0))
            shard_seconds += float(stats.get("audio_seconds", 0))
            shard_frames += int(row.get("frames", 0) or 0)
            dropped_total += int(row.get("dropped_phonemes", 0) or 0)
            peak = max(peak, int(stats.get("peak", 0)))
            clipped += int(stats.get("clipped", 0))
            dc_values.append(float(stats.get("dc_offset", 0)))
            silence_values.append(float(stats.get("silence", 0)))
            length_deltas.append(int(stats.get("audio_length_delta", 0)))
            teacher_models.add(str(row.get("teacher_model", "")))
            teacher_revisions.add(str(row.get("teacher_revision", "")))
            voices.add(str(row.get("voice", "")))
            languages.add(str(row.get("language", "")))
            schema_versions.add(row.get("schema_version"))
            audio_devices.add(str(row.get("audio_device", "")))

        sha = None if a.no_checksums else sha256_file(path)
        if side:
            if side.get("rows") != len(rows):
                err(f"{name}: sidecar rows {side.get('rows')} != {len(rows)}")
            if sha and side.get("sha256") and side.get("sha256") != sha:
                err(f"{name}: sidecar sha256 mismatch")
            if manifest_sha and side.get("manifest_sha256") and side.get("manifest_sha256") != manifest_sha:
                warn(f"{name}: sidecar manifest_sha256 differs from current manifest")
        else:
            warn(f"{name}: no sidecar {os.path.basename(side_path)}")
        shard_stats.append({
            "index": index,
            "file": name,
            "rows": len(rows),
            "first_id": rows[0].get("id", "") if rows else "",
            "last_id": rows[-1].get("id", "") if rows else "",
            "bytes": os.path.getsize(path),
            "sha256": sha,
            "audio_seconds": round(shard_seconds, 6),
            "frames": shard_frames,
            "splits": shard_splits,
        })
        del rows

    # audio_samples was a placeholder above; the true count is len(audio) summed, which is
    # already tracked through the per-row stats.
    audio_samples = sample_total

    missing = []
    if manifest is not None:
        missing = sorted(set(manifest_by_id) - seen_ids)
        if missing:
            msg = f"{len(missing)} manifest rows are not present"
            if a.allow_partial:
                warn(msg + " (partial cache)")
            else:
                err(msg + "; pass --allow-partial for a subset")
        extra = sorted(seen_ids - set(manifest_by_id))
        if extra:
            err(f"{len(extra)} rows are not in the manifest, e.g. {extra[:3]}")
        expected_splits = {"train": 0, "val": 0, "test": 0}
        for row in manifest:
            if row["id"] in seen_ids:
                expected_splits[row["split"]] += 1
    else:
        expected_splits = dict(split_seen)

    if split_seen["val"] == 0:
        warn("validation split is empty")
    if split_seen["test"] == 0:
        warn("test split is empty")

    indices = sorted(s["index"] for s in shard_stats)
    if indices and indices != list(range(min(indices), max(indices) + 1)):
        warn(f"shard indices are not contiguous: {indices[:5]}...")

    ok = error_count == 0
    complete = manifest is not None and not missing
    elapsed = time.time() - t0

    validation = {
        "ok": ok,
        "complete": complete,
        "partial_allowed": a.allow_partial,
        "error_count": error_count,
        "warning_count": warning_count,
        "errors": errors,
        "warnings": warnings,
        "checked_at": now(),
        "elapsed_seconds": round(elapsed, 2),
        "shards": len(paths),
        "examples": len(seen_ids),
        "expected_examples": len(manifest_by_id) if manifest is not None else None,
        "missing_examples": len(missing),
        "manifest": os.path.abspath(a.manifest) if a.manifest else "",
        "manifest_sha256": manifest_sha,
    }
    write_json(os.path.join(root, "validate.json"), validation)

    summary = {
        "examples": len(seen_ids),
        "shards": len(paths),
        "audio_seconds": audio_seconds,
        "audio_hours": audio_seconds / 3600,
        "splits": split_seen,
        "expected_splits": expected_splits,
        "phonemes": describe(phoneme_counts),
        "tokens": describe(token_counts),
        "frames": describe(frame_counts),
        "durations": describe(duration_values),
        "dropped_phonemes": dropped_total,
        "peak": peak,
        "clipped": clipped,
        "clipping_fraction": clipped / max(sample_total, 1),
        "dc_offset_mean": sum(dc_values) / max(len(dc_values), 1),
        "dc_offset_max": max((abs(x) for x in dc_values), default=0.0),
        "silence_fraction": sum(silence_values) / max(len(silence_values), 1),
        "audio_length_delta": {
            "min": min(length_deltas) if length_deltas else 0,
            "max": max(length_deltas) if length_deltas else 0,
        },
        "teacher_models": sorted(teacher_models),
        "teacher_revisions": sorted(teacher_revisions),
        "voices": sorted(voices),
        "languages": sorted(languages),
        "schema_versions": sorted(str(x) for x in schema_versions),
        "audio_devices": sorted(audio_devices),
    }

    commit, dirty = code_state()
    meta = {
        "dataset": "kokoro-hindi-teacher",
        "schema_version": SCHEMA_VERSION,
        "created_utc": now(),
        "code_commit": commit,
        "code_dirty": dirty,
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", ""),
        "kokoro": pkg_version("kokoro"),
        "misaki": pkg_version("misaki"),
        "espeak": espeak_version(),
        "source_dataset": report.get("source_dataset", SOURCE_DATASET),
        "manifest": os.path.abspath(a.manifest) if a.manifest else "",
        "manifest_sha256": manifest_sha,
        "teacher": {
            "model": (summary["teacher_models"][0] if len(summary["teacher_models"]) == 1 else summary["teacher_models"]),
            "revision": (summary["teacher_revisions"][0] if len(summary["teacher_revisions"]) == 1 else summary["teacher_revisions"]),
            "voice": (summary["voices"][0] if len(summary["voices"]) == 1 else summary["voices"]),
            "language": (summary["languages"][0] if len(summary["languages"]) == 1 else summary["languages"]),
            "g2p": "espeak-ng",
            "audio_device": (summary["audio_devices"][0] if len(summary["audio_devices"]) == 1 else summary["audio_devices"]),
        },
        "counts": {
            "examples": summary["examples"],
            "expected_examples": len(manifest_by_id) if manifest is not None else None,
            "missing_examples": len(missing),
            "complete": complete,
            "shards": summary["shards"],
            "parents": None,
        },
        "splits": {
            "seed": seed,
            "val_bps": val_bps,
            "test_bps": test_bps,
            "rule": "sha256(seed:parent_id) % 10000; val < val_bps; test < val_bps+test_bps",
            "counts": summary["splits"],
            "expected_counts": summary["expected_splits"],
        },
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "frame_hop": FRAME_HOP,
            "samples_per_frame": SAMPLES_PER_FRAME,
            "dtype": "int16",
            "seconds": round(audio_seconds, 6),
            "hours": round(audio_seconds / 3600, 6),
            "samples": audio_samples,
            "peak": summary["peak"],
            "clipped_samples": summary["clipped"],
            "clipping_fraction": summary["clipping_fraction"],
            "dc_offset_mean": summary["dc_offset_mean"],
            "dc_offset_max": summary["dc_offset_max"],
            "silence_fraction": summary["silence_fraction"],
            "length_delta_min": summary["audio_length_delta"]["min"],
            "length_delta_max": summary["audio_length_delta"]["max"],
        },
        "features": {
            "feature_dim": FEATURE_DIM,
            "style_dim": STYLE_DIM,
            "vocab_size": N_TOKEN,
            "max_phonemes": MAX_PHONEMES,
            "dtypes": {
                "ids": "int64",
                "pred_dur": "int64",
                "dur": "float32",
                "t_en": "float16",
                "d": "float16",
                "F0": "float32",
                "N": "float32",
                "style": "float16",
                "audio": "int16",
            },
            "shapes": {
                "t_en": "[512, T]",
                "d": "[T, 640]",
                "F0": "[2F]",
                "N": "[2F]",
                "audio": "[600F]",
            },
        },
        "phonemes": {
            **summary["phonemes"],
            "dropped": summary["dropped_phonemes"],
            "over_max": sum(n > MAX_PHONEMES for n in phoneme_counts),
        },
        "tokens": summary["tokens"],
        "frames": summary["frames"],
        "durations_seconds": summary["durations"],
        "shards": shard_stats,
        "checksums": {
            "algorithm": None if a.no_checksums else "sha256",
            "manifest": manifest_sha,
        },
        "validation": validation,
    }
    # Parent count is cheap after the fact.
    parent_ids = set()
    for split, ids_ in split_ids.items():
        for rid in ids_:
            parent_ids.add(rid.split("__s", 1)[0])
    meta["counts"]["parents"] = len(parent_ids)

    splits = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "val_bps": val_bps,
        "test_bps": test_bps,
        "rule": "sha256(seed:parent_id) % 10000; val < val_bps; test < val_bps+test_bps",
        "counts": summary["splits"],
        "expected_counts": summary["expected_splits"],
        "splits": split_ids,
    }

    if ok or a.force:
        write_json(os.path.join(root, "meta.json"), meta)
        write_json(os.path.join(root, "splits.json"), splits)
        write_json(os.path.join(root, "schema.json"), schema_json())
        with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as f:
            f.write(dataset_card(meta))
    else:
        print("validation failed; meta.json/splits.json/schema.json/README.md not written "
              "(use --force to write them anyway)", flush=True)

    print(f"shards      {summary['shards']}")
    print(f"examples    {summary['examples']:,}"
          f"{f' / {len(manifest_by_id):,} expected' if manifest is not None else ''}")
    print(f"audio       {summary['audio_hours']:.2f} h  ({audio_samples / 1e6:.1f} M samples)")
    print(f"splits      train {summary['splits']['train']:,}  "
          f"val {summary['splits']['val']:,}  test {summary['splits']['test']:,}")
    print(f"phonemes    min {summary['phonemes']['min']}  p50 {summary['phonemes']['p50']:.0f}  "
          f"p95 {summary['phonemes']['p95']:.0f}  max {summary['phonemes']['max']}")
    print(f"tokens      max {summary['tokens']['max']}")
    print(f"audio delta min {summary['audio_length_delta']['min']}  "
          f"max {summary['audio_length_delta']['max']} samples")
    print(f"peak        {summary['peak']}  clipped {summary['clipped']:,}  "
          f"dc {summary['dc_offset_mean']:+.2f}")
    print(f"errors      {error_count}  warnings {warning_count}")
    if errors:
        print("\nfirst errors:")
        for msg in errors[:10]:
            print(" ", msg)
    if warnings:
        print("\nfirst warnings:")
        for msg in warnings[:10]:
            print(" ", msg)
    print(f"\nwrote       {os.path.join(root, 'validate.json')}")
    if ok or a.force:
        print(f"            {os.path.join(root, 'meta.json')}")
        print(f"            {os.path.join(root, 'splits.json')}")
        print(f"            {os.path.join(root, 'schema.json')}")
        print(f"            {os.path.join(root, 'README.md')}")
    print(f"time        {elapsed:.1f} s")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
