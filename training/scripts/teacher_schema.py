"""Shared schema and row checks for the reusable Kokoro teacher cache.

One row is one phoneme-aligned teacher forward pass.  It stores the waveform *and* the
teacher's intermediate values, so a future student can reuse the cache without running
Kokoro again.  The canonical on-disk form is a list of these dicts in
``shard_XXXX.pt`` files.

The same checks are used by the generator (fail early) and by the finalizer
(``validate_teacher_dataset.py``).
"""
import hashlib

SCHEMA_VERSION = 1
SAMPLE_RATE = 24000
FRAME_HOP = 300
SAMPLES_PER_FRAME = 600
FEATURE_DIM = 512
PRED_FEATURE_DIM = 640          # 512 predictor hidden + 128 predictor style
STYLE_DIM = 256
N_TOKEN = 178
MAX_PHONEMES = 400
STYLE_INDEX_MAX = 509
SPLITS = ("train", "val", "test")

# Human-readable field list.  This is written to schema.json next to the shards so the
# dataset is self-describing even if this repository changes.
ROW_FIELDS = {
    "id": "stable segment id, <parent_id>__s<segment_index>",
    "parent_id": "id of the source dataset row (or a text hash for a plain list)",
    "segment_index": "0-based index of this segment inside its parent row",
    "n_segments": "number of segments produced from the parent row",
    "source_row_index": "0-based row index in the input file / source dataset",
    "source_row_id": "original id field, if the input had one",
    "source_index_kind": "'dataset' when source_row_index is the original dataset row, else 'input'",
    "source_dataset": "source dataset id, e.g. subhamgiri9748/kokoros-distill",
    "text": "normalised segment text",
    "text_hash": "sha256 of text",
    "phonemes": "espeak-ng phoneme string used by the teacher",
    "phoneme_count": "len(phonemes)",
    "ps": "alias of phonemes (the existing trainers read this name)",
    "token_count": "len(ids)",
    "language": "language code, e.g. hi",
    "voice": "Kokoro voice, e.g. hf_alpha",
    "split": "train / val / test",
    "style_index": "index into the 510-entry Kokoro voice pack",
    "style": "float16 [256] teacher style vector",
    "sample_rate": "24000",
    "frame_hop": "300",
    "schema_version": "1",
    "teacher_model": "teacher repo id",
    "teacher_revision": "resolved teacher commit, if known",
    "audio_device": "device that rendered the waveform, e.g. cuda:0",
    "audio_seed": "per-row torch seed used for Kokoro's stochastic source",
    "ids": "int64 [T] Kokoro token ids, including BOS/EOS",
    "token_ids": "alias of ids",
    "token_count": "len(ids)",
    "pred_dur": "int64 [T] predicted frames per token, every value >= 1",
    "frames": "sum(pred_dur), the number of 25 ms frames",
    "dur": "float32 [T] teacher duration prediction before rounding",
    "t_en": "float16 [512, T] teacher text-encoder features, feature-major",
    "d": "float16 [T, 640] teacher predictor text features, token-major (512 hidden + 128 style)",
    "F0": "float32 [2F] pitch, two values per frame",
    "N": "float32 [2F] loudness/noise, two values per frame",
    "audio": "int16 [600F] teacher waveform at 24 kHz, padded/trimmed to the frame grid",
    "duration_seconds": "len(audio) / 24000",
    "audio_seconds": "len(audio) / 24000",
    "dropped_phonemes": "phonemes not present in Kokoro's vocabulary",
}

TENSOR_FIELDS = ("style", "ids", "token_ids", "pred_dur", "dur", "t_en", "d", "F0", "N", "audio")


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_row(row, expected_language=None, expected_voice=None, max_phonemes=MAX_PHONEMES):
    """Return ``(errors, warnings, stats)`` for one teacher-cache row.

    ``stats`` contains counters the finalizer aggregates into meta.json.  This function
    deliberately imports torch lazily so the schema module stays importable in a plain
    Python process.
    """
    import torch

    errs, warns = [], []

    def err(msg):
        errs.append(msg)

    def warn(msg):
        warns.append(msg)

    stats = {
        "phonemes": 0,
        "tokens": 0,
        "frames": 0,
        "samples": 0,
        "audio_seconds": 0.0,
        "peak": 0,
        "clipped": 0,
        "dc_offset": 0.0,
        "silence": 0.0,
        "audio_length_delta": 0,
        "dropped_phonemes": 0,
    }

    if not isinstance(row, dict):
        return ["row is not a dict"], [], stats

    strings = (
        "id", "parent_id", "source_row_id", "source_index_kind", "source_dataset",
        "text", "text_hash", "phonemes", "ps", "language", "voice", "split",
        "teacher_model", "teacher_revision", "audio_device",
    )
    for key in strings:
        if not isinstance(row.get(key), str):
            err(f"{key}: expected str")

    ints = (
        "segment_index", "n_segments", "source_row_index", "phoneme_count", "token_count",
        "style_index", "sample_rate", "frame_hop", "schema_version", "frames",
        "dropped_phonemes", "audio_seed",
    )
    for key in ints:
        value = row.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            err(f"{key}: expected int")

    for key in ("duration_seconds", "audio_seconds"):
        if not isinstance(row.get(key), (int, float)) or isinstance(row.get(key), bool):
            err(f"{key}: expected float")

    if row.get("schema_version") != SCHEMA_VERSION:
        err(f"schema_version: expected {SCHEMA_VERSION}, got {row.get('schema_version')!r}")
    if row.get("sample_rate") != SAMPLE_RATE:
        err(f"sample_rate: expected {SAMPLE_RATE}, got {row.get('sample_rate')!r}")
    if row.get("frame_hop") != FRAME_HOP:
        err(f"frame_hop: expected {FRAME_HOP}, got {row.get('frame_hop')!r}")
    if row.get("split") not in SPLITS:
        err(f"split: expected one of {SPLITS}, got {row.get('split')!r}")
    if expected_language and row.get("language") != expected_language:
        err(f"language: expected {expected_language}, got {row.get('language')!r}")
    if expected_voice and row.get("voice") != expected_voice:
        err(f"voice: expected {expected_voice}, got {row.get('voice')!r}")

    text = row.get("text")
    ps = row.get("phonemes")
    if isinstance(text, str):
        if not text.strip():
            err("text: empty")
        if isinstance(row.get("text_hash"), str) and row["text_hash"] != text_hash(text):
            err("text_hash: does not match text")
    if isinstance(ps, str):
        if not ps:
            err("phonemes: empty")
        if row.get("ps") != ps:
            err("ps: does not match phonemes")
        if row.get("phoneme_count") != len(ps):
            err("phoneme_count: does not match len(phonemes)")
        if len(ps) > max_phonemes:
            err(f"phonemes: {len(ps)} > max {max_phonemes}")
        stats["phonemes"] = len(ps)

    if isinstance(row.get("n_segments"), int):
        if row["n_segments"] < 1:
            err("n_segments: must be >= 1")
        if isinstance(row.get("segment_index"), int) and not (0 <= row["segment_index"] < row["n_segments"]):
            err("segment_index: outside n_segments")
    if isinstance(row.get("source_row_index"), int) and row["source_row_index"] < 0:
        err("source_row_index: negative")

    ids = row.get("ids")
    tids = row.get("token_ids")
    if not torch.is_tensor(ids):
        err("ids: expected a tensor")
        ids = None
    if not torch.is_tensor(tids):
        err("token_ids: expected a tensor")
        tids = None
    if ids is not None and tids is not None and not torch.equal(ids, tids):
        err("token_ids: does not match ids")

    T = 0
    if ids is not None:
        if ids.ndim != 1:
            err(f"ids: expected 1-D, got {tuple(ids.shape)}")
        else:
            T = int(ids.numel())
            stats["tokens"] = T
            if ids.dtype not in (torch.int16, torch.int32, torch.int64):
                err(f"ids: expected an integer dtype, got {ids.dtype}")
            if T < 3:
                err(f"ids: too short ({T})")
            else:
                if int(ids[0]) != 0 or int(ids[-1]) != 0:
                    err("ids: must start and end with BOS/EOS id 0")
            if T > max_phonemes + 2:
                err(f"ids: {T} > max {max_phonemes + 2}")
            if row.get("token_count") != T:
                err("token_count: does not match len(ids)")

    if T and isinstance(row.get("phoneme_count"), int):
        dropped = row["phoneme_count"] - (T - 2)
        if dropped < 0:
            err("phoneme_count: fewer phonemes than token slots")
        if row.get("dropped_phonemes") != dropped:
            err(f"dropped_phonemes: expected {dropped}, got {row.get('dropped_phonemes')!r}")
        stats["dropped_phonemes"] = max(0, dropped)

    if isinstance(row.get("phoneme_count"), int) and isinstance(row.get("style_index"), int):
        want = min(row["phoneme_count"] - 1, STYLE_INDEX_MAX)
        if not (0 <= row["style_index"] <= STYLE_INDEX_MAX):
            err("style_index: outside 0..509")
        elif row["style_index"] != want:
            warn(f"style_index: {row['style_index']} != min(phoneme_count-1, 509) = {want}")

    def check_float_tensor(name, length=None, shape=None):
        value = row.get(name)
        if not torch.is_tensor(value):
            err(f"{name}: expected a tensor")
            return None
        if value.dtype not in (torch.float16, torch.float32, torch.float64):
            err(f"{name}: expected a float dtype, got {value.dtype}")
        if length is not None and value.ndim == 1 and value.numel() != length:
            err(f"{name}: expected length {length}, got {value.numel()}")
        if shape is not None and tuple(value.shape) != shape:
            err(f"{name}: expected shape {shape}, got {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            err(f"{name}: contains NaN or Inf")
        return value

    style = check_float_tensor("style", length=STYLE_DIM)
    if style is not None and style.ndim != 1:
        err(f"style: expected 1-D, got {tuple(style.shape)}")

    pred_dur = row.get("pred_dur")
    F = 0
    if not torch.is_tensor(pred_dur):
        err("pred_dur: expected a tensor")
    else:
        if pred_dur.ndim != 1:
            err(f"pred_dur: expected 1-D, got {tuple(pred_dur.shape)}")
        if pred_dur.dtype not in (torch.int16, torch.int32, torch.int64):
            err(f"pred_dur: expected an integer dtype, got {pred_dur.dtype}")
        if T and pred_dur.numel() != T:
            err(f"pred_dur: expected length {T}, got {pred_dur.numel()}")
        if pred_dur.numel() and int(pred_dur.min()) < 1:
            err("pred_dur: every frame count must be >= 1")
        F = int(pred_dur.sum()) if pred_dur.numel() else 0
        stats["frames"] = F
        if row.get("frames") != F:
            err(f"frames: expected {F}, got {row.get('frames')!r}")

    if T:
        dur = check_float_tensor("dur", length=T)
        if dur is not None and dur.numel() and float(dur.min()) <= 0:
            err("dur: values must be > 0")
        check_float_tensor("t_en", shape=(FEATURE_DIM, T))
        check_float_tensor("d", shape=(T, PRED_FEATURE_DIM))

    if F:
        check_float_tensor("F0", length=2 * F)
        check_float_tensor("N", length=2 * F)

    audio = row.get("audio")
    if not torch.is_tensor(audio):
        err("audio: expected a tensor")
    else:
        if audio.ndim != 1:
            err(f"audio: expected 1-D, got {tuple(audio.shape)}")
        if audio.dtype != torch.int16:
            err(f"audio: expected int16, got {audio.dtype}")
        if audio.numel() == 0:
            err("audio: empty")
        else:
            stats["samples"] = int(audio.numel())
            stats["audio_seconds"] = audio.numel() / SAMPLE_RATE
            a32 = audio.to(torch.int32)
            stats["peak"] = int(a32.abs().max())
            stats["clipped"] = int((a32.abs() >= 32767).sum())
            stats["dc_offset"] = float(audio.to(torch.float32).mean())
            stats["silence"] = float((audio == 0).to(torch.float32).mean())
            if F:
                delta = int(audio.numel()) - SAMPLES_PER_FRAME * F
                stats["audio_length_delta"] = delta
                if abs(delta) > SAMPLES_PER_FRAME:
                    err(f"audio: length {audio.numel()} differs from 600*{F}={SAMPLES_PER_FRAME * F} by {delta}")
                elif abs(delta) > 16:
                    warn(f"audio: length differs from 600*frames by {delta} samples")
            want = audio.numel() / SAMPLE_RATE
            if isinstance(row.get("audio_seconds"), (int, float)) and abs(row["audio_seconds"] - want) > 1e-3:
                err("audio_seconds: does not match len(audio)/24000")
            if isinstance(row.get("duration_seconds"), (int, float)) and abs(row["duration_seconds"] - want) > 1e-3:
                err("duration_seconds: does not match len(audio)/24000")

    return errs, warns, stats


def schema_json():
    """JSON-serialisable description written next to the shards."""
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "One row per phoneme-aligned Kokoro teacher forward pass.",
        "dtypes": {
            "ids": "int64",
            "token_ids": "int64 (same tensor as ids)",
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
            "T": "len(ids) = len(pred_dur) = len(dur) = phoneme_count + 2 - dropped_phonemes",
            "F": "sum(pred_dur); len(F0) = len(N) = 2F; len(audio) = 600F",
            "t_en": "[512, T]",
            "d": "[T, 640]",
            "style": "[256]",
        },
        "row": ROW_FIELDS,
        "shard": {
            "file": "shard_XXXX.pt",
            "contents": "list[dict], one dict per segment",
            "sidecar": "shard_XXXX.meta.json with row count, ids and sha256",
        },
        "invariants": [
            "ids[0] == ids[-1] == 0",
            "every pred_dur >= 1",
            "len(dur) == len(ids)",
            "t_en.shape == d.shape == [len(ids), 512]",
            "len(F0) == len(N) == 2 * sum(pred_dur)",
            "abs(len(audio) - 600 * sum(pred_dur)) <= 600",
            "no NaN or Inf in any float feature",
            "text_hash == sha256(text)",
            "every id is unique across the whole cache",
            "every row has train / val / test split",
        ],
        "splits": list(SPLITS),
    }
