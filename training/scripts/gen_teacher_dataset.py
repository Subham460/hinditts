"""Generate the reusable Kokoro teacher cache, one row per prepared Hindi segment.

This is the production generator.  It reads ``data/hindi_segments.jsonl`` (written by
``prepare_hindi.py``), runs the frozen Kokoro teacher once per segment, and writes
``shard_XXXX.pt`` files containing the waveform and every intermediate value a student
needs: phonemes/token ids, style, durations, t_en, d, F0, N and metadata.

It is resumable and shard-range aware, so two Kaggle T4s can generate disjoint halves:

  python scripts/gen_teacher_dataset.py --start 0 --stride 2
  python scripts/gen_teacher_dataset.py --start 1 --stride 2

Each shard is written atomically.  A sidecar ``shard_XXXX.meta.json`` records the row
range, ids, audio seconds and sha256; ``validate_teacher_dataset.py`` uses it to build
meta.json / splits.json / validate.json.

usage:
  gen_teacher_dataset.py [--manifest PATH] [--out DIR] [--limit N]
                         [--start K] [--end K] [--stride N] [--shard-size N]
                         [--device cuda:0] [--teacher-revision REV] [--overwrite]
                         [--list] [--dry-run]
"""
import argparse, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time, warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")

import torch
from kokoro import KPipeline, KModel

from lang import (
    LANG_CODE, VOICE, MANIFEST_PATH, TEACHER_DIR, MAX_PHONEMES, STYLE_INDEX_MAX,
    SAMPLE_RATE, FRAME_HOP, FEATURE_DIM, STYLE_DIM, N_TOKEN, TEACHER_REPO,
    SCHEMA_VERSION, DEVICE, SOURCE_DATASET, SAMPLES_PER_FRAME,
)
from teacher_schema import check_row, text_hash


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    tmp = path + f".tmp.{os.getpid()}"
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
            for key in ("id", "parent_id", "text", "phonemes", "split"):
                if key not in row:
                    raise SystemExit(f"{path}:{lineno}: missing {key!r}")
            if row["split"] not in ("train", "val", "test"):
                raise SystemExit(f"{path}:{lineno}: bad split {row['split']!r}")
            if row.get("phoneme_count", len(row["phonemes"])) != len(row["phonemes"]):
                raise SystemExit(f"{path}:{lineno}: phoneme_count mismatch")
            if len(row["phonemes"]) > MAX_PHONEMES:
                raise SystemExit(f"{path}:{lineno}: {len(row['phonemes'])} phonemes > {MAX_PHONEMES}")
            if row.get("text_hash") and row["text_hash"] != text_hash(row["text"]):
                raise SystemExit(f"{path}:{lineno}: text_hash mismatch")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: duplicate segment ids")
    return rows


def resolve_revision(repo_id, revision):
    if revision:
        return revision
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id)
        return info.sha or ""
    except Exception as e:
        print(f"warning: could not resolve teacher revision: {e}", flush=True)
        return ""


def load_teacher(device, repo_id, revision, voice):
    """Load Kokoro, pinned to a revision when one is supplied."""
    if revision:
        from huggingface_hub import hf_hub_download
        config = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        weight_name = getattr(KModel, "MODEL_NAMES", {}).get(repo_id, "kokoro-v1_0.pth")
        weights = hf_hub_download(repo_id=repo_id, filename=weight_name, revision=revision)
        model = KModel(repo_id=repo_id, config=config, model=weights).to(device).eval()
        pipe = KPipeline(lang_code=LANG_CODE, repo_id=repo_id, model=model)
        voice_path = hf_hub_download(repo_id=repo_id, filename=f"voices/{voice}.pt", revision=revision)
        pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    else:
        pipe = KPipeline(lang_code=LANG_CODE, device=device, repo_id=repo_id)
        model = pipe.model.eval()
        pack = pipe.load_voice(voice)
    return pipe, model, pack


def pkg_version(name):
    try:
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


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_row(rec, *, model, pack, pipe, device, voice, teacher_model, teacher_revision,
             seed=0):
    """Run one teacher forward and return a schema-complete row.

    Kokoro's harmonic source uses a random initial phase and additive noise, so the
    waveform is not reproducible between forwards.  Seeding per segment id makes each row
    stable across reruns and shard orders without making different rows identical.
    """
    row_seed = int(hashlib.sha256(f"{seed}:{rec['id']}".encode("utf-8")).hexdigest()[:8], 16)
    torch.manual_seed(row_seed)
    ps = rec.get("phonemes")
    if not ps:
        ps, _ = pipe.g2p(rec["text"])
    if not ps:
        raise RuntimeError(f"{rec['id']}: empty phonemes")
    if len(ps) > MAX_PHONEMES:
        raise RuntimeError(f"{rec['id']}: {len(ps)} phonemes > {MAX_PHONEMES}")

    kept = [c for c in ps if c in model.vocab]
    dropped = len(ps) - len(kept)
    if not kept:
        raise RuntimeError(f"{rec['id']}: no phoneme in Kokoro vocabulary")
    token = torch.LongTensor([[0, *[model.vocab[c] for c in kept], 0]]).to(device)
    T = int(token.shape[1])
    if T > MAX_PHONEMES + 2:
        raise RuntimeError(f"{rec['id']}: {T} tokens > {MAX_PHONEMES + 2}")

    # Kokoro indexes the voice pack by the *raw* phoneme count (KPipeline.infer uses
    # pack[len(ps)-1]), so keep that even if a rare character was dropped.
    style_index = min(len(ps) - 1, STYLE_INDEX_MAX)
    style = pack[style_index]
    if style.dim() == 1:
        style = style[None]
    style = style.to(device)

    length = torch.full((1,), T, device=device)
    mask = torch.zeros(1, T, dtype=torch.bool, device=device)
    bert_dur = model.bert(token, attention_mask=(~mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = style[:, 128:]
    d = model.predictor.text_encoder(d_en, s, length, mask)
    x, _ = model.predictor.lstm(d)
    dur = torch.sigmoid(model.predictor.duration_proj(x)).sum(-1)
    pred_dur = torch.round(dur).clamp(min=1).long().squeeze(0)

    idx = torch.repeat_interleave(torch.arange(T, device=device), pred_dur)
    aln = torch.zeros(T, idx.shape[0], device=device)
    aln[idx, torch.arange(idx.shape[0], device=device)] = 1
    aln = aln[None]
    en = d.transpose(-1, -2) @ aln
    F0, N = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(token, length, mask)
    asr = t_en @ aln
    audio = model.decoder(asr, F0, N, style[:, :128]).squeeze()

    ids = token.squeeze(0).detach().cpu().long()
    pred_dur = pred_dur.detach().cpu().long()
    dur = dur.squeeze(0).detach().cpu().float()
    d_out = d.squeeze(0).detach().cpu().half()                   # [T, 640], token-major
    t_en = t_en.squeeze(0).detach().cpu().half()                 # [512, T], feature-major
    F0 = F0.squeeze(0).detach().cpu().float()
    N = N.squeeze(0).detach().cpu().float()
    style = style.squeeze(0).detach().cpu().half()
    wav = (audio.detach().cpu().clamp(-1, 1) * 32767).short().reshape(-1)

    frames = int(pred_dur.sum())
    # Kokoro's iSTFT can return a few samples less (or more) than 600*frames.  Make the
    # cache exact so every trainer can slice by 600*frame without a length check.
    want = SAMPLES_PER_FRAME * frames
    if wav.numel() > want:
        wav = wav[:want]
    elif wav.numel() < want:
        wav = torch.nn.functional.pad(wav, (0, want - wav.numel()))
    audio_seconds = wav.numel() / SAMPLE_RATE
    text = rec["text"]
    source_row_index = rec.get("source_row_index", 0)
    source_row_index = int(source_row_index) if source_row_index is not None else 0
    segment_index = rec.get("segment_index", 0)
    segment_index = int(segment_index) if segment_index is not None else 0
    n_segments = rec.get("n_segments", 1)
    n_segments = int(n_segments) if n_segments is not None else 1
    row = {
        "id": rec["id"],
        "parent_id": rec["parent_id"],
        "segment_index": segment_index,
        "n_segments": n_segments,
        "source_row_index": source_row_index,
        "source_row_id": str(rec.get("source_row_id", rec["parent_id"])),
        "source_index_kind": str(rec.get("source_index_kind", "input")),
        "source_dataset": str(rec.get("source_dataset") or SOURCE_DATASET),
        "text": text,
        "text_hash": rec.get("text_hash") or text_hash(text),
        "phonemes": ps,
        "phoneme_count": len(ps),
        "ps": ps,
        "language": "hi",
        "voice": voice,
        "split": rec["split"],
        "style_index": style_index,
        "style": style,
        "sample_rate": SAMPLE_RATE,
        "frame_hop": FRAME_HOP,
        "schema_version": SCHEMA_VERSION,
        "teacher_model": teacher_model,
        "teacher_revision": teacher_revision,
        "audio_device": str(device),
        "audio_seed": row_seed,
        "ids": ids,
        "token_ids": ids,
        "token_count": T,
        "pred_dur": pred_dur,
        "frames": frames,
        "dur": dur,
        "t_en": t_en,
        "d": d_out,
        "F0": F0,
        "N": N,
        "audio": wav,
        "duration_seconds": audio_seconds,
        "audio_seconds": audio_seconds,
        "dropped_phonemes": dropped,
    }
    return row


def shard_paths(out_dir, index):
    base = os.path.join(out_dir, f"shard_{index:04d}")
    return base + ".pt", base + ".meta.json"


def load_sidecar(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def existing_shard_state(path, sidecar_path, index, row_start, row_end, manifest_sha,
                         teacher_revision, manifest, seed):
    """Return (complete, legacy).  A complete shard is left alone; a legacy one is an error."""
    if not os.path.exists(path):
        return False, False
    if os.path.exists(sidecar_path):
        side = load_sidecar(sidecar_path)
        if (
            side
            and side.get("schema_version") == SCHEMA_VERSION
            and side.get("shard_index") == index
            and side.get("row_start") == row_start
            and side.get("row_end") == row_end
            and side.get("rows") == row_end - row_start
            and side.get("manifest_sha256") == manifest_sha
            and side.get("teacher_revision") == teacher_revision
            and side.get("seed") == seed
        ):
            return True, False
    # No usable sidecar: inspect the shard itself.  This is also where we detect the old
    # gen_teacher.py format (no schema_version).
    try:
        rows = torch.load(path, map_location="cpu", mmap=True)
    except Exception:
        return False, False
    if not rows:
        return False, False
    if "schema_version" not in rows[0]:
        return False, True
    if len(rows) != row_end - row_start:
        return False, False
    if rows[0].get("schema_version") != SCHEMA_VERSION:
        return False, True
    if rows[0].get("id") != manifest[row_start]["id"] or rows[-1].get("id") != manifest[row_end - 1]["id"]:
        return False, False
    if rows[0].get("teacher_revision") != teacher_revision:
        return False, False
    return True, False


def make_sidecar(path, index, row_start, row_end, rows, manifest_sha, teacher_model,
                 teacher_revision, voice, device, seed):
    ids = [r["id"] for r in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "shard_index": index,
        "file": os.path.basename(path),
        "rows": len(rows),
        "row_start": row_start,
        "row_end": row_end,
        "first_id": ids[0] if ids else "",
        "last_id": ids[-1] if ids else "",
        "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "audio_seconds": round(sum(r["audio_seconds"] for r in rows), 6),
        "frames": int(sum(r["frames"] for r in rows)),
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
        "manifest_sha256": manifest_sha,
        "teacher_model": teacher_model,
        "teacher_revision": teacher_revision,
        "voice": voice,
        "device": str(device),
        "seed": seed,
        "created_utc": now(),
    }


def write_shard(path, rows):
    tmp = path + f".tmp.{os.getpid()}"
    try:
        torch.save(rows, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--out", default=TEACHER_DIR)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--start", type=int, default=0, help="first shard index")
    ap.add_argument("--end", type=int, default=None, help="exclusive shard index")
    ap.add_argument("--stride", type=int, default=1, help="shard step; use 2 on a two-GPU node")
    ap.add_argument("--limit", type=int, default=None, help="only the first N manifest rows")
    ap.add_argument("--device", default=os.environ.get("DEVICE", DEVICE))
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--teacher-repo", default=TEACHER_REPO)
    ap.add_argument("--teacher-revision", default="", help="pin a Kokoro commit; empty resolves HEAD")
    ap.add_argument("--seed", type=int, default=0, help="per-row audio seed; see make_row")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the shard plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="validate the manifest and print the plan")
    a = ap.parse_args()

    if a.shard_size <= 0 or a.stride <= 0:
        raise SystemExit("--shard-size and --stride must be positive")
    if not os.path.isfile(a.manifest):
        raise SystemExit(f"manifest not found: {a.manifest}\nRun prepare_hindi.py first.")

    manifest = read_manifest(a.manifest)
    if a.limit is not None:
        manifest = manifest[:a.limit]
    if not manifest:
        raise SystemExit("manifest is empty")

    total_shards = (len(manifest) + a.shard_size - 1) // a.shard_size
    end = total_shards if a.end is None else min(a.end, total_shards)
    indices = list(range(a.start, end, a.stride))
    manifest_sha = sha256_file(a.manifest)

    print(f"manifest    {a.manifest}")
    print(f"rows        {len(manifest)}  shards {total_shards} x {a.shard_size}")
    print(f"plan        {indices}")
    if a.list or a.dry_run:
        for i in indices:
            s, e = i * a.shard_size, min((i + 1) * a.shard_size, len(manifest))
            print(f"  shard_{i:04d}.pt  rows {s}:{e}  {e - s} rows  split "
                  f"{ {r['split'] for r in manifest[s:e]} }")
        return
    if not indices:
        print("nothing to do")
        return

    device = a.device
    if isinstance(device, str) and device.startswith("cuda") and ":" in device:
        torch.cuda.set_device(int(device.split(":")[1]))
    teacher_revision = resolve_revision(a.teacher_repo, a.teacher_revision)
    os.makedirs(a.out, exist_ok=True)

    print(f"device      {device}")
    print(f"teacher     {a.teacher_repo} @ {teacher_revision or '(unresolved)'}")
    print(f"voice       {a.voice}")
    print(f"torch       {torch.__version__}  cuda {torch.version.cuda}", flush=True)

    pipe, model, pack = load_teacher(device, a.teacher_repo, teacher_revision, a.voice)
    model.eval()

    env = {
        "created_utc": now(),
        "manifest": a.manifest,
        "manifest_sha256": manifest_sha,
        "out": a.out,
        "shard_size": a.shard_size,
        "start": a.start,
        "end": end,
        "stride": a.stride,
        "device": str(device),
        "teacher_model": a.teacher_repo,
        "teacher_revision": teacher_revision,
        "voice": a.voice,
        "seed": a.seed,
        "language": "hi",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "kokoro": pkg_version("kokoro"),
        "misaki": pkg_version("misaki"),
        "transformers": pkg_version("transformers"),
        "espeak": espeak_version(),
        "schema_version": SCHEMA_VERSION,
    }
    write_json_atomic(
        os.path.join(a.out, f"generation_start{a.start}_stride{a.stride}.json"), env
    )

    t0 = time.time()
    done_rows = 0
    done_secs = 0.0
    done_shards = 0
    with torch.inference_mode():
        for i in indices:
            start = i * a.shard_size
            stop = min((i + 1) * a.shard_size, len(manifest))
            path, sidecar_path = shard_paths(a.out, i)
            complete, legacy = existing_shard_state(
                path, sidecar_path, i, start, stop, manifest_sha, teacher_revision,
                manifest, a.seed,
            )
            if complete and not a.overwrite:
                print(f"shard {i:04d}: already complete, skipping", flush=True)
                continue
            if legacy and not a.overwrite:
                raise SystemExit(
                    f"{path} is a legacy shard without schema_version. "
                    "Move it aside or pass --overwrite."
                )

            rows, shard_t0, shard_secs = [], time.time(), 0.0
            for j in range(start, stop):
                rec = manifest[j]
                row = make_row(
                    rec, model=model, pack=pack, pipe=pipe, device=device, voice=a.voice,
                    teacher_model=a.teacher_repo, teacher_revision=teacher_revision,
                    seed=a.seed,
                )
                errors, warns, stats = check_row(
                    row, expected_language="hi", expected_voice=a.voice
                )
                if errors:
                    raise RuntimeError(f"{rec['id']}: " + "; ".join(errors[:6]))
                for w in warns:
                    print(f"warning {rec['id']}: {w}", flush=True)
                rows.append(row)
                shard_secs += row["audio_seconds"]
                if (j + 1) % 50 == 0:
                    elapsed = max(time.time() - shard_t0, 1e-9)
                    print(f"\rshard {i:04d}: {j + 1 - start}/{stop - start} "
                          f"{shard_secs / 3600:.2f} h  {shard_secs / elapsed:.1f}x RT", end="", flush=True)

            write_shard(path, rows)
            side = make_sidecar(
                path, i, start, stop, rows, manifest_sha, a.teacher_repo,
                teacher_revision, a.voice, device, a.seed,
            )
            write_json_atomic(sidecar_path, side)
            done_rows += len(rows)
            done_secs += shard_secs
            done_shards += 1
            elapsed = max(time.time() - shard_t0, 1e-9)
            print(f"\rshard {i:04d}: {len(rows)} rows  {shard_secs / 3600:.2f} h  "
                  f"{shard_secs / elapsed:.1f}x RT  {os.path.getsize(path) / 1e6:.0f} MB  "
                  f"sha {side['sha256'][:12]}", flush=True)

    elapsed = max(time.time() - t0, 1e-9)
    print(f"done: {done_shards} new shards, {done_rows} rows, {done_secs / 3600:.2f} h audio, "
          f"{elapsed / 60:.1f} min, {done_secs / elapsed:.1f}x RT")
    print(f"next: python scripts/validate_teacher_dataset.py --root {a.out} "
          f"--manifest {a.manifest}")


if __name__ == "__main__":
    main()
