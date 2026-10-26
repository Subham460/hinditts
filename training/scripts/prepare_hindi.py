"""Build an id-preserving, <=400-phoneme segment manifest from the Hindi source corpus.

The old ``prepare_hindi.py`` wrote a text-only list, so segments could not be traced back
to the 12,000 source rows and there was no stable train/val/test split.  This version keeps
``id``, ``parent_id``, ``source_row_index`` and ``segment_index``, computes the espeak-ng
phonemes once, and assigns a deterministic parent-level split.  It still writes the legacy
text-only list so older scripts keep working.

Input can be either:

* ``selected_texts.jsonl`` from the Kaggle selection (one ``{"id", "text", ...}`` per line),
* the JSONL written by ``fetch_hindi_sents.py``,
* the legacy ``hindi_sents.json`` list of strings.

Output:

* ``data/hindi_segments.jsonl``  canonical manifest, one segment per line
* ``data/hindi_sents_segments.json``  legacy list of segment texts
* ``data/hindi_segments_report.json`` counts, hashes and duplicate report

usage:
  prepare_hindi.py
  prepare_hindi.py --in /kaggle/working/kokoro_distill_selection/selected_texts.jsonl
  prepare_hindi.py --limit 2000 --out /tmp/segments.jsonl
"""
import argparse, hashlib, json, math, os, random, re, statistics, sys
sys.path.insert(0, "scripts")
from misaki import espeak
from lang import (
    RAW_SENTS_PATH, SENTS_PATH, MANIFEST_PATH, MAX_PHONEMES, G2P_LANG,
    LANG_CODE, VOICE, SOURCE_DATASET, SPLIT_SEED, VAL_BPS, TEST_BPS,
)

WS = re.compile(r"\s+")


def char_split(text, g2p):
    """Last resort for a single token that is itself over the limit."""
    out, cur = [], ""
    for ch in text:
        if cur and len(g2p(cur + ch)[0]) > MAX_PHONEMES:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def split_text(text, g2p):
    total = len(g2p(text)[0])
    if total <= MAX_PHONEMES:
        return [text]

    n = math.ceil(total / MAX_PHONEMES)
    target = total / n
    words = text.split()
    pieces, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur:
            cand_len = len(g2p(cand)[0])
            # Cut near the equal-share target, but never put a piece over the hard limit.
            if cand_len > MAX_PHONEMES or (len(pieces) < n - 1 and cand_len > target):
                pieces.append(cur)
                cur = w
                continue
        cur = cand
    if cur:
        pieces.append(cur)

    out = []
    for p in pieces:
        if len(g2p(p)[0]) <= MAX_PHONEMES:
            out.append(p)
        else:
            out += char_split(p, g2p)
    return out


def normalise(text):
    # Brackets are editorial marks in the PIB source text; Kokoro's vocabulary has no
    # tokens for them, so they would be dropped anyway.  Collapse all whitespace.
    return WS.sub(" ", str(text).replace("[", "").replace("]", "")).strip()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path):
    """Yield dicts with at least a ``text`` key, preserving ids when they exist."""
    if not os.path.isfile(path):
        raise SystemExit(f"input not found: {path}")

    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{i + 1}: {e}") from e
                if isinstance(row, str):
                    row = {"text": row}
                if not isinstance(row, dict):
                    raise SystemExit(f"{path}:{i + 1}: expected an object")
                yield i, row
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("rows", "data", "records"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise SystemExit(f"{path}: expected a list or a dict with rows/data/records")
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a list")
    for i, row in enumerate(data):
        if isinstance(row, str):
            row = {"text": row}
        if not isinstance(row, dict):
            raise SystemExit(f"{path}: row {i}: expected a string or an object")
        yield i, row


def parent_identity(row, index):
    for key in ("id", "parent_id", "source_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value), str(value)
    text = normalise(row.get("text", ""))
    return "text-" + digest(text)[:16], ""


def split_for(parent_id, val_bps, test_bps, seed):
    bucket = int.from_bytes(
        hashlib.sha256(f"{seed}:{parent_id}".encode("utf-8")).digest()[:4], "big"
    ) % 10000
    if bucket < val_bps:
        return "val", bucket
    if bucket < val_bps + test_bps:
        return "test", bucket
    return "train", bucket


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main():
    global MAX_PHONEMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=RAW_SENTS_PATH,
                    help="JSONL with id/text, or the legacy JSON list of strings")
    ap.add_argument("--out", default=MANIFEST_PATH, help="manifest JSONL")
    ap.add_argument("--texts-out", default=SENTS_PATH, help="legacy text-only JSON list")
    ap.add_argument("--report", default="data/hindi_segments_report.json")
    ap.add_argument("--max", type=int, default=MAX_PHONEMES)
    ap.add_argument("--limit", type=int, default=None, help="only the first N input records")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--val-bps", type=int, default=VAL_BPS, help="1/10000 of parents -> val")
    ap.add_argument("--test-bps", type=int, default=TEST_BPS, help="1/10000 of parents -> test")
    ap.add_argument("--no-shuffle-train", action="store_true")
    a = ap.parse_args()
    MAX_PHONEMES = a.max

    g2p = espeak.EspeakG2P(language=G2P_LANG)
    records = list(load_records(a.inp))
    if a.limit is not None:
        records = records[:a.limit]
    if not records:
        raise SystemExit("no input records")

    rows = []
    seen_text = {}
    duplicates = []
    parent_ids = set()
    skipped_empty = 0
    input_has_ids = 0

    for index, rec in records:
        text = normalise(rec.get("text", ""))
        if not text:
            skipped_empty += 1
            continue

        pid, original_id = parent_identity(rec, index)
        if original_id:
            input_has_ids += 1
        if pid in parent_ids:
            raise SystemExit(f"duplicate parent id in input: {pid!r}")
        parent_ids.add(pid)

        source_row_index = rec.get("source_row_index", index)
        try:
            source_row_index = int(source_row_index)
        except (TypeError, ValueError):
            source_row_index = index
        index_kind = "dataset" if ("source_row_index" in rec and original_id) else "input"
        split, bucket = split_for(pid, a.val_bps, a.test_bps, a.seed)

        pieces = split_text(text, g2p)
        n_segments = len(pieces)
        for seg_index, piece in enumerate(pieces):
            ps, _ = g2p(piece)
            if not ps:
                skipped_empty += 1
                continue
            if len(ps) > MAX_PHONEMES:
                raise SystemExit(f"{pid}: segment {seg_index} has {len(ps)} phonemes > {MAX_PHONEMES}")
            h = digest(piece)
            sid = f"{pid}__s{seg_index:03d}"
            duplicate_of = seen_text.get(piece)
            if duplicate_of:
                # Keep it anyway.  Dropping it would silently remove a segment of its
                # parent row (one source row has a segment identical to another row's).
                # The id stays unique and the teacher value is identical, so the cost is
                # one extra forward pass, not a hole in the corpus.
                duplicates.append({"text_hash": h, "kept_id": duplicate_of, "duplicate_id": sid})
            else:
                seen_text[piece] = sid
            rows.append({
                "id": sid,
                "parent_id": pid,
                "segment_index": seg_index,
                "n_segments": n_segments,
                "source_row_index": source_row_index,
                "source_row_id": original_id or pid,
                "source_index_kind": index_kind,
                "source_dataset": SOURCE_DATASET,
                "text": piece,
                "text_hash": h,
                "phonemes": ps,
                "phoneme_count": len(ps),
                "language": "hi",
                "voice": rec.get("voice") or VOICE,
                "split": split,
                "split_bucket": bucket,
                "language_code": LANG_CODE,
                "duplicate_of": duplicate_of or "",
            })

        if (index + 1) % 1000 == 0:
            print(f"\r{index + 1}/{len(records)} records -> {len(rows)} segments", end="", flush=True)
    print()

    if not rows:
        raise SystemExit("no segments produced")
    represented = {r["parent_id"] for r in rows}
    if parent_ids - represented:
        raise SystemExit(f"{len(parent_ids - represented)} source rows produced no segment")

    # val and test first: the existing evaluation scripts read the first rows of shard 0.
    order = {"val": 0, "test": 1, "train": 2}
    rows.sort(key=lambda r: (order[r["split"]], r["parent_id"], r["segment_index"]))
    groups = {name: [r for r in rows if r["split"] == name] for name in ("val", "test", "train")}
    if not a.no_shuffle_train:
        random.Random(a.seed).shuffle(groups["train"])
    rows = groups["val"] + groups["test"] + groups["train"]

    lens = [r["phoneme_count"] for r in rows]
    q = statistics.quantiles(lens, n=100) if len(lens) > 1 else [lens[0]] * 99
    counts = {name: len(groups[name]) for name in ("train", "val", "test")}
    counts["total"] = len(rows)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)

    # Keep the legacy text-only list in sync.  It has no ids, so it is for old scripts only.
    write_json_atomic(a.texts_out, [r["text"] for r in rows])

    report = {
        "input": a.inp,
        "input_sha256": sha256_file(a.inp),
        "output": a.out,
        "output_sha256": sha256_file(a.out),
        "texts_output": a.texts_out,
        "texts_output_sha256": sha256_file(a.texts_out),
        "seed": a.seed,
        "val_bps": a.val_bps,
        "test_bps": a.test_bps,
        "source_dataset": SOURCE_DATASET,
        "max_phonemes": MAX_PHONEMES,
        "records": len(records),
        "records_with_ids": input_has_ids,
        "skipped_empty": skipped_empty,
        "segments": len(rows),
        "unique_segments": len(seen_text),
        "duplicates": duplicates,
        "duplicate_count": len(duplicates),
        "counts": counts,
        "phonemes": {
            "min": min(lens),
            "p50": q[49],
            "p95": q[94],
            "max": max(lens),
            "over_max": sum(n > MAX_PHONEMES for n in lens),
        },
        "note": "split is assigned per parent_id so all segments of a source row stay together; "
                "duplicate segment text is kept so no source row loses a segment",
    }
    write_json_atomic(a.report, report)

    print(f"input       {a.inp}")
    print(f"records     {len(records)} ({input_has_ids} with ids)")
    print(f"segments    {len(rows)} ({len(seen_text)} unique texts, "
          f"{len(duplicates)} duplicate segments kept)")
    print(f"splits      train {counts['train']}  val {counts['val']}  test {counts['test']}")
    print(f"phonemes    min {min(lens)}  p50 {q[49]:.0f}  p95 {q[94]:.0f}  max {max(lens)}  "
          f"> {MAX_PHONEMES}: {report['phonemes']['over_max']}")
    print(f"manifest    {a.out}")
    print(f"legacy list {a.texts_out}")
    print(f"report      {a.report}")


if __name__ == "__main__":
    main()
