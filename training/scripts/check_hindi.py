"""Preflight check for the Hindi training corpus before spending hours on teacher generation.

Reports duplicates, script mix, phoneme-length distribution, split counts, sentences that
would exceed MAX_PHONEMES and characters that Kokoro's vocabulary drops.  Reads the
id-preserving manifest when it exists, otherwise the legacy text list.

usage:
  check_hindi.py
  check_hindi.py --in data/hindi_segments.jsonl
  check_hindi.py --in data/hindi_sents_segments.json
"""
import argparse, collections, json, os, statistics, sys, urllib.request
sys.path.insert(0, "scripts")
from misaki import espeak
from lang import SENTS_PATH, MANIFEST_PATH, MAX_PHONEMES, G2P_LANG


def load_rows(path):
    if path.endswith(".jsonl"):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return json.load(open(path, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None,
                    help=f"manifest JSONL or text list (default: {MANIFEST_PATH} if it exists, else {SENTS_PATH})")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()

    path = a.inp or (MANIFEST_PATH if os.path.isfile(MANIFEST_PATH) else SENTS_PATH)
    rows = load_rows(path)
    if a.n is not None:
        rows = rows[:a.n]
    if not rows:
        raise SystemExit(f"no rows in {path}")

    vocab = json.load(urllib.request.urlopen("https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json"))["vocab"]
    g2p = espeak.EspeakG2P(language=G2P_LANG)

    is_manifest = isinstance(rows[0], dict)
    ids = [r.get("id") for r in rows] if is_manifest else []
    texts = [r.get("text") for r in rows] if is_manifest else rows
    splits = collections.Counter(r.get("split", "unknown") for r in rows) if is_manifest else collections.Counter()

    dup_text = len(texts) - len(set(texts))
    dup_id = len(ids) - len(set(ids)) if is_manifest else 0
    lens, dropped, too_long, deva = [], collections.Counter(), 0, []
    for i, (text, row) in enumerate(zip(texts, rows), 1):
        ps = row.get("phonemes") if is_manifest and row.get("phonemes") else g2p(text)[0]
        lens.append(len(ps))
        dropped.update(c for c in ps if c not in vocab)
        too_long += len(ps) > MAX_PHONEMES
        deva.append(sum("\u0900" <= c <= "\u097f" for c in text) / max(1, len(text)))
        if i % 1000 == 0:
            print(f"\r{i}/{len(texts)}", end="", flush=True)
    print()

    q = statistics.quantiles(lens, n=100) if len(lens) > 1 else [lens[0]] * 99
    print(f"file        {path}")
    print(f"rows        {len(texts)}  unique text {len(set(texts))}  duplicates {dup_text}")
    if is_manifest:
        print(f"ids         {len(ids)}  unique {len(set(ids))}  duplicates {dup_id}")
        print(f"parents     {len({r.get('parent_id') for r in rows})}")
        print(f"splits      train {splits['train']}  val {splits['val']}  test {splits['test']}")
    print(f"Devanagari  {100 * statistics.mean(deva):.1f}% of characters")
    print(f"phonemes    min {min(lens)}  p50 {q[49]:.0f}  p95 {q[94]:.0f}  max {max(lens)}  "
          f"> {MAX_PHONEMES}: {too_long} ({100 * too_long / len(texts):.1f}%)")
    print(f"dropped     {sum(dropped.values())} chars over {len(dropped)} types: {dict(dropped)}")
    print("\nfirst 5:")
    for text in texts[:5]:
        print(" ", text)


if __name__ == "__main__":
    main()
