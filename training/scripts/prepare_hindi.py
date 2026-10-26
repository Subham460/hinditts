"""Split the 12,000 raw Hindi sentences into <=400-phoneme training segments.

Kokoro's text side has 512 positions and `gen_teacher.py` skips anything above 400
phonemes.  On the raw corpus that would silently throw away about 14% of the text and
leave the longest 6% untrainable.  Sentences that fit are left exactly as they are; longer
ones are split into `ceil(total / 400)` roughly equal parts at word boundaries, so a
410-phoneme sentence becomes two 205-phoneme halves instead of a 400/10 pair.

Each raw sentence is processed on its own, so a segment never mixes two dataset rows and
the held-out split stays clean.

usage: prepare_hindi.py [--in data/hindi_sents.json] [--out data/hindi_sents_segments.json]
"""
import argparse, json, math, re, statistics, sys
sys.path.insert(0, "scripts")
from misaki import espeak
from lang import RAW_SENTS_PATH, SENTS_PATH, MAX_PHONEMES, G2P_LANG

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


def main():
    global MAX_PHONEMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=RAW_SENTS_PATH)
    ap.add_argument("--out", default=SENTS_PATH)
    ap.add_argument("--max", type=int, default=MAX_PHONEMES)
    a = ap.parse_args()
    MAX_PHONEMES = a.max

    g2p = espeak.EspeakG2P(language=G2P_LANG)
    raw = json.load(open(a.inp, encoding="utf-8"))
    # Brackets are editorial marks in the PIB source text; Kokoro's vocabulary has no
    # tokens for them, so they would be dropped anyway.  Collapse all whitespace.
    raw = [WS.sub(" ", t.replace("[", "").replace("]", "")).strip() for t in raw]
    raw = [t for t in raw if t]

    segs, seen = [], set()
    for i, t in enumerate(raw, 1):
        for s in split_text(t, g2p):
            if s not in seen:
                seen.add(s)
                segs.append(s)
        if i % 1000 == 0:
            print(f"\r{i}/{len(raw)} texts -> {len(segs)} segments", end="", flush=True)
    print()

    lens = [len(g2p(s)[0]) for s in segs]
    q = statistics.quantiles(lens, n=100)
    too_long = sum(n > MAX_PHONEMES for n in lens)
    print(f"raw {len(raw)} texts -> {len(segs)} unique segments "
          f"({len(segs) / len(raw):.2f} per text, {len(segs) - len(raw):+d} vs raw)")
    print(f"phonemes: min {min(lens)}  p50 {q[49]:.0f}  p95 {q[94]:.0f}  max {max(lens)}  "
          f">{MAX_PHONEMES} {too_long}")
    assert not too_long, "segment above the limit"
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(segs, f, ensure_ascii=False, indent=0)
        f.write("\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
