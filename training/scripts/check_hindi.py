"""Preflight check for the Hindi training corpus before spending hours on teacher generation.

Reports duplicates, script mix, phoneme-length distribution, sentences that gen_teacher.py
would skip (over MAX_PHONEMES) and characters that Kokoro's vocabulary drops.

usage: check_hindi.py [--n N]      # default: every segment
"""
import argparse, collections, json, statistics, sys, urllib.request
sys.path.insert(0, "scripts")
from misaki import espeak
from lang import SENTS_PATH, MAX_PHONEMES, G2P_LANG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    texts = json.load(open(SENTS_PATH, encoding="utf-8"))[:a.n]
    vocab = json.load(urllib.request.urlopen("https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json"))["vocab"]
    g2p = espeak.EspeakG2P(language=G2P_LANG)

    dup = len(texts) - len(set(texts))
    lens, dropped, too_long, deva = [], collections.Counter(), 0, []
    for i, t in enumerate(texts, 1):
        ps, _ = g2p(t)
        lens.append(len(ps))
        dropped.update(c for c in ps if c not in vocab)
        too_long += len(ps) > MAX_PHONEMES
        deva.append(sum("\u0900" <= c <= "\u097f" for c in t) / max(1, len(t)))
        if i % 1000 == 0:
            print(f"\r{i}/{len(texts)}", end="", flush=True)
    print()
    q = statistics.quantiles(lens, n=100)
    print(f"texts {len(texts)}  duplicates {dup}  Devanagari chars {100 * statistics.mean(deva):.1f}%")
    print(f"phonemes: min {min(lens)}  p50 {q[49]:.0f}  p95 {q[94]:.0f}  max {max(lens)}  "
          f">{MAX_PHONEMES} {too_long} ({100 * too_long / len(texts):.1f}%)")
    print(f"dropped (not in Kokoro vocab): {sum(dropped.values())} chars over {len(dropped)} types: {dict(dropped)}")
    print("\nfirst 5:")
    for t in texts[:5]:
        print(" ", t)


if __name__ == "__main__":
    main()
