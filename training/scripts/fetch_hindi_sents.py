"""Write data/hindi_sents.json from the public Kokoro-Hindi distillation dataset.

The dataset (subhamgiri9748/kokoros-distill) has 12,000 Hindi sentences paired with
Kokoro-82M audio from the `hf_alpha` voice.  Paradee only needs the text: the teacher is
re-run locally to save every intermediate.  The order is shuffled with a fixed seed so the
first 200 sentences (the held-out set in every training script) are not all from one source.

usage: fetch_hindi_sents.py [--n 12000] [--out data/hindi_sents.json]
"""
import argparse, json, os, random, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

DATASET = "subhamgiri9748/kokoros-distill"
ROWS = "https://datasets-server.huggingface.co/rows"
PAGE = 100
CACHE = ""


def get(offset):
    """One page, cached on disk so a rerun resumes instead of refetching everything."""
    cache = os.path.join(CACHE, f"{offset:05d}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return [r["row"] for r in json.load(f)["rows"]]
    q = urllib.parse.urlencode(dict(dataset=DATASET, config="default", split="train", offset=offset, length=PAGE))
    for attempt in range(10):
        try:
            time.sleep(0.3)
            with urllib.request.urlopen(f"{ROWS}?{q}", timeout=90) as r:
                data = json.load(r)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return [r["row"] for r in data["rows"]]
        except urllib.error.HTTPError as e:
            if attempt == 9:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or min(60, 10 * (attempt + 1))
            print(f"\noffset {offset}: HTTP {e.code}, retrying in {wait}s", flush=True)
            time.sleep(wait)
        except Exception:
            if attempt == 9:
                raise
            time.sleep(2 ** attempt)


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", default="data/hindi_sents.json")
    ap.add_argument("--workers", type=int, default=1, help="keep at 1: the dataset server rate-limits parallel requests")
    a = ap.parse_args()
    CACHE = os.path.join(os.path.dirname(a.out) or ".", ".hindi_pages")
    os.makedirs(CACHE, exist_ok=True)

    offsets = list(range(0, a.n, PAGE))
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, page in enumerate(ex.map(get, offsets), 1):
            rows += page
            print(f"\r{len(rows)}/{a.n}", end="", flush=True)
    print()
    rows = rows[:a.n]
    assert len(rows) == a.n, (len(rows), a.n)

    texts = [r["text"].strip() for r in rows]
    assert all(texts), "empty text row"
    assert all(r["language"] == "hi" for r in rows), "non-Hindi row"
    voices = sorted({r["voice"] for r in rows})
    hours = sum(r["duration_seconds"] for r in rows) / 3600

    random.Random(0).shuffle(texts)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=0)
        f.write("\n")
    print(f"wrote {len(texts)} texts to {a.out} ({len(set(texts))} unique), "
          f"teacher voices {voices}, {hours:.2f} h of audio")


if __name__ == "__main__":
    main()
