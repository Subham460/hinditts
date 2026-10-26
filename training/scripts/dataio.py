"""Memory-efficient loading for the teacher shards.

A full Hindi decoder run keeps every waveform in the dataset in RAM: 51 hours at 24 kHz
is about 8.8 GB of int16, more than a small laptop has.  The shards are ordinary
`torch.save` files, so `mmap=True` leaves the tensors on disk and pages in only the slices
the current batch reads.  Older torch versions, or a shard saved in the legacy format,
fall back to a normal load.

The teacher cache also carries a `split` field.  `partition()` and `load_held()` let the
trainers use the dataset's stable train/val/test split instead of the old "first 200 rows
of shard 0" convention.  Shards without the field still work.
"""
import glob
import torch

SHARD_GLOB = "data/teacher/shard_*.pt"
SPLITS = ("train", "val", "test")


def load_shard(path):
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu")


def shard_paths(root="data/teacher", limit=999):
    return sorted(glob.glob(f"{root}/shard_*.pt"))[:limit]


def has_splits(rows):
    return any(isinstance(r, dict) and "split" in r for r in rows)


def partition(rows):
    """Split rows by the dataset's split field, or return None for a legacy cache."""
    if not has_splits(rows):
        return None
    out = {name: [] for name in SPLITS}
    for row in rows:
        split = row.get("split")
        if split in out:
            out[split].append(row)
    return out


def load_held(n=200, root="data/teacher"):
    """Held-out rows for evaluation.

    With a split-aware cache this returns val then test rows.  With a legacy cache it
    returns the first n rows of the first shard, preserving the old behaviour.
    """
    paths = shard_paths(root)
    if not paths:
        raise FileNotFoundError(f"no teacher shards under {root}")
    held = []
    for path in paths:
        rows = load_shard(path)
        for row in rows:
            if row.get("split") in ("val", "test"):
                held.append(row)
                if len(held) >= n:
                    return held
        # A legacy shard has no split field; stop after the first shard if nothing matched.
        if not has_splits(rows):
            return rows[:n]
    if held:
        return held
    return load_shard(paths[0])[:n]
