"""Memory-efficient loading for the teacher shards.

A full Hindi decoder run keeps every waveform in the dataset in RAM: 51 hours at 24 kHz
is about 8.8 GB of int16, more than a small laptop has.  The shards are ordinary
`torch.save` files, so `mmap=True` leaves the tensors on disk and pages in only the slices
the current batch reads.  Older torch versions, or a shard saved in the legacy format,
fall back to a normal load.
"""
import torch


def load_shard(path):
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu")
