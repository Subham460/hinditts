"""Paradee: an 8M-parameter text-to-speech model distilled from Kokoro-82M.

This fork targets Hindi with Kokoro's `hf_alpha` voice; English releases still load with
`Paradee()`.
"""
from .tts import Paradee, SAMPLE_RATE

__all__ = ["Paradee", "SAMPLE_RATE"]
__version__ = "1.1.0"
