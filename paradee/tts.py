"""Text -> phonemes -> token ids -> one ONNX graph -> 24 kHz audio.

English uses misaki (Kokoro's own G2P). Hindi uses espeak-ng, which is what Kokoro's Hindi
voices and the `subhamgiri9748/kokoros-distill` corpus use.
"""
import json, re
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPOS = {"1.1": "sahilmahendrakar/Paradee-8M-v1.1", "1.0": "sahilmahendrakar/Paradee-8M-v1.0"}
SAMPLE_RATE = 24000
MAX_PHONEMES = 510   # the text side has 512 positions, two of which are the pad tokens at each end


class Paradee:
    """Load once, then call with text to get a float32 waveform at 24 kHz.

    version: "1.1" (default) or "1.0", the earlier English release with the slower decoder.
    quantized: use the 8.5 MB int8 file (default) instead of the 35 MB fp32 one. They sound the same.
    model_path / config_path: local files instead of downloading from the Hugging Face Hub.
    threads: CPU threads for onnxruntime. One thread already runs about 60x faster than real time.
    language: "en" (misaki G2P) or "hi" (espeak-ng, the G2P of the Hindi teacher). Other espeak
        language codes work too, as long as the model was trained on that G2P.
    repo_id / revision: model repository for a non-English model. Required for language="hi"
        unless both model_path and config_path are local.
    """

    def __init__(self, version="1.1", quantized=True, model_path=None, config_path=None, threads=1,
                 language="en", repo_id=None, revision=None):
        if repo_id is None and (model_path is None or config_path is None):
            if language != "en":
                raise ValueError("pass repo_id=... for a non-English model, or give both model_path and config_path")
            repo_id, revision = REPOS[version], revision or f"v{version}"
        name = "onnx/paradee_int8.onnx" if quantized else "onnx/paradee.onnx"
        model_path = model_path or hf_hub_download(repo_id, name, revision=revision)
        config_path = config_path or hf_hub_download(repo_id, "config.json", revision=revision)
        self.vocab = json.load(open(config_path, encoding="utf-8"))["vocab"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        if language == "en":
            from misaki import en, espeak   # the same G2P setup as kokoro.KPipeline(lang_code="a")
            self.g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")
        else:
            from misaki import espeak   # e.g. "hi"; Kokoro's non-English path
            self.g2p = espeak.EspeakG2P(language=language)

    def phonemize(self, text):
        phonemes, _ = self.g2p(text)
        return phonemes

    def generate_from_phonemes(self, phonemes, speed=1.0):
        ids = [0] + [self.vocab[c] for c in phonemes if c in self.vocab][:MAX_PHONEMES] + [0]
        feed = {"input_ids": np.array([ids], dtype=np.int64), "speed": np.array([speed], dtype=np.float32)}
        return self.session.run(None, feed)[0][0]

    def __call__(self, text, speed=1.0):
        """speed > 1 speaks faster. Long text is read one sentence at a time."""
        parts = [self.generate_from_phonemes(ps, speed) for ps in self._chunks(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _chunks(self, text):
        # Latin punctuation plus the Devanagari danda (।) and double danda (॥); a decimal point
        # is safe because Latin punctuation only splits when followed by whitespace.
        for sentence in re.split(r"(?<=[।॥])\s*|(?<=[.!?…])\s+|\n+", text.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            ps = self.phonemize(sentence)
            while len(ps) > MAX_PHONEMES:   # a very long sentence: cut at the last space that fits
                cut = ps.rfind(" ", 0, MAX_PHONEMES)
                cut = cut if cut > 0 else MAX_PHONEMES
                yield ps[:cut]
                ps = ps[cut:].lstrip()
            if ps:
                yield ps
