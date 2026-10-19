"""Score released-style ONNX files (input_ids, speed -> waveform) on the held-out sentences: UTMOS, Whisper WER and
speed on one onnxruntime thread. usage: onnx_eval.py N_SENT NAME=PATH.onnx [NAME=PATH.onnx ...]"""
import sys, glob, os, re, time, warnings, torch, numpy as np, soundfile as sf, onnxruntime as ort; warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
import whisper, jiwer, torchaudio
from utmos import score
N = int(sys.argv[1]); SR = 24000; held = torch.load(sorted(glob.glob("data/teacher/shard_*.pt"))[0])[:N]
wm = whisper.load_model("base"); norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1); out = f"out/onnx_eval_{name}"; os.makedirs(out, exist_ok=True)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); tot = secs = 0.0; us, hyp, ref = [], [], []
    for i, r in enumerate(held):
        feed = {"input_ids": r["ids"][None].numpy(), "speed": np.ones(1, np.float32)}
        if i == 0: sess.run(None, feed)
        t = time.perf_counter(); y = sess.run(None, feed)[0][0]; tot += time.perf_counter() - t; secs += len(y) / SR
        f = f"{out}/held_{i + 1}.wav"; sf.write(f, y, SR); us.append(score(f))
        x = torchaudio.functional.resample(torch.tensor(y), SR, 16000).numpy()
        hyp.append(norm(wm.transcribe(x, language="en", fp16=False)["text"])); ref.append(norm(r["text"]))
    print(f"{name:10s} {os.path.getsize(path) / 1e6:5.2f} MB  UTMOS {np.mean(us):.3f} +-{1.96 * np.std(us, ddof=1) / np.sqrt(N):.3f}  WER {jiwer.wer(ref, hyp) * 100:.1f}%  {secs / tot:.1f}x realtime (onnxruntime, 1 thread)", flush=True)
