"""A/B for the frame-rate head: whole Paradee pipeline (student text side) through (a) the released decoder A +
locked33 filter and (b) a vhead checkpoint. Writes WAVs to out/vh_ab{tag}/ and prints UTMOS, harmonicity
(energy at harmonics vs between them, voiced frames, dB) and optionally Whisper WER.
usage: vh_ab.py VH_CKPT [N_SENT] [--tag T] [--wer] [--dim D --blocks B]"""
import sys, glob, os, re, argparse, time, warnings, torch, numpy as np, soundfile as sf; warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
from student import TextStudent, alignment
from student_decoder import load_student_decoder
from vhead import VDecoder, HV
from phase_lock import lock
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("n", type=int, nargs="?", default=8); ap.add_argument("--tag", default=""); ap.add_argument("--wer", action="store_true")
ap.add_argument("--dim", type=int, default=160); ap.add_argument("--blocks", type=int, default=4); A = ap.parse_args()
torch.set_num_threads(1); out = f"out/vh_ab{A.tag}"; os.makedirs(out, exist_ok=True)
held = torch.load(sorted(glob.glob("data/teacher/shard_*.pt"))[0])[:A.n]
ts = TextStudent("s+mlp").eval(); ts.load_state_dict(torch.load("checkpoints/text_s+mlp/last.pt", map_location="cpu")["model"])
da = load_student_decoder("A", "checkpoints/dec_A_gan3/last.pt"); cap = {}
da.generator.m_source.register_forward_hook(lambda m, i, o: cap.__setitem__("src", o[0].squeeze()))
vh = VDecoder(head=HV(A.ckpt), dim=A.dim, blocks=A.blocks).eval(); vh.load_ckpt(A.ckpt)
NF, WIN, HOP, SR = 4096, 1024, 256, 24000; w = torch.hann_window(WIN); fr = np.arange(NF // 2 + 1) * SR / NF; BANDS = [(1000, 2000), (2000, 4000), (4000, 8000)]
def hnr(x, f0):
    X = torch.stft(x, NF, HOP, WIN, window=w, return_complex=True).abs().numpy() ** 2; res = {b: [] for b in BANDS}
    for j in range(X.shape[1]):
        a, f, c = [float(f0[min(int((j + d) * HOP / 300), len(f0) - 1)]) for d in (-2, 0, 2)]
        if min(a, f, c) < 80: continue
        for lo, hi in BANDS:
            ks = np.arange(int(np.ceil(lo / f)), int(hi / f)); P = X[:, j]
            h = sum(P[(fr > k * f - .15 * f) & (fr < k * f + .15 * f)].sum() for k in ks); ih = sum(P[(fr > (k + .5) * f - .15 * f) & (fr < (k + .5) * f + .15 * f)].sum() for k in ks)
            res[(lo, hi)].append(10 * np.log10((h + 1e-12) / (ih + 1e-12)))
    return {b: float(np.mean(v)) for b, v in res.items()}
names = ["teacher", "current", "newhead"]; H = {n: [] for n in names}; T = {"current": 0.0, "newhead": 0.0}; secs = 0
for i, r in enumerate(held):
    with torch.no_grad():
        ids = r["ids"][None]; Ln = torch.tensor([len(r["ids"])])
        pd, _, _, pasr = ts(ids, Ln); aln = alignment(torch.round(pd[0]).clamp(min=1).long())[None]
        _, F0, N, _ = ts(ids, Ln, aln); asr = pasr @ aln
        torch.manual_seed(i); t = time.perf_counter(); s = da(asr, F0, N).squeeze(); cur = lock(s, cap["src"], F0[0], 33); T["current"] += time.perf_counter() - t
        torch.manual_seed(i); t = time.perf_counter(); new = vh(asr, F0, N).squeeze(); T["newhead"] += time.perf_counter() - t
    secs += len(new) / SR; tea = r["audio"].float() / 32767
    for n, x in zip(names, (tea, cur, new)):
        sf.write(f"{out}/held_{i + 1}_{n}.wav", x.numpy(), SR)
        if n != "teacher": H[n].append(hnr(x, F0[0]))
    aln_t = alignment(r["pred_dur"]); H["teacher"].append(hnr(tea, r["F0"][:2 * aln_t.shape[1]]))
from utmos import score
print(f"{A.ckpt}, {A.n} held-out sentences, whole pipeline")
if A.wer:
    import whisper, jiwer, torchaudio; wm = whisper.load_model("base"); norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())
for n in names:
    us = [score(f"{out}/held_{i + 1}_{n}.wav") for i in range(A.n)]; u = np.mean(us)
    line = f"{n:8s} UTMOS {u:.3f} +-{1.96 * np.std(us, ddof=1) / np.sqrt(A.n):.3f}  harmonicity dB " + "  ".join(f"{lo // 1000}-{hi // 1000}k {np.mean([h[(lo, hi)] for h in H[n]]):5.2f}" for lo, hi in BANDS)
    if n in T: line += f"  decoder {secs / T[n]:4.0f}x realtime (torch fp32, 1 thread)"
    if A.wer:
        hyp, ref = [], []
        for i, r in enumerate(held):
            x, _ = sf.read(f"{out}/held_{i + 1}_{n}.wav", dtype="float32"); x = torchaudio.functional.resample(torch.tensor(x), SR, 16000).numpy()
            hyp.append(norm(wm.transcribe(x, language="en", fp16=False)["text"])); ref.append(norm(r["text"]))
        line += f"  WER {jiwer.wer(ref, hyp) * 100:.1f}%"
    print(line, flush=True)
