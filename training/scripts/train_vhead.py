"""Train decoder A's trunk + the frame-rate source-filter head (vhead.py) on teacher intermediates to
reproduce the teacher waveform. Same data, losses and critics as train_decoder.py.
usage: train_vhead.py [--steps N] [--gan --mel-weight 3 --init-from CKPT] [--tag T]   resumable from checkpoints/vh{tag}/last.pt"""
import sys, os, glob, math, time, json, argparse, warnings, random
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F, soundfile as sf, torchaudio
sys.path.insert(0, "scripts")
from vhead import VDecoder
from student import alignment
from lang import DEVICE
from dataio import load_shard
ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--seg", type=int, default=64)
ap.add_argument("--gan", action="store_true"); ap.add_argument("--gan-weight", type=float, default=1.0); ap.add_argument("--fm-weight", type=float, default=2.0); ap.add_argument("--mel-weight", type=float, default=45.0)
ap.add_argument("--init-from", default="", help="vhead checkpoint to start from (fresh optimiser)"); ap.add_argument("--trunk", default="checkpoints/dec_A_gan3/last.pt")
ap.add_argument("--phase-weight", type=float, default=0.0, help="wrapped loss between the head's phase offset and the teacher's phase relative to the sine bank, at harmonic bins")
ap.add_argument("--fine-stft", action="store_true", help="add STFT losses with 64- and 128-sample windows, shorter than a pitch period, so the loss sees how energy pulses within each period")
ap.add_argument("--flat-weight", type=float, default=0.0, help="match the teacher's spectral flatness (how tonal vs noise-like) per 750 Hz band and frame above 1.5 kHz; a statistic, so noise is not penalised for being random")
ap.add_argument("--harm-weight", type=float, default=0.0, help="match the teacher's harmonicity (energy at multiples of the pitch vs halfway between them, log ratio) per 1 kHz band and voiced frame, 1-9 kHz")
ap.add_argument("--mod-weight", type=float, default=0.0, help="match how each upper band's loudness pulses with the pitch (depth and timing, at 1x and 2x the pitch) in 50 ms windows; v2 head only")
ap.add_argument("--head", default="v1"); ap.add_argument("--dim", type=int, default=160); ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--utmos", action="store_true"); ap.add_argument("--eval-every", type=int, default=2500); ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--device", default=DEVICE); ap.add_argument("--shards", type=int, default=8); ap.add_argument("--tag", default="")
A = ap.parse_args(); DEV = A.device; RUN = f"checkpoints/vh{A.tag}"; os.makedirs(RUN, exist_ok=True)

def load_rows():
    shards = sorted(glob.glob("data/teacher/shard_*.pt"))[:A.shards]
    def load(sh):
        rows = load_shard(sh)
        for r in rows:
            for k in ("audio_mps", "audio_cpu", "d", "dur", "ids"): r.pop(k, None)
        return rows
    rows0 = load(shards[0])
    rest = [r for sh in shards[1:] for r in load(sh)]
    if any("split" in r for r in rows0):
        held = [r for r in rows0 + rest if r.get("split") in ("val", "test")]
        train = [r for r in rows0 + rest if r.get("split") == "train"]
    else:
        held = rows0[:200]
        train = rows0[200:] + rest
    return held, train
held, train = load_rows(); print(f"{len(train)} train / {len(held)} held-out", flush=True)
def full(r):
    aln = alignment(r["pred_dur"]); asr = r["t_en"].float() @ aln; Fr = aln.shape[1]
    return asr, r["F0"][:2 * Fr], r["N"][:2 * Fr], r["audio"][:600 * Fr].float() / 32767
def segment(r, seg):
    asr, F0, N, au = full(r); Fr = asr.shape[1]
    if Fr <= seg: pad = seg - Fr; asr = F.pad(asr, (0, pad)); F0 = F.pad(F0, (0, 2 * pad)); N = F.pad(N, (0, 2 * pad)); au = F.pad(au, (0, 600 * pad)); a = 0
    else: a = random.randint(0, Fr - seg)
    c0 = (F0[:2 * a].double() * 300 / 24000).sum() % 1   # bank phase accumulated before the crop
    return asr[:, a:a + seg], F0[2 * a:2 * (a + seg)], N[2 * a:2 * (a + seg)], au[600 * a:600 * (a + seg)], c0.float()
def collate(rows, seg):
    x = [segment(r, seg) for r in rows]; return [torch.stack([t[i] for t in x]).to(DEV) for i in range(5)]

mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80).to(DEV)
def spec_losses(y, t):
    lm = F.l1_loss(torch.log(mel(y).clamp_min(1e-5)), torch.log(mel(t).clamp_min(1e-5))); ls = 0
    SZ = (64, 128, 512, 1024, 2048) if A.fine_stft else (512, 1024, 2048)
    for n in SZ:
        w = torch.hann_window(n, device=DEV)
        Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs(); T = torch.stft(t, n, n // 4, window=w, return_complex=True).abs()
        ls = ls + (T - Y).norm() / T.norm().clamp_min(1e-5) + F.l1_loss(torch.log(Y.clamp_min(1e-5)), torch.log(T.clamp_min(1e-5)))
    return lm, ls / len(SZ)

model = VDecoder(head=A.head, dim=A.dim, blocks=A.blocks).to(DEV); step = 0; last = f"{RUN}/last.pt"
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, betas=(0.8, 0.99), weight_decay=0.01)
sched = lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, A.steps) / A.steps))
if A.gan:
    from discriminators import Discriminators, d_loss, g_loss
    D = Discriminators().to(DEV); optD = torch.optim.AdamW(D.parameters(), lr=A.lr, betas=(0.8, 0.99), weight_decay=0.01)
if os.path.exists(last):
    ck = torch.load(last, map_location=DEV); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]; print(f"resumed at {step}", flush=True)
    if A.gan and "D" in ck: D.load_state_dict(ck["D"]); optD.load_state_dict(ck["optD"])
elif A.init_from: model.load_ckpt(A.init_from, DEV); print(f"initialised from {A.init_from}", flush=True)
else: model.load_trunk(A.trunk); print(f"trunk from {A.trunk}, head from scratch", flush=True)

def flat_loss(y, t, n=2048, bw=64):
    w = torch.hann_window(n, device=DEV); out = []
    for x in (y, t):
        P = torch.stft(x, n, n // 4, window=w, return_complex=True).abs().pow(2).clamp_min(1e-10)[:, 128:128 + 13 * bw]   # 1.5-11.25 kHz
        P = P.reshape(P.shape[0], 13, bw, -1); out.append(torch.log(P).mean(2) - torch.log(P.mean(2)))
    return F.l1_loss(out[0], out[1])
def harm_loss(y, t, f0, n=2048, hop=512):
    w = torch.hann_window(n, device=DEV); nfr = y.shape[-1] // hop + 1
    idx = (torch.arange(nfr, device=DEV) * hop // 300).clamp(max=f0.shape[1] - 1); lo = (idx - 2).clamp(min=0); hi = (idx + 2).clamp(max=f0.shape[1] - 1)
    f = f0[:, idx]; v = (torch.minimum(torch.minimum(f0[:, lo], f), f0[:, hi]) > 100).float()                     # steady voiced frames [B, T]
    fr = torch.arange(n // 2 + 1, device=DEV) * 24000 / n; ratio = fr[None, :, None] / f.clamp_min(60)[:, None]; frac = (ratio - ratio.round()).abs()
    mh, mi = (frac < 0.15).float(), ((frac - 0.5).abs() < 0.15).float(); bands = torch.stack([((fr >= 1000 * b) & (fr < 1000 * (b + 1))).float() for b in range(1, 9)])   # [8, bins]
    out = []
    for x in (y, t):
        P = torch.stft(x, n, hop, window=w, return_complex=True).abs().pow(2)
        out.append(torch.log((torch.einsum("bft,kf->bkt", P * mh, bands) + 1e-8) / (torch.einsum("bft,kf->bkt", P * mi, bands) + 1e-8)))
    return ((out[0] - out[1]).abs() * v[:, None]).sum() / (v.sum() * 8).clamp_min(1)
MODB = [(4, 8), (8, 12), (12, 16), (16, 21), (21, 29)]   # 375 Hz bins: 1.5-3, 3-4.5, 4.5-6, 6-8, 8-11 kHz
def mod_loss(y, t, cyc, uv, n=64, hop=16, win=75):
    w = torch.hann_window(n, device=DEV); k = y.shape[-1] // hop + 1
    ph = 2 * torch.pi * F.pad(cyc, (0, hop))[:, ::hop][:, :k]; v = F.pad(uv, (0, hop))[:, ::hop][:, :k]
    carr = torch.stack([fn(m * ph) for m in (1, 2) for fn in (torch.cos, torch.sin)], 1)                       # [B, 4, T]
    D = []
    for x in (y, t):
        X = torch.stft(x, n, hop, window=w, return_complex=True).abs(); e = torch.stack([X[:, a:b].sum(1) for a, b in MODB], 1)   # [B, 5, T]
        num = F.avg_pool1d((e[:, :, None] * carr[:, None]).flatten(1, 2), win, win // 2); den = F.avg_pool1d(e, win, win // 2).clamp_min(1e-5)
        D.append(2 * num.view(e.shape[0], 5, 4, -1) / den[:, :, None])
    vm = (F.avg_pool1d(v[:, None], win, win // 2) > 0.99).float()[:, :, None]                                  # [B, 1, 1, W]
    return ((D[0] - D[1]).abs() * vm).sum() / (vm.sum() * 20).clamp_min(1)
def phase_terms(head, au, eq=True):
    """Wrapped losses (1 - cos) between the head's phase offset and the teacher's phase relative to the sine bank:
    absolute, group delay (difference across 1/2/3/5 bins) and frame-to-frame change. Weights = teacher x bank
    magnitude (only bins at voiced harmonics count), divided per bin by their average so every band counts."""
    L = head.last; T = head.stft(au); k = min(T.shape[-1], L["E"].shape[-1]); T, E, ph = T[..., :k], L["E"][..., :k], L["ph"][..., :k]
    w = T.abs() * E.abs(); tgt = (T * E.conj()).angle()
    if eq: w = w / w.mean((0, 2), keepdim=True).clamp_min(1e-5)
    err = ph - tgt; f = lambda e, ww: (ww * (1 - torch.cos(e))).sum() / ww.sum().clamp_min(1e-5)
    gd = sum(f(err[:, d:] - err[:, :-d], (w[:, d:] * w[:, :-d]).sqrt()) for d in (1, 2, 3, 5)) / 4
    return f(err, w), gd, f(err[..., 1:] - err[..., :-1], (w[..., 1:] * w[..., :-1]).sqrt())
def phase_loss(au): return sum(phase_terms(model.head, au))
@torch.no_grad()
def evaluate(n_audio=8):
    model.eval(); from common import compare
    outs, refs, lp_acc = [], [], []; d = f"{RUN}/eval_{step}"; os.makedirs(d, exist_ok=True)
    for i, r in enumerate(held[:n_audio]):
        asr, F0, N, au = full(r); torch.manual_seed(i)
        y = model(asr[None].to(DEV), F0[None].to(DEV), N[None].to(DEV)).squeeze().cpu()
        if A.phase_weight > 0: lp_acc.append(phase_loss(au[None].to(DEV)).item())
        k = min(len(y), len(au)); outs.append(y[:k]); refs.append(au[:k]); sf.write(f"{d}/held_{i + 1}_student.wav", y.numpy(), 24000)
    raw, dtw, _ = compare(refs, outs); model.train(); ev = dict(step=step, dtw_l1=round(dtw, 4), raw_l1=round(raw, 4))
    if lp_acc: ev["held_phase"] = round(sum(lp_acc) / len(lp_acc), 4)
    if A.utmos:
        from utmos import score; ev["utmos"] = round(float(sum(score(f"{d}/held_{i + 1}_student.wav") for i in range(len(outs))) / len(outs)), 3)
    return ev
if A.eval_only: print(json.dumps(evaluate())); sys.exit()
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params (head {sum(p.numel() for p in model.head.parameters()) / 1e6:.2f}M), device {DEV}", flush=True)
model.train(); t0 = time.time(); s0 = step; log = open(f"{RUN}/log.jsonl", "a")
while step < A.steps:
    asr, F0, N, au, c0 = collate(random.sample(train, A.bs), A.seg)
    for g in opt.param_groups: g["lr"] = A.lr * sched(step)
    y = model(asr, F0, N, c0=c0); lm, ls = spec_losses(y, au)
    if A.gan:
        ld = d_loss(D, au, y); optD.zero_grad(); ld.backward(); optD.step()
        adv, fm = g_loss(D, au, y); loss = A.mel_weight * (lm + ls) + A.gan_weight * adv + A.fm_weight * fm
    else: loss = lm + ls
    if A.phase_weight > 0: lp = phase_loss(au); loss = loss + A.phase_weight * lp
    if A.flat_weight > 0: lf = flat_loss(y, au); loss = loss + A.flat_weight * lf
    if A.harm_weight > 0: lh = harm_loss(y, au, F0); loss = loss + A.harm_weight * lh
    if A.mod_weight > 0: lmod = mod_loss(y, au, model.head.last["cyc"], model.head.last["uv"]); loss = loss + A.mod_weight * lmod
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); step += 1
    if step % 50 == 0: print(f"step {step} loss {loss.item():.4f} mel {lm.item():.4f} stft {ls.item():.4f}" + (f" adv {adv.item():.3f} fm {fm.item():.3f} d {ld.item():.3f}" if A.gan else "") + (f" phase {lp.item():.4f}" if A.phase_weight > 0 else "") + (f" flat {lf.item():.4f}" if A.flat_weight > 0 else "") + (f" harm {lh.item():.4f}" if A.harm_weight > 0 else "") + (f" mod {lmod.item():.4f}" if A.mod_weight > 0 else "") + f" {(time.time() - t0) / (step - s0):.2f}s/step", flush=True)
    if step % 500 == 0 or step == A.steps:
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
        if A.gan: ck["D"] = D.state_dict(); ck["optD"] = optD.state_dict()
        torch.save(ck, last)
    if step % A.eval_every == 0 or step == A.steps:
        ev = evaluate(); print("EVAL", json.dumps(ev), flush=True); log.write(json.dumps(ev) + "\n"); log.flush()
        torch.save({"model": model.state_dict(), "step": step}, f"{RUN}/step{step}.pt")
print("done")
