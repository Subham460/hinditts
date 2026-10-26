"""Distillation corpus: run the Kokoro teacher (voice from scripts/lang.py) over sentences and store every
intermediate the half-wise experiments need. Resumable: writes shards of SHARD sentences,
skips shards that already exist.  usage: gen_teacher.py [n_sentences]  (default: all in SENTS_PATH)"""
import sys, os, json, time, warnings, torch
warnings.filterwarnings("ignore")
from kokoro import KPipeline
from lang import LANG_CODE, VOICE, SENTS_PATH, STYLE_INDEX_MAX, MAX_PHONEMES, DEVICE
N = int(sys.argv[1]) if len(sys.argv) > 1 else None
SHARD = 500; DEV = os.environ.get("DEVICE", DEVICE)
sents = json.load(open(SENTS_PATH, encoding="utf-8"))[:N]
pipe = KPipeline(lang_code=LANG_CODE, device=DEV, repo_id="hexgrad/Kokoro-82M"); m = pipe.model.eval()
ref = pipe.load_voice(VOICE)
os.makedirs("data/teacher", exist_ok=True)

@torch.no_grad()
def run(ps):
    kept = [c for c in ps if c in m.vocab]
    ids = torch.LongTensor([[0, *[m.vocab[c] for c in kept], 0]]).to(m.device)
    T = ids.shape[1]; r = ref[min(T - 3, STYLE_INDEX_MAX)].to(m.device)  # style indexed by phoneme count
    L = torch.full((1,), T, device=m.device); mask = torch.zeros(1, T, dtype=torch.bool, device=m.device)
    bert_dur = m.bert(ids, attention_mask=(~mask).int()); d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
    s = r[:, 128:]; d = m.predictor.text_encoder(d_en, s, L, mask); x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)
    pred_dur = torch.round(dur).clamp(min=1).long().squeeze(0)
    idx = torch.repeat_interleave(torch.arange(T, device=m.device), pred_dur)
    aln = torch.zeros(T, idx.shape[0], device=m.device); aln[idx, torch.arange(idx.shape[0])] = 1; aln = aln[None]
    en = d.transpose(-1, -2) @ aln; F0, Nn = m.predictor.F0Ntrain(en, s)
    t_en = m.text_encoder(ids, L, mask); asr = t_en @ aln
    audio = m.decoder(asr, F0, Nn, r[:, :128]).squeeze()
    c = lambda t: t.squeeze(0).cpu()
    return dict(ids=c(ids), dur=c(dur), pred_dur=pred_dur.cpu(), d=c(d).half(), t_en=c(t_en).half(),
                F0=c(F0), N=c(Nn), audio=(audio.cpu().clamp(-1, 1) * 32767).short(),
                _dropped=len(ps) - len(kept))

t0 = time.time(); secs = 0.0
for sh in range(0, len(sents), SHARD):
    path = f"data/teacher/shard_{sh // SHARD:04d}.pt"
    if os.path.exists(path): continue
    rows = []; dropped = 0
    for text in sents[sh:sh + SHARD]:
        ps, _ = pipe.g2p(text)
        if not ps or len(ps) > MAX_PHONEMES: continue
        r = run(ps); dropped += r.pop("_dropped"); r["text"] = text; r["ps"] = ps; r["voice"] = VOICE; rows.append(r); secs += len(r["audio"]) / 24000
    torch.save(rows, path)
    print(f"{path} {len(rows)} sents  {secs/3600:.2f} h audio  {dropped} phonemes dropped  {time.time()-t0:.0f}s elapsed  {secs/(time.time()-t0):.1f}x RT", flush=True)
print("done")
