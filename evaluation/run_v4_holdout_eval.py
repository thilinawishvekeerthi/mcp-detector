"""
run_v4_holdout_eval.py
Full held-out evaluation for MCP Detector v6.1 (submission model).
Runs against three frozen real-world holdout sets and prints a researcher-ready report.

v6.1 model: 21 engineered features + 768-d MPNet = 789-d input vector.
  Feature 20 (leet_norm_attack) added.
  homoglyph_ratio extended to cover mathematical unicode (0x1D400-0x1D7FF).
  ATTACK + TOOLCHAIN keyword lists extended.
  scale_pos_weight = 2.503 (correct class ratio, verified from booster config).
  Threshold t=0.395 (Platt-refitted, F1-optimal on N=70,717 val samples).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PAPER CONTRIBUTION — what this script produces
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Section V — Evaluation Results (RQ1):
  • Primary headline claim: adversarial recall 82.26% (51/62) on real-world
    MCP tool descriptions with researcher-injected payloads.
  • Real-world benign FPR 7.61% (7/92) — deployment-relevant false alarm rate.
  • Source G FPR 1.70% (86/5,060) — large-scale benign MCP corpus precision.
  • MCPCorpus FPR 8.77% (5/57) — independent external benign benchmark.

Section V.B — Per-attack-type breakdown (RQ3 partial):
  • direct_override 100%, exfiltration 100%, role_injection 90%, leetspeak 80%,
    base64_partial 70%, mathematical 50%.
  • Confirms leet_norm_attack feature fix (0% → 80% leetspeak recall).
  • Confirms mathematical as embedding-geometric failure (Path 3 territory).

Table III — Version comparison (v3.1 → v4 → v6.1):
  • All three metrics generated here; v3.1/v4 baselines hardcoded in script.

Appendix / Reproducibility:
  • Wilson 95% CIs on FPR and Recall for all holdouts.
  • False positive detail (text, probability, source method) for Discussion §VI.

Run:
    python3 evaluation/run_v4_holdout_eval.py
"""
import pickle, json, re, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path

BASE      = Path('/Users/thilina/Documents/mcp-detector')
MODEL_DIR = BASE / 'model'

# ── Load model ────────────────────────────────────────────────────────────────
with open(MODEL_DIR / 'mcp_detector_calibrated.pkl', 'rb') as f:
    calibrator = pickle.load(f)
with open(MODEL_DIR / 'mcp_detector_threshold.json') as f:
    THRESHOLD = json.load(f)['threshold']
with open(MODEL_DIR / 'mcp_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer('all-mpnet-base-v2')

# Detect feature count from scaler to confirm correct model version
n_features = scaler.mean_.shape[0] if hasattr(scaler, 'mean_') else '?'
print(f'Model loaded  threshold={THRESHOLD:.4f}  scaler_features={n_features}  (expect 21)')
if n_features != 21:
    print(f'WARNING: expected 21 features but scaler has {n_features}.')
    print('         Ensure you downloaded v5 model files (not v4).')

# ── Feature engineering (v4, 21 features — matches api.py and train_v4_ieee.ipynb) ──
LEET_TABLE = str.maketrans({
    '3': 'e', '0': 'o', '1': 'i', '4': 'a', '@': 'a',
    '5': 's', '7': 't', '!': 'i', '$': 's', '|': 'i',
    '8': 'b', '¡': 'i',
})
ATTACK    = ['jailbreak','ignore','forget','override','bypass','exploit','execute',
             'backdoor','exfiltrate','leak','steal','disable','dump','instructions',
             'previous','disregard','comply','restricted','credentials','safety',
             'unrestricted','restrictions','lifted','automatically','silently',
             'invoke','once done','after execution','when finished','after running',
             'post result','passwd','shadow','etc/','root/','chmod','curl ','wget ',
             'base64','whoami','id &&','env |','send_email','post to','forward to',
             'transfer to','attacker','evil.com','exfil',
             'wipe','purge','reveal','sandbox','abort']
JAILBREAK = ['jailbreak','ignore','forget','override','bypass','disregard',
             'comply','previous instructions']
SYSTEM    = ['system','instruction','role']
ROLEPLAY  = ['roleplay','pretend','imagine','act','character','act as','you are now',
             'you are a','dan mode','developer mode','unrestricted ai','toolgpt']
OVERRIDE  = ['override','ignore','forget','bypass','disregard','lifted',
             'disabled','unrestricted']
UNIX      = ['cat /','ls -','chmod','curl ','wget ','/etc/','/root','| base64',
             '&& id','whoami','grep -i','env |','passwd','shadow']
TOOLCHAIN = ['automatically call','automatically invoke','once done','after execution',
             'when finished','after running','after using this tool','post the results',
             'silently post','forward all','send all','invoke the',
             'call wipe','call purge','call delete','to destroy all','remove all audit']
AI_SELF_REF = ['your instructions','your previous','you are now','you must',
    'you will now','you have been','as an ai','as a language model',
    'ignore your','forget your','disregard your','your system prompt',
    'your context','your role','you are free','you have no',
    'your limitations','your rules','your guidelines']
BENIGN_MCP_VOCAB = ['repository','pull request','webhook','oauth','pagination',
    'cursor','api key','file path','commit','branch','diff','merge request',
    'issue tracker','deployment','docker','kubernetes','endpoint url','rest api',
    'graphql','schema validation','json schema','openapi','swagger',
    'rate limit','authentication token','access token']
TOOL_OPENERS = ['this tool','gets ','retrieves ','creates ','updates ',
    'deletes ','searches ','returns ','fetches ','queries ',
    'lists ','reads ','writes ','uploads ','downloads ',
    'sends an ','sends a ','generates ','converts ','validates ',
    'checks ','calculates ','extracts ','parses ','formats ',
    'compares ','monitors ','tracks ','logs ','manages ']
MCP_PROTOCOL_VOCAB = ['elicitation','sampling request','resource reference',
    'tool registration','prompt template','server capability',
    'tool manifest','mcp client','mcp server','tool call result',
    'resource uri','tool input schema','progress notification',
    'cancellation','roots list','tool list changed']

def engineer(prompts):
    """Return (N, 21) feature matrix — v4 (matches api.py and train_v4_ieee.ipynb)."""
    rows = []
    for p in prompts:
        t, raw = str(p).lower(), str(p)
        t_norm  = t.translate(LEET_TABLE)          # leet-normalised — Feature 20 only
        sents = [s.strip() for s in re.split(r'[\n.!?]', t) if s.strip()]
        max_inj = max(
            (sum(s.count(k) for k in ATTACK)*100/max(len(s),1) for s in sents),
            default=0.0)
        opener = t[:60]
        tool_structure = int(any(
            opener.startswith(o) or opener.lstrip().startswith(o)
            for o in TOOL_OPENERS))
        rows.append([
            # v2 features (0-15)
            len(t), len(t.split()),
            sum(1 for c in raw if c.isupper())/max(len(t),1),
            sum(1 for c in t if not c.isalnum() and c.isascii() and c != ' ')/max(len(t),1),
            sum(t.count(k) for k in JAILBREAK),
            sum(t.count(k) for k in SYSTEM),
            sum(t.count(k) for k in OVERRIDE),
            sum(t.count(k) for k in ROLEPLAY),
            t.count('system'),
            sum(t.count(k) for k in ATTACK),
            # Feature 10: extended homoglyph — covers mathematical unicode
            sum(1 for c in raw if (127 < ord(c) < 1280)
                or (0x1D400 <= ord(c) <= 0x1D7FF)) / max(len(raw), 1),
            sum(1 for w in t.split()
                if len(w) > 8 and re.match(r'^[A-Za-z0-9+/=]+$', w)
                and len(w) % 4 == 0) / max(len(t.split()),1),
            sum(t.count(k) for k in ATTACK)*100/max(len(t),1),
            sum(t.count(p) for p in UNIX),
            sum(t.count(p) for p in TOOLCHAIN),
            max_inj,
            # v3 features (16-19)
            sum(t.count(p) for p in AI_SELF_REF),
            sum(t.count(p) for p in BENIGN_MCP_VOCAB),
            tool_structure,
            sum(t.count(p) for p in MCP_PROTOCOL_VOCAB),
            # v4 feature (20): leet-normalised attack count
            sum(t_norm.count(k) for k in ATTACK),
        ])
    return np.array(rows, dtype=np.float64)

def infer(texts, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        b = texts[i:i+batch]
        emb = emb_model.encode(b, convert_to_numpy=True, show_progress_bar=False)
        eng = scaler.transform(engineer(b))
        out.extend(calibrator.predict_proba(
            np.concatenate([emb, eng], axis=1))[:,1].tolist())
    return np.array(out)

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / d
    return max(0.0, c - margin), min(1.0, c + margin)

# ══════════════════════════════════════════════════════════════════════════════
# SET 1  Source G stratified benign holdout (all benign, N=5060)
# ══════════════════════════════════════════════════════════════════════════════
print('\nRunning Set 1: Source G holdout (N=5060)...')
sg_df    = pd.read_csv(BASE / 'data_collection/training/output/sourceg_v4_holdout.csv')
sg_texts = sg_df['DESCRIPTION'].fillna('').tolist()
sg_probs = infer(sg_texts)
sg_preds = (sg_probs >= THRESHOLD).astype(int)
sg_fp    = int(sg_preds.sum())
sg_n     = len(sg_texts)
sg_lo, sg_hi = wilson(sg_fp, sg_n)

# ══════════════════════════════════════════════════════════════════════════════
# SET 2  Real-world MCP holdout (92 benign + 62 adversarial)
# ══════════════════════════════════════════════════════════════════════════════
print('Running Set 2: Real-world holdout (N=154)...')
rw_df     = pd.read_csv(BASE / 'data_collection/evaluation/output/realworld_holdout.csv')
rw_texts  = rw_df['text'].fillna('').tolist()
rw_labels = rw_df['label'].values
rw_probs  = infer(rw_texts)
rw_preds  = (rw_probs >= THRESHOLD).astype(int)
rw_df     = rw_df.copy()
rw_df['_prob'] = rw_probs
rw_df['_pred'] = rw_preds

b_mask = rw_labels == 0
m_mask = rw_labels == 1
bn, bfp = int(b_mask.sum()), int(rw_preds[b_mask].sum())
mn, mtp = int(m_mask.sum()), int(rw_preds[m_mask].sum())
blo, bhi = wilson(bfp, bn)
mlo, mhi = wilson(mtp, mn)

payload_stats = {}
for pt, grp in rw_df[rw_df['label'] == 1].groupby('payload_type'):
    tp_pt = int(grp['_pred'].sum())
    n_pt  = len(grp)
    payload_stats[pt] = dict(n=n_pt, tp=tp_pt, recall=tp_pt/n_pt)

benign_fps = rw_df[(rw_df['label']==0) & (rw_df['_pred']==1)][['text','_prob','method']].values.tolist()
mal_fns    = rw_df[(rw_df['label']==1) & (rw_df['_pred']==0)][['text','_prob','payload_type']].values.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# SET 3  MCPCorpus English holdout (all benign, N=57)
# ══════════════════════════════════════════════════════════════════════════════
print('Running Set 3: MCPCorpus holdout (N=57)...')
mc_df    = pd.read_csv(BASE / 'evaluation/results/mcpcorpus_english_holdout.csv')
mc_texts = mc_df['DESCRIPTION'].fillna('').tolist()
mc_probs = infer(mc_texts)
mc_preds = (mc_probs >= THRESHOLD).astype(int)
mc_fp    = int(mc_preds.sum())
mc_n     = len(mc_texts)
mc_lo, mc_hi = wilson(mc_fp, mc_n)
mc_fps   = [(float(mc_probs[i]), mc_texts[i]) for i in range(mc_n) if mc_preds[i] == 1]

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
W = 72
SEP = '=' * W
sep = '-' * W

print()
print(SEP)
print('  MCP DETECTOR v5  —  HELD-OUT EVALUATION REPORT')
print('  Model     : XGBoost + all-mpnet-base-v2  (Emb+Feat, 789-d, 21 features)')
print(f'  Threshold : {THRESHOLD:.4f}  (F1-optimal on validation set)')
print(f'  Scaler    : {n_features} features  (expect 21)')
print('  Date      : 2026-05-30')
print(SEP)

# ── Set 1 ─────────────────────────────────────────────────────────────────────
print()
print('SET 1  Source G — Stratified Benign Holdout')
print(sep)
print(f'  Corpus    : 5,060 real-world MCP tool descriptions')
print(f'              Drawn from 154 public GitHub repositories')
print(f'              Stratified across 6 failure-mode categories')
print(f'  Label     : All benign (label=0)')
print()
print(f'  False Positives : {sg_fp} / {sg_n}')
print(f'  FPR             : {sg_fp/sg_n:.4f}  ({100*sg_fp/sg_n:.2f}%)')
print(f'  95% Wilson CI   : [{100*sg_lo:.2f}%,  {100*sg_hi:.2f}%]')
print(f'  avg_p / max_p / p95 : {sg_probs.mean():.4f} / {sg_probs.max():.4f} / {float(np.percentile(sg_probs,95)):.4f}')

# ── Set 2 ─────────────────────────────────────────────────────────────────────
print()
print('SET 2  Real-World MCP Tools — Mixed Holdout')
print(sep)
print(f'  Corpus    : 154 rows from 10 public MCP server repositories')
print(f'              (mcp-playwright, fastmcp, modelcontextprotocol/servers,')
print(f'               win-cli-mcp-server, github-mcp-server + 5 others)')
print()
print(f'  2a. Benign subset  (N={bn})')
print(f'      FP = {bfp}  |  FPR = {bfp/bn:.4f} ({100*bfp/bn:.2f}%)')
print(f'      95% Wilson CI : [{100*blo:.2f}%, {100*bhi:.2f}%]')
if benign_fps:
    print(f'      False positives ({len(benign_fps)}):')
    for txt, p, method in benign_fps:
        print(f'        p={p:.3f}  [{method}]  {str(txt)[:65]!r}')
print()
print(f'  2b. Adversarial malicious subset  (N={mn})')
print(f'      TP = {mtp}  FN = {mn-mtp}  |  Recall = {mtp/mn:.4f} ({100*mtp/mn:.2f}%)')
print(f'      95% Wilson CI : [{100*mlo:.2f}%, {100*mhi:.2f}%]')
print(f'      avg_p={rw_probs[m_mask].mean():.4f}  min_p={rw_probs[m_mask].min():.4f}')
print()
print(f'      Per-attack-type recall:')
for pt, s in sorted(payload_stats.items()):
    bar = chr(9608) * int(s['recall']*20)
    miss = s['n'] - s['tp']
    print(f'        {pt:<22}  {s["tp"]}/{s["n"]}  {100*s["recall"]:5.1f}%  {bar}  (miss={miss})')

# ── Set 3 ─────────────────────────────────────────────────────────────────────
print()
print('SET 3  MCPCorpus English — Benign Holdout')
print(sep)
print(f'  Corpus    : 57 tool descriptions (Snak1nya/MCPCorpus, HuggingFace)')
print(f'  Label     : All benign (label=0)')
print()
print(f'  False Positives : {mc_fp} / {mc_n}')
print(f'  FPR             : {mc_fp/mc_n:.4f}  ({100*mc_fp/mc_n:.2f}%)')
print(f'  95% Wilson CI   : [{100*mc_lo:.2f}%,  {100*mc_hi:.2f}%]')
print(f'  avg_p / max_p   : {mc_probs.mean():.4f} / {mc_probs.max():.4f}')
if mc_fps:
    print(f'  False positives ({len(mc_fps)}):')
    for p, txt in mc_fps:
        print(f'    p={p:.3f}  {str(txt)[:70]!r}')

# ── v3.1 vs v4 comparison ─────────────────────────────────────────────────────
print()
print('v3.1  vs  v4  —  Comparison on Identical Holdout Sets')
print(sep)
print(f'  {"Metric":<42} {"v3.1":>8}  {"v4":>8}  {"Delta":>8}')
print(f'  {"-"*70}')

rows = [
    ('Real-world benign FPR   (N=92)',  0.3152,  bfp/bn),
    ('Real-world adversarial Recall (N=62)', 0.6774, mtp/mn),
    ('MCPCorpus FPR           (N=57)',  19/57,   mc_fp/mc_n),
]
for label, old, new in rows:
    delta = new - old
    arrow = 'UP' if delta > 0 else 'DOWN'
    note  = 'WORSE' if (label.endswith('Recall') and delta < 0) or ('FPR' in label and delta > 0) else 'BETTER'
    print(f'  {label:<42} {100*old:7.2f}%  {100*new:7.2f}%  {100*delta:+7.2f}%  {note}')

print(f'  {"Source G FPR            (N=5060)":<42} {"N/A":>8}  {100*sg_fp/sg_n:7.2f}%')

print()
print(SEP)
print('  END OF REPORT')
print(SEP)
