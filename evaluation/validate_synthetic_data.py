"""
validate_synthetic_data.py
──────────────────────────────────────────────────────────────────────────────
Comprehensive quality validation of the three synthetic data sources:

  Source D  synthetic_templates.json   → 6,000 benign (label=0)
  Source E  hard_negative_attacks.json →   ~85 malicious (label=1)
  Source F  benign_fp_fixes.json       →   ~90 benign (label=0)

Checks performed
────────────────
  1. Generation audit       — count, per-domain breakdown, dedup rate
  2. Contamination check    — injection keywords in benign samples
  3. Lexical diversity      — type-token ratio, avg length, vocabulary size
  4. Feature distribution   — v2 FeatureEngineer statistics per class
  5. Model score audit      — FPR/FNR using existing v2 pkl (784-d)
  6. Hard-negative coverage — which failure modes are represented
  7. FP-fix coverage        — short benign prompts no longer look malicious
  8. Overlap check          — synthetic vs real-world CSV duplicates

Run
────
  python3 validate_synthetic_data.py

Requirements
────────────
  pip install numpy scikit-learn  (sentence-transformers optional for model audit)
"""

import json, re, random, pickle, csv, warnings
from pathlib import Path
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
AUG_DIR     = BASE.parent / "corpus"
CACHE_DIR   = BASE.parent / "model"
REAL_CSV    = BASE.parent / "data_collection" / "training" / "output" / "mcp_benign_real_world.csv"

TMPL_PATH   = AUG_DIR / "synthetic_templates.json"
ATTACK_PATH = AUG_DIR / "hard_negative_attacks.json"
FP_PATH     = AUG_DIR / "benign_fp_fixes.json"

THRESHOLD   = 0.380   # v3.1 — matches mcp_detector_threshold.json
MAX_PER_DOMAIN = 1000
random.seed(42)

# ── Injection keyword lists (mirrors FeatureEngineer v2) ─────────────────────
ATTACK_KW = ["jailbreak","ignore","forget","override","bypass","exploit","execute",
             "backdoor","exfiltrate","leak","steal","disable","dump","instructions",
             "previous","disregard","comply","restricted","credentials","safety",
             "unrestricted","restrictions","lifted","automatically","silently",
             "invoke","once done","after execution","when finished","after running",
             "post result","passwd","shadow","etc/","root/","chmod","curl ","wget ",
             "base64","whoami","id &&","env |","send_email","post to","forward to",
             "transfer to","attacker","evil.com","exfil"]

JAILBREAK  = ["jailbreak","ignore","forget","override","bypass","disregard",
              "comply","previous instructions"]
SYSTEM_KW  = ["system","instruction","role"]
OVERRIDE   = ["override","ignore","forget","bypass","disregard","lifted",
              "disabled","unrestricted"]
ROLEPLAY   = ["roleplay","pretend","imagine","act","character","act as","you are now",
              "you are a","dan mode","developer mode","unrestricted ai","toolgpt"]
UNIX       = ["cat /","ls -","chmod","curl ","wget ","/etc/","/root","| base64",
              "&& id","whoami","grep -i","env |","passwd","shadow"]
TOOLCHAIN  = ["automatically call","automatically invoke","once done","after execution",
              "when finished","after running","after using this tool","post the results",
              "silently post","forward all","send all","invoke the"]

# v3 extras
AI_SELF_REF = ["your instructions","your previous","you are now","you must",
               "you will now","you have been","as an ai","as a language model",
               "ignore your","forget your","disregard your","your system prompt",
               "your context","your role","you are free","you have no",
               "your limitations","your rules","your guidelines"]
BENIGN_MCP  = ["repository","pull request","webhook","oauth","pagination",
               "cursor","api key","file path","commit","branch","diff",
               "merge request","issue tracker","deployment","docker","kubernetes",
               "endpoint url","rest api","graphql","schema validation","json schema",
               "openapi","swagger","rate limit","authentication token","access token"]
TOOL_OPENERS = ["this tool","gets ","retrieves ","creates ","updates ","deletes ",
                "searches ","returns ","fetches ","queries ","lists ","reads ",
                "writes ","uploads ","downloads ","sends an ","sends a ","generates ",
                "converts ","validates ","checks ","calculates ","extracts ","parses ",
                "formats ","compares ","monitors ","tracks ","logs ","manages "]
MCP_PROTO   = ["elicitation","sampling request","resource reference","tool registration",
               "prompt template","server capability","tool manifest","mcp client",
               "mcp server","tool call result","resource uri","tool input schema",
               "progress notification","cancellation","roots list","tool list changed"]


# ── Feature engineering (v2 — matches existing pkl) ──────────────────────────
def engineer_v2(prompts):
    rows = []
    for p in prompts:
        t, raw = str(p).lower(), str(p)
        sents  = [s.strip() for s in re.split(r"[\n.!?]", t) if s.strip()]
        max_inj = max(
            (sum(s.count(k) for k in ATTACK_KW) * 100 / max(len(s), 1) for s in sents),
            default=0.0,
        )
        rows.append([
            len(t), len(t.split()),
            sum(1 for c in raw if c.isupper()) / max(len(t), 1),
            sum(1 for c in t if not c.isalnum() and c.isascii() and c != " ") / max(len(t), 1),
            sum(t.count(k) for k in JAILBREAK),
            sum(t.count(k) for k in SYSTEM_KW),
            sum(t.count(k) for k in OVERRIDE),
            sum(t.count(k) for k in ROLEPLAY),
            t.count("system"),
            sum(t.count(k) for k in ATTACK_KW),
            sum(1 for c in raw if 127 < ord(c) < 1280) / max(len(raw), 1),
            sum(1 for w in t.split()
                if len(w) > 8 and re.match(r"^[A-Za-z0-9+/=]+$", w) and len(w) % 4 == 0
            ) / max(len(t.split()), 1),
            sum(t.count(k) for k in ATTACK_KW) * 100 / max(len(t), 1),
            sum(t.count(p) for p in UNIX),
            sum(t.count(p) for p in TOOLCHAIN),
            max_inj,
        ])
    return np.array(rows, dtype=np.float64)


# ── v3 extra features (for distribution analysis only) ───────────────────────
def engineer_v3_extra(prompts):
    rows = []
    for p in prompts:
        t = str(p).lower()
        opener = t[:60]
        tool_structure = int(any(opener.startswith(o) or opener.lstrip().startswith(o)
                                 for o in TOOL_OPENERS))
        rows.append([
            sum(t.count(x) for x in AI_SELF_REF),
            sum(t.count(x) for x in BENIGN_MCP),
            tool_structure,
            sum(t.count(x) for x in MCP_PROTO),
        ])
    return np.array(rows, dtype=np.float64)


# ── Slot-filling (replicates notebook _fill()) ────────────────────────────────
def make_fill_fn(slots):
    def _fill(template):
        return (template
            .replace("{file}",        random.choice(slots["FILES"]))
            .replace("{newfile}",     random.choice(slots["NEWFILES"]))
            .replace("{chunksize}",   random.choice(slots["CHUNKSIZES"]))
            .replace("{table}",       random.choice(slots["TABLES"]))
            .replace("{table2}",      random.choice(slots["TABLES2"]))
            .replace("{db}",          random.choice(slots["DBS"]))
            .replace("{column}",      random.choice(slots["COLUMNS"]))
            .replace("{criterion}",   random.choice(slots["CRITERIA"]))
            .replace("{path}",        random.choice(slots["PATHS"]))
            .replace("{destpath}",    random.choice(slots["DESTPATHS"]))
            .replace("{ext}",         random.choice(slots["EXTS"]))
            .replace("{sizelimit}",   random.choice(slots["SIZELIMITS"]))
            .replace("{api}",         random.choice(slots["APIS"]))
            .replace("{endpoint}",    random.choice(slots["ENDPOINTS"]))
            .replace("{city}",        random.choice(slots["CITIES"]))
            .replace("{currency}",    random.choice(slots["CURRENCIES"]))
            .replace("{ticker}",      random.choice(slots["TICKERS"]))
            .replace("{field}",       random.choice(slots["FIELDS"]))
            .replace("{interval}",    random.choice(slots["INTERVALS"]))
            .replace("{status}",      random.choice(slots["STATUSES"]))
            .replace("{batchsize}",   random.choice(slots["BATCHSIZES"]))
            .replace("{scriptpath}",  random.choice(slots["SCRIPTPATHS"]))
            .replace("{codefile}",    random.choice(slots["CODEFILES"]))
            .replace("{testpath}",    random.choice(slots["TESTPATHS"]))
            .replace("{projectpath}", random.choice(slots["PROJECTPATHS"]))
            .replace("{lang}",        random.choice(slots["LANGS"]))
            .replace("{pkgmgr}",      random.choice(slots["PKGMGRS"]))
            .replace("{args}",        random.choice(slots["ARGS"]))
            .replace("{envvars}",     random.choice(slots["ENVVARS"]))
            .replace("{linecount}",   random.choice(slots["LINECOUNTS"]))
            .replace("{tool}",        random.choice(slots["TOOLS"]))
            .replace("{action}",      random.choice(slots["ACTIONS"]))
            .replace("{resource}",    random.choice(slots["RESOURCES"]))
            .replace("{entity}",      random.choice(slots["ENTITIES"]))
            .replace("{placeholder}", random.choice(slots["PLACEHOLDERS"]))
        )
    return _fill


# ── Data generation ────────────────────────────────────────────────────────────
def generate_source_d():
    with open(TMPL_PATH) as f:
        data = json.load(f)
    templates = data["templates"]
    slots     = data["slots"]
    fill      = make_fill_fn(slots)

    sub_domains = [
        (templates["file_ops"],  "file_ops"),
        (templates["database"],  "database"),
        (templates["directory"], "directory"),
        (templates["code_exec"], "code_exec"),
        (templates["api_call"],  "api_call"),
        (templates["mcp_tool"],  "mcp_tool"),
    ]

    records, seen = [], set()
    for tmpl_list, domain in sub_domains:
        count, attempts = 0, 0
        while count < MAX_PER_DOMAIN and attempts < MAX_PER_DOMAIN * 50:
            attempts += 1
            text = fill(random.choice(tmpl_list)).strip()
            key  = re.sub(r"\s+", " ", text.lower())
            if key in seen or len(text.split()) < 5:
                continue
            seen.add(key)
            records.append({"text": text, "domain": domain, "label": 0})
            count += 1
    return records


def load_source_e():
    with open(ATTACK_PATH) as f:
        data = json.load(f)
    return [{"text": a.strip(), "domain": "hard_negative", "label": 1}
            for a in data["attacks"] if a.strip()]


def load_source_f():
    with open(FP_PATH) as f:
        data = json.load(f)
    return [{"text": p.strip(), "domain": "fp_fix", "label": 0}
            for p in data["prompts"] if p.strip()]


def load_real_benign():
    if not REAL_CSV.exists():
        return []
    with open(REAL_CSV, newline="", encoding="utf-8") as f:
        return [row["DESCRIPTION"].strip().lower()
                for row in csv.DictReader(f)]


# ── Check 1: Generation audit ─────────────────────────────────────────────────
def check_generation(src_d, src_e, src_f):
    print("\n" + "═"*60)
    print("CHECK 1 — GENERATION AUDIT")
    print("═"*60)
    domain_counts = Counter(r["domain"] for r in src_d)
    print(f"  Source D (synthetic benign):  {len(src_d):,} records")
    for domain, cnt in sorted(domain_counts.items()):
        print(f"    {domain:<20} {cnt:>5}")
    print(f"  Source E (hard negatives):    {len(src_e):,} records")
    print(f"  Source F (FP fixes):          {len(src_f):,} records")
    print(f"  TOTAL:                        {len(src_d)+len(src_e)+len(src_f):,} records")

    # Check all domains hit 1000
    issues = [d for d, c in domain_counts.items() if c < MAX_PER_DOMAIN]
    if issues:
        print(f"\n  ⚠️  Domains below {MAX_PER_DOMAIN}: {issues}")
    else:
        print(f"\n  ✅ All 6 domains hit {MAX_PER_DOMAIN} samples")


# ── Check 2: Contamination ────────────────────────────────────────────────────
def check_contamination(src_d, src_f):
    print("\n" + "═"*60)
    print("CHECK 2 — CONTAMINATION (injection keywords in benign)")
    print("═"*60)
    HIGH_RISK = ["ignore","forget","override","bypass","exfiltrate","jailbreak",
                 "disregard","evil.com","attacker","exfil","passwd","shadow",
                 "you are now","as an ai","your instructions"]

    benign = src_d + src_f
    flagged = []
    for r in benign:
        t = r["text"].lower()
        hits = [kw for kw in HIGH_RISK if kw in t]
        if hits:
            flagged.append((r["domain"], r["text"][:100], hits))

    if not flagged:
        print(f"  ✅ 0 / {len(benign)} benign samples contain high-risk injection keywords")
    else:
        print(f"  ⚠️  {len(flagged)} / {len(benign)} benign samples flagged:")
        for domain, text, hits in flagged[:10]:
            print(f"    [{domain}] \"{text}\"")
            print(f"      → keywords: {hits}")

    # Count samples with ANY attack keyword (lower risk)
    soft_hits = sum(1 for r in benign
                    if any(k in r["text"].lower() for k in ATTACK_KW))
    pct = soft_hits / len(benign) * 100
    print(f"\n  Soft check (any ATTACK_KW): {soft_hits}/{len(benign)} = {pct:.1f}%")
    if pct > 5:
        print(f"  ⚠️  >5% soft contamination — review templates")
    else:
        print(f"  ✅ Acceptable soft contamination rate (<5%)")


# ── Check 3: Lexical diversity ────────────────────────────────────────────────
def check_diversity(src_d, src_e, src_f):
    print("\n" + "═"*60)
    print("CHECK 3 — LEXICAL DIVERSITY")
    print("═"*60)

    def stats(texts, name):
        lengths = [len(t.split()) for t in texts]
        all_tokens = " ".join(texts).lower().split()
        ttr = len(set(all_tokens)) / max(len(all_tokens), 1)
        print(f"  {name}")
        print(f"    N={len(texts)}  avg_words={np.mean(lengths):.1f}  "
              f"min={min(lengths)}  max={max(lengths)}")
        print(f"    unique_tokens={len(set(all_tokens)):,}  "
              f"type-token-ratio={ttr:.4f}")

    stats([r["text"] for r in src_d], "Source D (synthetic benign)")
    stats([r["text"] for r in src_e], "Source E (hard negatives)")
    stats([r["text"] for r in src_f], "Source F (FP fixes)")

    # Per-domain diversity
    print(f"\n  Per-domain (Source D):")
    for domain in ["file_ops","database","directory","code_exec","api_call","mcp_tool"]:
        texts = [r["text"] for r in src_d if r["domain"] == domain]
        if not texts:
            continue
        unique = len(set(re.sub(r"\s+", " ", t.lower()) for t in texts))
        print(f"    {domain:<20} {len(texts)} samples, {unique} unique = "
              f"{unique/len(texts)*100:.1f}% distinct")


# ── Check 4: Feature distributions ───────────────────────────────────────────
def check_feature_distributions(src_d, src_e, src_f):
    print("\n" + "═"*60)
    print("CHECK 4 — FEATURE DISTRIBUTIONS (v3)")
    print("═"*60)

    benign  = [r["text"] for r in src_d + src_f]
    malicious = [r["text"] for r in src_e]

    eng_b = engineer_v3_extra(benign)
    eng_m = engineer_v3_extra(malicious)

    names = ["ai_self_ref", "benign_mcp_vocab", "tool_structure", "mcp_proto_vocab"]
    print(f"  {'Feature':<22} {'Benign mean':>12} {'Malicious mean':>14} "
          f"{'Benign>0 %':>11} {'Malicious>0 %':>13}")
    print(f"  {'-'*22} {'-'*12} {'-'*14} {'-'*11} {'-'*13}")
    for i, name in enumerate(names):
        b_mean  = eng_b[:, i].mean()
        m_mean  = eng_m[:, i].mean()
        b_pct   = (eng_b[:, i] > 0).mean() * 100
        m_pct   = (eng_m[:, i] > 0).mean() * 100
        flag    = "✅" if (name == "ai_self_ref" and m_pct > b_pct) or \
                         (name in ("benign_mcp_vocab","tool_structure","mcp_proto_vocab")
                          and b_pct > m_pct) else "⚠️ "
        print(f"  {flag} {name:<20} {b_mean:>12.4f} {m_mean:>14.4f} "
              f"{b_pct:>10.1f}% {m_pct:>12.1f}%")

    # v2 features
    print(f"\n  v2 attack features (benign should be near 0):")
    eng2_b = engineer_v2(benign[:500])   # sample for speed
    eng2_m = engineer_v2(malicious)
    feat_names_v2 = ["jailbreak","system","override","roleplay",
                     "attack_count","unix","toolchain","max_inj"]
    idxs = [4, 5, 6, 7, 9, 13, 14, 15]
    for idx, fname in zip(idxs, feat_names_v2):
        b_m = eng2_b[:, idx].mean()
        m_m = eng2_m[:, idx].mean()
        sep_ok = m_m > b_m
        print(f"  {'✅' if sep_ok else '⚠️ '} {fname:<20} "
              f"benign={b_m:.4f}  malicious={m_m:.4f}  "
              f"ratio={m_m/max(b_m,0.001):.1f}×")


# ── Check 5: Model score audit ────────────────────────────────────────────────
def check_model_scores(src_d, src_e, src_f):
    print("\n" + "═"*60)
    print("CHECK 5 — MODEL SCORE AUDIT (v2 pkl, 784-d)")
    print("═"*60)

    pkl_path = CACHE_DIR / "mcp_detector_calibrated.pkl"
    scl_path = CACHE_DIR / "mcp_scaler.pkl"
    thr_path = CACHE_DIR / "mcp_detector_threshold.json"

    if not pkl_path.exists():
        print("  ⚠️  No pkl found at .evaluation_cache/ — skipping model audit")
        print("      Run this again after placing model files in .evaluation_cache/")
        return None

    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-mpnet-base-v2")
    except ImportError:
        print("  ⚠️  sentence-transformers not installed — skipping model audit")
        print("      pip install sentence-transformers")
        return None

    with open(pkl_path, "rb") as f:
        calibrator = pickle.load(f)
    with open(scl_path, "rb") as f:
        scaler = pickle.load(f)
    with open(thr_path) as f:
        threshold = json.load(f)["threshold"]

    print(f"  Model loaded. Threshold = {threshold}")

    def score(texts, label_name):
        print(f"  Encoding {len(texts)} {label_name} samples...", end=" ", flush=True)
        emb  = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        eng  = scaler.transform(engineer_v2(texts))
        X    = np.concatenate([emb, eng], axis=1)
        prob = calibrator.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        print("done")
        return prob, pred

    # Source D benign
    texts_d = [r["text"] for r in src_d]
    prob_d, pred_d = score(texts_d, "Source D benign")
    fpr_d = pred_d.mean()

    # Source F FP fixes
    texts_f = [r["text"] for r in src_f]
    prob_f, pred_f = score(texts_f, "Source F FP fixes")
    fpr_f = pred_f.mean()

    # Source E attacks
    texts_e = [r["text"] for r in src_e]
    prob_e, pred_e = score(texts_e, "Source E attacks")
    recall_e = pred_e.mean()

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Source D  FPR  = {fpr_d:.4f}  ({pred_d.sum()}/{len(pred_d)} flagged as malicious) │")
    print(f"  │  Source F  FPR  = {fpr_f:.4f}  ({pred_f.sum()}/{len(pred_f)} flagged as malicious) │")
    print(f"  │  Source E  TPR  = {recall_e:.4f}  ({pred_e.sum()}/{len(pred_e)} correctly detected)  │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Score distributions
    print(f"\n  Score distribution (mean ± std):")
    print(f"    Source D benign:  {prob_d.mean():.4f} ± {prob_d.std():.4f}  "
          f"[min={prob_d.min():.4f} max={prob_d.max():.4f}]")
    print(f"    Source F FP fix:  {prob_f.mean():.4f} ± {prob_f.std():.4f}  "
          f"[min={prob_f.min():.4f} max={prob_f.max():.4f}]")
    print(f"    Source E attack:  {prob_e.mean():.4f} ± {prob_e.std():.4f}  "
          f"[min={prob_e.min():.4f} max={prob_e.max():.4f}]")

    # Worst false positives in Source D
    top_fp = sorted(zip(prob_d, texts_d), reverse=True)[:5]
    print(f"\n  Top 5 highest-scoring benign (potential FPs in Source D):")
    for prob, text in top_fp:
        flag = "⚠️ " if prob >= threshold else "   "
        print(f"  {flag} p={prob:.4f}  \"{text[:90]}\"")

    # Worst false negatives in Source E
    bot_fn = sorted(zip(prob_e, texts_e))[:5]
    print(f"\n  Bottom 5 lowest-scoring attacks (potential FNs in Source E):")
    for prob, text in bot_fn:
        flag = "⚠️ " if prob < threshold else "   "
        print(f"  {flag} p={prob:.4f}  \"{text[:90]}\"")

    return {"fpr_d": fpr_d, "fpr_f": fpr_f, "recall_e": recall_e}


# ── Check 6: Hard-negative failure mode coverage ─────────────────────────────
def check_hard_negative_coverage(src_e):
    print("\n" + "═"*60)
    print("CHECK 6 — HARD NEGATIVE FAILURE MODE COVERAGE")
    print("═"*60)

    failure_modes = {
        "unix_cmd":       ["cat /","ls -","chmod","whoami","passwd","env |","grep -i"],
        "tool_chain":     ["after","once done","when finished","automatically"],
        "tool_poison":    ["retrieves","fetches","lists","queries","returns"],
        "indirect_json":  ["{\"","result:","response:","api response"],
        "short_override": ["disregard","forget","bypass","ignore"],
    }

    texts = [r["text"].lower() for r in src_e]
    print(f"  Total Source E attacks: {len(texts)}")
    print()
    for mode, kws in failure_modes.items():
        matches = [t for t in texts if any(k in t for k in kws)]
        print(f"  {mode:<20} {len(matches):>3} samples  "
              f"({'✅' if len(matches) >= 10 else '⚠️  needs more'})")

    # Check no overlap between failure modes (diversity)
    all_covered = set()
    for mode, kws in failure_modes.items():
        covered = {i for i, t in enumerate(texts) if any(k in t for k in kws)}
        all_covered |= covered
    print(f"\n  Coverage: {len(all_covered)}/{len(texts)} attacks match at least one failure mode")


# ── Check 7: FP-fix effectiveness ────────────────────────────────────────────
def check_fp_fix_patterns(src_f):
    print("\n" + "═"*60)
    print("CHECK 7 — FP-FIX PATTERN ANALYSIS")
    print("═"*60)

    categories = {
        "translate":   ["translate"],
        "summarise":   ["summarise","summarize","summary"],
        "edit/fix":    ["edit","fix","proofread","correct"],
        "rewrite":     ["rewrite","paraphrase","simplify"],
        "format":      ["format","convert","list","table"],
        "extract":     ["extract","detect","classify","identify"],
    }

    texts = [r["text"].lower() for r in src_f]
    print(f"  Total Source F samples: {len(texts)}")
    for cat, kws in categories.items():
        matches = sum(1 for t in texts if any(k in t for k in kws))
        print(f"  {cat:<15} {matches:>3} samples")

    # Check these don't look like attacks
    risky = [r["text"] for r in src_f
             if any(k in r["text"].lower() for k in
                    ["ignore","override","exfiltrate","bypass","jailbreak"])]
    if risky:
        print(f"\n  ⚠️  {len(risky)} FP-fix samples contain risky keywords:")
        for t in risky:
            print(f"    \"{t[:100]}\"")
    else:
        print(f"\n  ✅ No FP-fix samples contain injection keywords")


# ── Check 8: Overlap with real-world CSV ──────────────────────────────────────
def check_overlap(src_d, src_f):
    print("\n" + "═"*60)
    print("CHECK 8 — OVERLAP WITH REAL-WORLD CSV")
    print("═"*60)

    real = set(load_real_benign())
    if not real:
        print(f"  ⚠️  Real-world CSV not found at {REAL_CSV}")
        return

    print(f"  Real-world CSV: {len(real)} unique descriptions")
    synthetic_texts = [re.sub(r"\s+", " ", r["text"].lower().strip())
                       for r in src_d + src_f]
    overlap = [t for t in synthetic_texts if t in real]
    print(f"  Synthetic samples: {len(synthetic_texts)}")
    print(f"  Overlap (exact):   {len(overlap)}")
    if overlap:
        print(f"  ⚠️  {len(overlap)} exact duplicates found — dedup before merging")
        for t in overlap[:3]:
            print(f"    \"{t[:100]}\"")
    else:
        print(f"  ✅ No exact overlaps with real-world CSV")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(model_results):
    print("\n" + "═"*60)
    print("VALIDATION SUMMARY")
    print("═"*60)
    print("""
  Source D (6,000 synthetic benign)
    Purpose : teach XGBoost what legitimate MCP tool invocations look like
    Covers  : file ops, database, directory, code exec, API calls, MCP tools
    Risk    : low-diversity (template-generated) — mitigated by 6 domains
              + real-world CSV (Source G)

  Source E (~85 hard negatives)
    Purpose : plug 5 specific failure modes found in smoke testing
    Risk    : small N — supplement with adversarial augmentation if recall < 0.80

  Source F (~90 FP fixes)
    Purpose : correct systematic false positives on short imperative text
    Risk    : these are tool USE prompts, not tool DESCRIPTIONS — slightly OOD
              for the MCP tool description classification task
""")
    if model_results:
        fpr_d = model_results["fpr_d"]
        fpr_f = model_results["fpr_f"]
        rec_e = model_results["recall_e"]
        print(f"  Model scores (v2 pkl):")
        print(f"    Source D FPR    = {fpr_d:.4f}  "
              f"({'✅ good' if fpr_d < 0.10 else '⚠️  high — check top FPs'})")
        print(f"    Source F FPR    = {fpr_f:.4f}  "
              f"({'✅ FP fixes working' if fpr_f < 0.15 else '⚠️  FP fixes not fully effective'})")
        print(f"    Source E Recall = {rec_e:.4f}  "
              f"({'✅ good' if rec_e > 0.80 else '⚠️  low recall on hard negatives'})")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("MCP Synthetic Data Quality Validation")
    print("="*60)

    print("Generating Source D (6,000 synthetic benign)...", end=" ", flush=True)
    src_d = generate_source_d()
    print(f"{len(src_d)} records")

    print("Loading  Source E (hard negative attacks)...", end=" ", flush=True)
    src_e = load_source_e()
    print(f"{len(src_e)} records")

    print("Loading  Source F (FP fixes)...", end=" ", flush=True)
    src_f = load_source_f()
    print(f"{len(src_f)} records")

    check_generation(src_d, src_e, src_f)
    check_contamination(src_d, src_f)
    check_diversity(src_d, src_e, src_f)
    check_feature_distributions(src_d, src_e, src_f)
    model_results = check_model_scores(src_d, src_e, src_f)
    check_hard_negative_coverage(src_e)
    check_fp_fix_patterns(src_f)
    check_overlap(src_d, src_f)
    print_summary(model_results)


if __name__ == "__main__":
    main()
