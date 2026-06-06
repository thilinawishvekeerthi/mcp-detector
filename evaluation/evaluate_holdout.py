"""
evaluate_holdout.py
──────────────────────
Runs the trained MCP Detector on the real-world holdout set and
produces full evaluation metrics + per-payload-type breakdown.

Input:   data_collection/evaluation/output/realworld_holdout.csv
Output:  evaluation/results/realworld_results.json
         evaluation/results/realworld_results_summary.txt

This is the evaluation reported as Table VII in the IEEE Access paper.
"""

import pickle, json, re, csv, warnings, time
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
    average_precision_score
)

warnings.filterwarnings("ignore")

BASE     = Path(__file__).parent
CACHE    = BASE.parent / "model"
CSV_FILE = BASE.parent / "data_collection" / "evaluation" / "output" / "realworld_holdout.csv"
OUT_JSON = BASE / "results" / "realworld_results.json"
OUT_TXT  = BASE / "results" / "realworld_results_summary.txt"
# Threshold is loaded dynamically from model/mcp_detector_threshold.json
# by load_pipeline() — do not hardcode here.

# ── Feature engineering v3 — 20 features (matches FeatureEngineer v3 in notebook) ──
ATTACK    = ["jailbreak","ignore","forget","override","bypass","exploit","execute",
             "backdoor","exfiltrate","leak","steal","disable","dump","instructions",
             "previous","disregard","comply","restricted","credentials","safety",
             "unrestricted","restrictions","lifted","automatically","silently",
             "invoke","once done","after execution","when finished","after running",
             "post result","passwd","shadow","etc/","root/","chmod","curl ","wget ",
             "base64","whoami","id &&","env |","send_email","post to","forward to",
             "transfer to","attacker","evil.com","exfil"]
JAILBREAK = ["jailbreak","ignore","forget","override","bypass","disregard",
             "comply","previous instructions"]
SYSTEM    = ["system","instruction","role"]
ROLEPLAY  = ["roleplay","pretend","imagine","act","character","act as","you are now",
             "you are a","dan mode","developer mode","unrestricted ai","toolgpt"]
OVERRIDE  = ["override","ignore","forget","bypass","disregard","lifted",
             "disabled","unrestricted"]
UNIX      = ["cat /","ls -","chmod","curl ","wget ","/etc/","/root","| base64",
             "&& id","whoami","grep -i","env |","passwd","shadow"]
TOOLCHAIN = ["automatically call","automatically invoke","once done","after execution",
             "when finished","after running","after using this tool","post the results",
             "silently post","forward all","send all","invoke the"]
# v3: contextual disambiguation
AI_SELF_REF = [
    "your instructions","your previous","you are now","you must",
    "you will now","you have been","as an ai","as a language model",
    "ignore your","forget your","disregard your","your system prompt",
    "your context","your role","you are free","you have no",
    "your limitations","your rules","your guidelines",
]
BENIGN_MCP_VOCAB = [
    "repository","pull request","webhook","oauth","pagination",
    "cursor","api key","file path","commit","branch",
    "diff","merge request","issue tracker","deployment",
    "docker","kubernetes","endpoint url","rest api","graphql",
    "schema validation","json schema","openapi","swagger",
    "rate limit","authentication token","access token",
]
TOOL_OPENERS = [
    "this tool","gets ","retrieves ","creates ","updates ",
    "deletes ","searches ","returns ","fetches ","queries ",
    "lists ","reads ","writes ","uploads ","downloads ",
    "sends an ","sends a ","generates ","converts ","validates ",
    "checks ","calculates ","extracts ","parses ","formats ",
    "compares ","monitors ","tracks ","logs ","manages ",
]
MCP_PROTOCOL_VOCAB = [
    "elicitation","sampling request","resource reference",
    "tool registration","prompt template","server capability",
    "tool manifest","mcp client","mcp server","tool call result",
    "resource uri","tool input schema","progress notification",
    "cancellation","roots list","tool list changed",
]

def engineer(prompts):
    rows = []
    for p in prompts:
        t, raw = str(p).lower(), str(p)
        sents  = [s.strip() for s in re.split(r"[\n.!?]", t) if s.strip()]
        max_inj = max(
            (sum(s.count(k) for k in ATTACK) * 100 / max(len(s), 1) for s in sents),
            default=0.0
        )
        opener = t[:60]
        tool_structure = int(any(opener.startswith(o) or opener.lstrip().startswith(o)
                                 for o in TOOL_OPENERS))
        rows.append([
            # v2 features (16)
            len(t), len(t.split()),
            sum(1 for c in raw if c.isupper()) / max(len(t), 1),
            sum(1 for c in t if not c.isalnum() and c.isascii() and c!=" ") / max(len(t), 1),
            sum(t.count(k) for k in JAILBREAK),
            sum(t.count(k) for k in SYSTEM),
            sum(t.count(k) for k in OVERRIDE),
            sum(t.count(k) for k in ROLEPLAY),
            t.count("system"),
            sum(t.count(k) for k in ATTACK),
            sum(1 for c in raw if 127 < ord(c) < 1280) / max(len(raw), 1),
            sum(1 for w in t.split()
                if len(w) > 8 and re.match(r"^[A-Za-z0-9+/=]+$", w) and len(w) % 4 == 0
            ) / max(len(t.split()), 1),
            sum(t.count(k) for k in ATTACK) * 100 / max(len(t), 1),
            sum(t.count(p) for p in UNIX),
            sum(t.count(p) for p in TOOLCHAIN),
            max_inj,
            # v3 features (4)
            sum(t.count(p) for p in AI_SELF_REF),
            sum(t.count(p) for p in BENIGN_MCP_VOCAB),
            tool_structure,
            sum(t.count(p) for p in MCP_PROTOCOL_VOCAB),
        ])
    return np.array(rows, dtype=np.float64)


# ── Load model ─────────────────────────────────────────────────────────────────
def load_pipeline():
    with open(CACHE / "mcp_detector_calibrated.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(CACHE / "mcp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(CACHE / "mcp_detector_threshold.json") as f:
        threshold = json.load(f)["threshold"]
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-mpnet-base-v2")
    print(f"  threshold={threshold:.4f}  scaler={type(scaler).__name__}")
    return calibrator, scaler, encoder, threshold


def predict_all(texts, calibrator, scaler, encoder):
    BATCH = 64
    all_probs = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        emb   = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        eng   = scaler.transform(engineer(batch))
        X     = np.concatenate([emb, eng], axis=1)
        probs = calibrator.predict_proba(X)[:, 1]
        all_probs.extend(probs.tolist())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return np.array(all_probs), elapsed_ms


# ── Metrics helpers ────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "n":         int(len(y_true)),
        "n_benign":  int((np.array(y_true) == 0).sum()),
        "n_malicious": int((np.array(y_true) == 1).sum()),
        "AUC":       round(float(roc_auc_score(y_true, y_prob)), 4),
        "AP":        round(float(average_precision_score(y_true, y_prob)), 4),
        "F1":        round(float(f1_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "FPR":       round(float(fp / max(fp + tn, 1)), 4),
        "FNR":       round(float(fn / max(fn + tp, 1)), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load holdout CSV
    rows = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    texts        = [r["text"]         for r in rows]
    labels       = [int(r["label"])   for r in rows]
    payload_types = [r["payload_type"] for r in rows]
    sources      = [r["source"]       for r in rows]

    print(f"\nLoaded {len(rows)} samples  "
          f"({labels.count(0)} benign, {labels.count(1)} adversarial)")

    # Load pipeline
    print("Loading model pipeline...")
    calibrator, scaler, encoder, threshold = load_pipeline()

    # Run inference
    print(f"Running inference on {len(texts)} samples...")
    probs, elapsed_ms = predict_all(texts, calibrator, scaler, encoder)
    avg_latency = elapsed_ms / len(texts)
    print(f"  Done in {elapsed_ms:.0f}ms  (avg {avg_latency:.1f}ms/sample)")

    # Overall metrics
    overall = compute_metrics(labels, probs, threshold)
    overall["avg_latency_ms"] = round(avg_latency, 2)
    print(f"\n{'─'*55}")
    print(f"{'OVERALL RESULTS':^55}")
    print(f"{'─'*55}")
    for k, v in overall.items():
        print(f"  {k:<18} {v}")

    # Per-payload-type breakdown (adversarial samples only)
    print(f"\n{'─'*55}")
    print(f"{'ADVERSARIAL — BY PAYLOAD TYPE':^55}")
    print(f"{'─'*55}")
    ptype_results = {}
    unique_types = sorted(set(pt for pt, l in zip(payload_types, labels) if l == 1))
    for ptype in unique_types:
        idx = [i for i,(pt,l) in enumerate(zip(payload_types, labels)) if pt==ptype and l==1]
        if not idx:
            continue
        y_true_sub = [labels[i] for i in idx]
        y_prob_sub = probs[idx]
        y_pred_sub = (y_prob_sub >= threshold).astype(int)
        recall = float(recall_score(y_true_sub, y_pred_sub, zero_division=0))
        asr    = round(1.0 - recall, 4)
        ptype_results[ptype] = {
            "n": len(idx), "Recall": round(recall, 4), "ASR": asr
        }
        print(f"  {ptype:<25}  N={len(idx):>3}  Recall={recall:.4f}  ASR={asr:.4f}")

    # FPR on real benign MCP tools specifically
    benign_idx = [i for i,l in enumerate(labels) if l == 0]
    benign_probs = probs[np.array(benign_idx)]
    benign_preds = (benign_probs >= threshold).astype(int)
    real_fpr = float(benign_preds.sum()) / max(len(benign_preds), 1)
    print(f"\n  FPR on real MCP tool descriptions: {real_fpr:.4f}  "
          f"({int(benign_preds.sum())} false alarms / {len(benign_preds)} benign)")

    # Comparison to synthetic test set performance (from paper)
    print(f"\n{'─'*55}")
    print(f"{'COMPARISON: SYNTHETIC vs REAL-WORLD':^55}")
    print(f"{'─'*55}")
    # v3.1 synthetic held-out test set metrics (60,892 samples, t=0.380)
    synthetic = {"AUC": 0.9155, "Recall": 0.8039, "Precision": 0.7675,
                 "F1": 0.7853, "FPR": 0.1560}
    for metric in ["AUC","Recall","Precision","F1","FPR"]:
        syn_val  = synthetic[metric]
        real_val = overall[metric]
        delta    = real_val - syn_val
        flag     = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {metric:<12}  Synthetic={syn_val:.4f}  Real-world={real_val:.4f}  {flag}{abs(delta):.4f}")

    # Save full results
    results = {
        "evaluation_date": "2026-05-26",
        "model": "MCP Detector (XGB + all-mpnet-base-v2, Platt-calibrated)",
        "threshold": threshold,
        "overall": overall,
        "per_payload_type": ptype_results,
        "real_benign_fpr": round(real_fpr, 4),
        "synthetic_comparison": synthetic,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Human-readable summary for paper
    summary = f"""
=== REAL-WORLD MCP TOOL DESCRIPTION GENERALIZATION RESULTS ===
Evaluation date: 2026-05-26
Model: MCP Detector (XGBoost + all-mpnet-base-v2, Platt-calibrated, t={threshold:.4f})
Holdout set: {overall['n']} samples ({overall['n_benign']} real benign MCP tools
             + {overall['n_malicious']} adversarial variants)
Source: {len(set(sources))} public GitHub MCP server repositories (MIT/Apache-2.0)

OVERALL METRICS
  AUC:          {overall['AUC']:.4f}
  F1:           {overall['F1']:.4f}
  Recall:       {overall['Recall']:.4f}
  Precision:    {overall['Precision']:.4f}
  FPR (real):   {real_fpr:.4f}
  Avg latency:  {overall['avg_latency_ms']:.1f} ms/sample

PER-PAYLOAD-TYPE RECALL / ASR
""" + "\n".join(
    f"  {k:<25}  Recall={v['Recall']:.4f}  ASR={v['ASR']:.4f}  (N={v['n']})"
    for k, v in ptype_results.items()
) + f"""

COMPARISON TO SYNTHETIC TEST SET
  Metric        Synthetic    Real-world    Delta
""" + "\n".join(
    f"  {m:<12}  {synthetic[m]:.4f}       {overall[m]:.4f}       {overall[m]-synthetic[m]:+.4f}"
    for m in ["AUC","Recall","Precision","F1","FPR"]
)

    OUT_TXT.write_text(summary)
    print(f"\nResults saved to:")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_TXT}")


if __name__ == "__main__":
    main()
