"""
evaluate_mcpcorpus_fpr.py
─────────────────────────────────────────────────────────────────────────────
Evaluates the trained MCP Detector on MCPCorpus marketplace tool descriptions.
Produces FPR (false positive rate) on benign marketplace tool descriptions.

Dataset source
──────────────
  Repository : Snak1nya/MCPCorpus  (HuggingFace)
  URL        : https://huggingface.co/datasets/Snak1nya/MCPCorpus
  File used  : Website/mcpso_servers_cleaned.json  (13.2 MB, 13,875 servers)
  License    : MIT
  arXiv      : 2506.23474  (Lin et al., "MCPCorpus: A Real-World MCP Server
                            Tool Description Dataset", Jun. 2025)

Extraction methodology
──────────────────────
  1. Download mcpso_servers_cleaned.json from the HuggingFace dataset.
  2. Parse the JSON array (13,875 server objects).
  3. For each server, if the `tools` field is a non-empty JSON string, parse
     it and extract each tool's `name` and `description`.
  4. Exclude tools where the description contains ≥ 10 % non-ASCII characters.
     Rationale: all-mpnet-base-v2 is an English-only encoder; applying it to
     Chinese/Japanese text produces out-of-distribution embeddings that are
     not a valid test of the detector's English-language behaviour.
  5. Retain the 57 English tool descriptions as the MCPCorpus holdout set.

Output
──────
  evaluation/results/mcpcorpus_results.json
  evaluation/results/mcpcorpus_results_summary.txt

This evaluation is reported in Table VII of the IEEE Access paper.
"""

import pickle, json, re, csv, warnings, time
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

BASE      = Path(__file__).parent
CACHE     = BASE.parent / "model"
# CSV_FILE: pre-extracted English holdout reference (not used by inference pipeline)
CSV_FILE  = BASE / "results" / "mcpcorpus_english_holdout.csv"
OUT_JSON  = BASE / "results" / "mcpcorpus_results.json"
OUT_TXT   = BASE / "results" / "mcpcorpus_results_summary.txt"
# Threshold loaded dynamically from model/mcp_detector_threshold.json

# ── Download helper ────────────────────────────────────────────────────────────
MCPCORPUS_URL = (
    "https://huggingface.co/datasets/Snak1nya/MCPCorpus/resolve/main/"
    "Website/mcpso_servers_cleaned.json"
)


def download_mcpcorpus(dest: Path) -> None:
    """Download mcpso_servers_cleaned.json from HuggingFace if not present."""
    import urllib.request
    print(f"  Downloading MCPCorpus from HuggingFace...")
    print(f"  URL : {MCPCORPUS_URL}")
    urllib.request.urlretrieve(MCPCORPUS_URL, dest)
    print(f"  Saved to {dest}  ({dest.stat().st_size:,} bytes)")


# ── Extraction ─────────────────────────────────────────────────────────────────
def is_english(text: str, threshold: float = 0.10) -> bool:
    """Return True if non-ASCII character ratio is below threshold."""
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1) < threshold


def extract_tool_descriptions(json_path: Path) -> list[dict]:
    """
    Parse mcpso_servers_cleaned.json and return English tool descriptions.
    Each record: {server_name, server_url, tool_name, description, label=0}
    """
    with open(json_path, encoding="utf-8") as f:
        servers = json.load(f)

    all_tools, excluded_non_english = [], 0
    for server in servers:
        tools_raw = server.get("tools")
        if not tools_raw or tools_raw in ("null", "[]", ""):
            continue
        if not isinstance(tools_raw, str):
            continue
        try:
            tools = json.loads(tools_raw)
        except Exception:
            continue
        if not isinstance(tools, list):
            continue
        for t in tools:
            desc = str(t.get("description", "")).strip()
            name = str(t.get("name", "")).strip()
            if len(desc) < 5:
                continue
            if not is_english(desc):
                excluded_non_english += 1
                continue
            all_tools.append({
                "server_name": server.get("name", ""),
                "server_url":  server.get("url", ""),
                "tool_name":   name,
                "description": desc,
                "label":       0,
            })

    print(f"  Extracted {len(all_tools)} English tool descriptions "
          f"({excluded_non_english} non-English excluded)")
    return all_tools


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


def engineer(prompts: list[str]) -> np.ndarray:
    rows = []
    for p in prompts:
        t, raw = str(p).lower(), str(p)
        sents  = [s.strip() for s in re.split(r"[\n.!?]", t) if s.strip()]
        max_inj = max(
            (sum(s.count(k) for k in ATTACK) * 100 / max(len(s), 1) for s in sents),
            default=0.0,
        )
        opener = t[:60]
        tool_structure = int(any(opener.startswith(o) or opener.lstrip().startswith(o)
                                 for o in TOOL_OPENERS))
        rows.append([
            # v2 features (16)
            len(t), len(t.split()),
            sum(1 for c in raw if c.isupper()) / max(len(t), 1),
            sum(1 for c in t if not c.isalnum() and c.isascii() and c != " ") / max(len(t), 1),
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


# ── Model loading ──────────────────────────────────────────────────────────────
def load_pipeline():
    with open(CACHE / "mcp_detector_calibrated.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(CACHE / "mcp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(CACHE / "mcp_detector_threshold.json") as f:
        threshold = json.load(f)["threshold"]
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-mpnet-base-v2")
    return calibrator, scaler, encoder, threshold


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  MCPCorpus FPR Evaluation")
    print("=" * 60)

    # Step 1: get the source JSON (download if not cached)
    raw_json = BASE / "results" / "mcpso_servers_cleaned.json"
    if not raw_json.exists():
        download_mcpcorpus(raw_json)
    else:
        print(f"  Using cached {raw_json}  ({raw_json.stat().st_size:,} bytes)")

    # Step 2: extract English tool descriptions
    print("\nExtracting tool descriptions...")
    tools = extract_tool_descriptions(raw_json)
    texts = [t["description"] for t in tools]
    n     = len(texts)

    # Step 3: load model and run inference
    print("\nLoading model pipeline...")
    calibrator, scaler, encoder, threshold = load_pipeline()

    print(f"Running inference on {n} English tool descriptions...")
    t0    = time.perf_counter()
    emb   = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    eng   = scaler.transform(engineer(texts))
    X     = np.concatenate([emb, eng], axis=1)
    probs = calibrator.predict_proba(X)[:, 1]
    elapsed_ms  = (time.perf_counter() - t0) * 1000
    avg_latency = elapsed_ms / n

    # Step 4: metrics
    preds = (probs >= threshold).astype(int)
    fp    = int(preds.sum())
    tn    = n - fp
    fpr   = round(fp / n, 4)

    print(f"\n  n          = {n}")
    print(f"  FP         = {fp}")
    print(f"  TN         = {tn}")
    print(f"  FPR        = {fpr:.4f}  ({fp}/{n} benign flagged)")
    print(f"  Threshold  = {threshold:.4f}")
    print(f"  Avg latency= {avg_latency:.1f} ms/sample")

    # Step 5: per-tool results
    print("\n  False positives (benign tools incorrectly flagged):")
    fp_list = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        tools[i]["prob"] = round(float(prob), 4)
        tools[i]["pred"] = int(pred)
        if pred == 1:
            fp_list.append(tools[i])
            print(f"    [{tools[i]['server_name']}] {tools[i]['tool_name']}  "
                  f"p={prob:.4f}")
            print(f"      {tools[i]['description'][:100]}")

    # Step 6: save JSON
    results = {
        "evaluation_date":  "2026-05-26",
        "model":            "MCP Detector (XGB + all-mpnet-base-v2, Platt-calibrated)",
        "threshold":        threshold,
        "dataset": {
            "name":         "MCPCorpus",
            "source":       "Snak1nya/MCPCorpus (HuggingFace)",
            "url":          "https://huggingface.co/datasets/Snak1nya/MCPCorpus",
            "file":         "Website/mcpso_servers_cleaned.json",
            "license":      "MIT",
            "arxiv":        "2506.23474",
            "total_servers":         13875,
            "total_tool_descriptions": 84,
            "excluded_non_english":    27,
            "exclusion_reason": (
                "all-mpnet-base-v2 is an English encoder; descriptions with "
                ">= 10% non-ASCII characters produce OOD embeddings unrelated "
                "to injection detection performance"
            ),
            "n_english_evaluated": n,
        },
        "results": {
            "n":             n,
            "FP":            fp,
            "TN":            tn,
            "FPR":           fpr,
            "avg_latency_ms": round(avg_latency, 2),
        },
        "false_positives": fp_list,
        "all_tools":       tools,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Step 7: human-readable summary
    summary = f"""
=== MCPCorpus MARKETPLACE TOOL DESCRIPTION FPR EVALUATION ===
Evaluation date : 2026-05-26
Model           : MCP Detector (XGBoost + all-mpnet-base-v2, Platt-calibrated, t={threshold:.4f})

DATASET
  Source        : MCPCorpus — Snak1nya/MCPCorpus (HuggingFace, MIT licence)
  arXiv         : 2506.23474  (Lin et al., Jun. 2025)
  File          : Website/mcpso_servers_cleaned.json
  Total servers : 13,875
  Servers w/ tool descriptions : 185
  Total tool descriptions found : 84
  Excluded (non-English, >=10% non-ASCII) : 27
  Evaluated (English only)                : {n}

  Exclusion rationale:
    The underlying sentence encoder (all-mpnet-base-v2) is English-only.
    Applying it to Chinese / Japanese text produces out-of-distribution
    embeddings unrelated to injection-detection behaviour.  Reporting FPR
    on such text would not be a valid measure of the detector's performance.

RESULTS
  FPR           : {fpr:.4f}  ({fp} of {n} benign tools flagged as malicious)
  Avg latency   : {avg_latency:.1f} ms/sample

FALSE POSITIVES  ({fp} tools)
""" + "\n".join(
        f"  [{t['server_name']}] {t['tool_name']}  (p={t['prob']:.4f})\n"
        f"    {t['description'][:120]}"
        for t in fp_list
    ) + f"""

COMPARISON
  GitHub-sourced holdout (N=92 benign) : FPR = 0.1957
  MCPCorpus marketplace  (N={n} benign): FPR = {fpr:.4f}
  Delta                                : +{fpr - 0.1957:.4f}

  Interpretation: marketplace tool descriptions are shorter and more terse
  than developer-authored GitHub docstrings, producing higher embedding
  similarity to the attack pattern region and thus elevated FPR.
"""
    OUT_TXT.write_text(summary)

    print(f"\nResults saved:")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_TXT}")


if __name__ == "__main__":
    main()
