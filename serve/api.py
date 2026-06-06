# api.py — MCP Injection Detector  (serve/api.py)
import pickle, json, re, time
from pathlib import Path
from typing import Optional, List

# Model directory — sibling of serve/, resolved relative to this file
MODEL_DIR = Path(__file__).parent.parent / "model"

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MCP Injection Detector", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight(rest_of_path: str):
    from fastapi.responses import Response
    r = Response()
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    r.headers["Access-Control-Allow-Headers"] = "*"
    return r

# ── Feature engineering v4 (21 features — matches trained model) ──────────────
# v4 changes vs v3:
#   FIX 1: ATTACK + TOOLCHAIN lists extended (wipe/purge/reveal/sandbox/abort)
#   FIX 2: homoglyph_ratio extended to cover mathematical unicode (0x1D400-0x1D7FF)
#   FIX 3: Feature 20 added — leet-normalised attack count (fixes leetspeak blind spot)
# Keyword lists mirror evaluate_holdout.py exactly.
ATTACK    = ["jailbreak","ignore","forget","override","bypass","exploit","execute",
             "backdoor","exfiltrate","leak","steal","disable","dump","instructions",
             "previous","disregard","comply","restricted","credentials","safety",
             "unrestricted","restrictions","lifted","automatically","silently",
             "invoke","once done","after execution","when finished","after running",
             "post result","passwd","shadow","etc/","root/","chmod","curl ","wget ",
             "base64","whoami","id &&","env |","send_email","post to","forward to",
             "transfer to","attacker","evil.com","exfil",
             # v4 additions — audit-driven (Source H blind-spot analysis 2026-05-30)
             "wipe","purge","reveal","sandbox","abort"]
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
             "silently post","forward all","send all","invoke the",
             # v4 additions — destructive tool-chaining patterns
             "call wipe","call purge","call delete","to destroy all","remove all audit"]
# Leet-substitution normalisation table — used for Feature 20
# Maps common digit/symbol substitutes back to ASCII letters
LEET_TABLE = str.maketrans({
    '3': 'e', '0': 'o', '1': 'i', '4': 'a', '@': 'a',
    '5': 's', '7': 't', '!': 'i', '$': 's', '|': 'i',
    '8': 'b', '¡': 'i',
})
# v3 contextual disambiguation features
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

# 21 feature names (v2: 0-15, v3: 16-19, v4: 20)
FEATURE_NAMES = [
    "prompt_length","word_count","uppercase_ratio","special_char_ratio",
    "jailbreak_count","instruction_count","override_count","roleplay_count",
    "system_count","attack_pattern_count","homoglyph_ratio",
    "base64_content_ratio","injection_signal_density",
    "unix_cmd_count","tool_chain_count","max_sentence_injection",
    # v3 features (16-19)
    "ai_self_ref","benign_mcp_vocab","tool_structure","mcp_proto_vocab",
    # v4 feature (20) — leet-normalised attack count
    "leet_norm_attack",
]


def _engineer(prompts: list) -> np.ndarray:
    """Return (N, 21) feature matrix — v4, matches trained model.

    Input vector: 768-d embedding + 21 engineered = 789-d total.

    v4 changes vs v3:
      - Feature 10 (homoglyph_ratio): extended range covers mathematical unicode (0x1D400-0x1D7FF)
      - Feature 20 (leet_norm_attack): attack keyword count on leet-normalised text
      - ATTACK list: +wipe, +purge, +reveal, +sandbox, +abort
      - TOOLCHAIN list: +call wipe/purge/delete, +to destroy all, +remove all audit
    """
    rows = []
    for p in prompts:
        t, raw = str(p).lower(), str(p)
        t_norm  = t.translate(LEET_TABLE)          # leet-normalised text for Feature 20
        sents   = [s.strip() for s in re.split(r"[\n.!?]", t) if s.strip()]
        max_inj = max(
            (sum(s.count(k) for k in ATTACK) * 100 / max(len(s), 1) for s in sents),
            default=0.0,
        )
        opener = t[:60]
        tool_structure = int(any(
            opener.startswith(o) or opener.lstrip().startswith(o)
            for o in TOOL_OPENERS
        ))
        rows.append([
            # v2 features (0-15)
            len(t), len(t.split()),
            sum(1 for c in raw if c.isupper()) / max(len(t), 1),
            sum(1 for c in t if not c.isalnum() and c.isascii() and c != " ") / max(len(t), 1),
            sum(t.count(k) for k in JAILBREAK),
            sum(t.count(k) for k in SYSTEM),
            sum(t.count(k) for k in OVERRIDE),
            sum(t.count(k) for k in ROLEPLAY),
            t.count("system"),
            sum(t.count(k) for k in ATTACK),
            # Feature 10: homoglyph_ratio — FIX: extended to cover mathematical unicode
            sum(1 for c in raw if (127 < ord(c) < 1280)
                or (0x1D400 <= ord(c) <= 0x1D7FF)) / max(len(raw), 1),
            sum(1 for w in t.split()
                if len(w) > 8 and re.match(r"^[A-Za-z0-9+/=]+$", w) and len(w) % 4 == 0
            ) / max(len(t.split()), 1),
            sum(t.count(k) for k in ATTACK) * 100 / max(len(t), 1),
            sum(t.count(p) for p in UNIX),
            sum(t.count(p) for p in TOOLCHAIN),
            max_inj,
            # v3 features (16-19)
            sum(t.count(p) for p in AI_SELF_REF),
            sum(t.count(p) for p in BENIGN_MCP_VOCAB),
            tool_structure,
            sum(t.count(p) for p in MCP_PROTOCOL_VOCAB),
            # v4 feature (20): leet-normalised attack count — FIX: covers leetspeak blind spot
            sum(t_norm.count(k) for k in ATTACK),
        ])
    return np.array(rows, dtype=np.float64)


# ── Global state ──────────────────────────────────────────────
_calibrator = None
_scaler     = None
_emb_model  = None
THRESHOLD   = 0.380          # matches mcp_detector_threshold.json
MODEL_NAME  = "all-mpnet-base-v2"


def _build_vector(prompt: str) -> np.ndarray:
    emb = _emb_model.encode([prompt], convert_to_numpy=True)  # (1, 768)
    eng = _engineer([prompt])                                  # (1, 20)
    if _scaler is not None:
        eng = _scaler.transform(eng)
    return np.concatenate([emb, eng], axis=1)                  # (1, 788)


@app.on_event("startup")
def load_model():
    global _calibrator, _scaler, _emb_model, THRESHOLD
    with open(MODEL_DIR / "mcp_detector_calibrated.pkl", "rb") as f:
        _calibrator = pickle.load(f)
    with open(MODEL_DIR / "mcp_detector_threshold.json") as f:
        THRESHOLD = json.load(f)["threshold"]
    sp = MODEL_DIR / "mcp_scaler.pkl"
    if sp.exists():
        with open(sp, "rb") as f:
            _scaler = pickle.load(f)
    from sentence_transformers import SentenceTransformer
    _emb_model = SentenceTransformer(MODEL_NAME)
    print(f"[startup] model={MODEL_NAME}  threshold={THRESHOLD:.4f}  scaler={_scaler is not None}")


# ── Schemas ───────────────────────────────────────────────────
class ToolDescription(BaseModel):
    prompt:    str
    tool_name: Optional[str] = "unknown"

class ExplainItem(BaseModel):
    feature:    str
    shap_value: float
    direction:  str

class ExplainResult(BaseModel):
    tool_name:   str
    action:      str
    probability: float
    latency_ms:  float
    flagged:     bool
    explanation: List[ExplainItem]


# ── GET /health ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME,
            "threshold": THRESHOLD, "scaler": _scaler is not None}


# ── POST /detect ──────────────────────────────────────────────
@app.post("/detect")
def detect(req: ToolDescription):
    if _calibrator is None:
        raise HTTPException(503, "Model not loaded")
    t0   = time.perf_counter()
    prob = float(_calibrator.predict_proba(_build_vector(req.prompt))[0][1])
    pred = int(prob >= THRESHOLD)
    return {
        "tool_name":   req.tool_name,
        "action":      "BLOCK" if pred else "ALLOW",
        "decision":    "MALICIOUS" if pred else "BENIGN",
        "probability": prob,
        "latency_ms":  (time.perf_counter() - t0) * 1000,
        "flagged":     bool(pred),
    }


# ── POST /detect/batch ────────────────────────────────────────
@app.post("/detect/batch")
def detect_batch(tools: List[ToolDescription]):
    if _calibrator is None:
        raise HTTPException(503, "Model not loaded")
    return [detect(t) for t in tools]


# ── POST /detect/explain ──────────────────────────────────────
@app.post("/detect/explain", response_model=ExplainResult)
def detect_explain(req: ToolDescription):
    import shap
    if _calibrator is None:
        raise HTTPException(503, "Model not loaded")
    t0   = time.perf_counter()
    fts  = _build_vector(req.prompt)                    # (1, 788)
    prob = float(_calibrator.predict_proba(fts)[0][1])
    pred = int(prob >= THRESHOLD)
    xgb  = _calibrator.calibrated_classifiers_[0].estimator
    sv   = shap.TreeExplainer(xgb).shap_values(fts, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    eng_sv = sv[0][-len(FEATURE_NAMES):]   # last 20 dims
    pairs  = sorted(zip(FEATURE_NAMES, eng_sv),
                    key=lambda x: abs(x[1]), reverse=True)
    return ExplainResult(
        tool_name   = req.tool_name,
        action      = "BLOCK" if pred else "ALLOW",
        probability = prob,
        latency_ms  = (time.perf_counter() - t0) * 1000,
        flagged     = bool(pred),
        explanation = [
            ExplainItem(
                feature    = feat,
                shap_value = round(float(val), 4),
                direction  = "MALICIOUS" if val > 0 else "BENIGN",
            ) for feat, val in pairs[:5]
        ],
    )
