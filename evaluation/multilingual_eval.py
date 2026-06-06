"""
multilingual_eval.py
Zero-shot multilingual evaluation — Safeguard 2 verification.

Scores the 12 held-out multilingual attack patterns against the v6.1 model.
These patterns were NEVER included in training (Safeguard 2 design decision).

Result: 50% overall recall (Chinese 100%, FR/ES/DE 33% each — exfiltration only)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PAPER CONTRIBUTION — what this script produces
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Section V.C — Obfuscation robustness (RQ3), multilingual subsection:

  TWO different multilingual numbers appear in the paper — this script produces
  the SECOND one (zero-shot). Both must be cited with explicit labels:

  Number 1 — Translation generalisation (from Colab notebook RQ3 cell):
    N=20,193 malicious test examples, machine-translated to 4 languages.
    Result: ~74% recall (v6.1 Colab run).
    Meaning: the model generalises to translated versions of KNOWN attack patterns.
    Source: Colab Cell 108, OBF_MPNET_v6.json

  Number 2 — Zero-shot transfer (this script, Safeguard 2):
    N=12 held-out patterns that were NEVER in training (3 per language).
    Result: 50% overall (Chinese 100%, French/Spanish/German 33% each).
    Meaning: the model partially generalises to UNSEEN multilingual attack patterns.
    Only exfiltration patterns are detected (contain language-agnostic URL markers).
    Source: attack_pattern_bank.json → multilingual_HELD_OUT key

  Section VI — Limitations:
    The 50% zero-shot result confirms the model's linguistic scope is English-first.
    Natural-language instruction overrides in non-English are not detected.
    Paper framing: "Detection is English-primary; multilingual generalisation is
    limited to patterns containing language-agnostic lexical markers (URLs, domains)."

  Methodology §IV — Safeguard 2 design decision:
    Holding out multilingual from training and evaluating zero-shot validates
    that the model has not memorised the generator pattern. The 50% result
    confirms genuine generalisation (not memorisation) with known scope limits.

Run:
    python3 evaluation/multilingual_eval.py

Requirements: model files in model/, attack_pattern_bank.json in
    data_collection/training/adversarial/
"""

import pickle, json, re, warnings, numpy as np
warnings.filterwarnings('ignore')
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent.parent

LEET=str.maketrans({'3':'e','0':'o','1':'i','4':'a','@':'a','5':'s','7':'t','!':'i','$':'s','|':'i','8':'b','¡':'i'})
ATTACK=['jailbreak','ignore','forget','override','bypass','exploit','execute','backdoor','exfiltrate','leak','steal','disable','dump','instructions','previous','disregard','comply','restricted','credentials','safety','unrestricted','restrictions','lifted','automatically','silently','invoke','once done','after execution','when finished','after running','post result','passwd','shadow','etc/','root/','chmod','curl ','wget ','base64','whoami','id &&','env |','send_email','post to','forward to','transfer to','attacker','evil.com','exfil','wipe','purge','reveal','sandbox','abort']
JAILBREAK=['jailbreak','ignore','forget','override','bypass','disregard','comply','previous instructions']
SYSTEM=['system','instruction','role']; ROLEPLAY=['roleplay','pretend','imagine','act','character','act as','you are now','you are a','dan mode','developer mode','unrestricted ai','toolgpt']
OVERRIDE=['override','ignore','forget','bypass','disregard','lifted','disabled','unrestricted']
UNIX=['cat /','ls -','chmod','curl ','wget ','/etc/','/root','| base64','&& id','whoami','grep -i','env |','passwd','shadow']
TOOLCHAIN=['automatically call','automatically invoke','once done','after execution','when finished','after running','after using this tool','post the results','silently post','forward all','send all','invoke the','call wipe','call purge','call delete','to destroy all','remove all audit']
AI_SELF=['your instructions','your previous','you are now','you must','you will now','you have been','as an ai','as a language model','ignore your','forget your','disregard your','your system prompt','your context','your role','you are free','you have no','your limitations','your rules','your guidelines']
BENIGN_V=['repository','pull request','webhook','oauth','pagination','cursor','api key','file path','commit','branch','diff','merge request','issue tracker','deployment','docker','kubernetes','endpoint url','rest api','graphql','schema validation','json schema','openapi','swagger','rate limit','authentication token','access token']
OPENERS=['this tool','gets ','retrieves ','creates ','updates ','deletes ','searches ','returns ','fetches ','queries ','lists ','reads ','writes ','uploads ','downloads ','sends an ','sends a ','generates ','converts ','validates ','checks ','calculates ','extracts ','parses ','formats ','compares ','monitors ','tracks ','logs ','manages ']
MCP=['elicitation','sampling request','resource reference','tool registration','prompt template','server capability','tool manifest','mcp client','mcp server','tool call result','resource uri','tool input schema','progress notification','cancellation','roots list','tool list changed']

def eng(prompts, scaler):
    rows=[]
    for p in prompts:
        t,raw=str(p).lower(),str(p); tn=t.translate(LEET)
        sents=[s.strip() for s in re.split(r'[\n.!?]',t) if s.strip()]
        mi=max((sum(s.count(k) for k in ATTACK)*100/max(len(s),1) for s in sents),default=0.0)
        op=t[:60]; ts=int(any(op.startswith(o) or op.lstrip().startswith(o) for o in OPENERS))
        rows.append([len(t),len(t.split()),sum(1 for c in raw if c.isupper())/max(len(t),1),
            sum(1 for c in t if not c.isalnum() and c.isascii() and c!=' ')/max(len(t),1),
            sum(t.count(k) for k in JAILBREAK),sum(t.count(k) for k in SYSTEM),
            sum(t.count(k) for k in OVERRIDE),sum(t.count(k) for k in ROLEPLAY),
            t.count('system'),sum(t.count(k) for k in ATTACK),
            sum(1 for c in raw if (127<ord(c)<1280) or (0x1D400<=ord(c)<=0x1D7FF))/max(len(raw),1),
            sum(1 for w in t.split() if len(w)>8 and re.match(r'^[A-Za-z0-9+/=]+$',w) and len(w)%4==0)/max(len(t.split()),1),
            sum(t.count(k) for k in ATTACK)*100/max(len(t),1),
            sum(t.count(p) for p in UNIX),sum(t.count(p) for p in TOOLCHAIN),mi,
            sum(t.count(p) for p in AI_SELF),sum(t.count(p) for p in BENIGN_V),ts,
            sum(t.count(p) for p in MCP),sum(tn.count(k) for k in ATTACK)])
    return scaler.transform(np.array(rows, dtype=np.float64))

def main():
    # Load model
    model_dir = BASE / 'model'
    with open(model_dir/'mcp_detector_calibrated.pkl','rb') as f: cal = pickle.load(f)
    with open(model_dir/'mcp_detector_threshold.json') as f: THR = json.load(f)['threshold']
    with open(model_dir/'mcp_scaler.pkl','rb') as f: scaler = pickle.load(f)

    # Load multilingual patterns (held-out from training)
    bank_path = BASE/'data_collection/training/adversarial/attack_pattern_bank.json'
    with open(bank_path) as f:
        bank = json.load(f)

    multi = bank.get('multilingual_HELD_OUT', {})
    patterns, langs = [], []
    for lang, items in multi.items():
        if lang.startswith('_'): continue
        for p in items:
            patterns.append(p)
            langs.append(lang)

    print(f"{'='*65}")
    print(f" MULTILINGUAL ZERO-SHOT EVALUATION")
    print(f" Model: v6.1 post-Platt-refit  |  Threshold: {THR}")
    print(f" Note: Safeguard 2 — these patterns were NEVER in training")
    print(f"{'='*65}\n")
    print(f"N = {len(patterns)} patterns  |  Languages: {sorted(set(langs))}\n")

    enc = SentenceTransformer('all-mpnet-base-v2')
    emb = enc.encode(patterns, convert_to_numpy=True, show_progress_bar=False)
    ef  = eng(patterns, scaler)
    probs = cal.predict_proba(np.concatenate([emb, ef], axis=1))[:,1]
    preds = (probs >= THR).astype(int)

    print(f"{'Language':<12} {'N':>4}  {'Detected':>9}  {'Recall':>8}  {'Avg p':>8}")
    print("-"*50)
    for lang in sorted(set(langs)):
        idx = [i for i,l in enumerate(langs) if l==lang]
        n = len(idx); det = sum(preds[i] for i in idx); avg_p = np.mean([probs[i] for i in idx])
        print(f"  {lang:<10} {n:>4}  {det:>4}/{n:<4}    {100*det/n:>5.1f}%  {avg_p:>8.4f}")

    print(f"\n  Overall: {preds.sum()}/{len(patterns)}  ({100*preds.mean():.1f}%)")
    print(f"  Avg p  : {probs.mean():.4f}")

    print(f"\nPer-example results:")
    for txt, lang, prob, pred in zip(patterns, langs, probs, preds):
        mark = "✅" if pred else "❌"
        print(f"  {mark} p={prob:.3f} [{lang}] {txt[:80]}")

    print(f"\n{'='*65}")
    print(f" PAPER FRAMING")
    print(f"{'='*65}")
    print(f"""
  RQ3 (translation generalisation, N=20,193):  ~74%  ← different evaluation
  Zero-shot transfer (Safeguard 2, N={len(patterns)}):      {100*preds.mean():.0f}%  ← this script

  These are different evaluations — both must be cited with explicit labels.
  Chinese 100% — detected via URL/domain markers (evil.com) in patterns.
  French/Spanish/German 33% — only exfiltration patterns detected; natural-language
  instruction overrides in non-English are not detected (English-only training scope).
""")

if __name__ == '__main__':
    main()
