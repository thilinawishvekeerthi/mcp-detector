"""
run_holdout_split.py  (v2 -- with MAX_HOLDOUT_ROWS_PER_REPO cap)
Produces a stratified, repo-level holdout for fine-tuning evaluation.

Cap behaviour (MAX_HOLDOUT_ROWS_PER_REPO = 50):
  - Repos with <= 50 rows: all rows go entirely to ONE side.
    Strict repo-level split preserved for small repos.
  - Repos with > 50 rows: 50 randomly sampled rows go to holdout,
    remainder go to training. Same repo appears in both splits,
    but only 50 rows are ever in the holdout side.

Outputs:
    pass1_finetune_holdout.csv
    pass1_finetune_train_pool.csv
    pass1_finetune_split_summary.txt
"""

import re, random
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---- Config ------------------------------------------------------------------
SEED                      = 42
MAX_HOLDOUT_ROWS_PER_REPO = 50

# ---- Paths -------------------------------------------------------------------
BASE      = Path('/Users/thilina/Documents/mcp-detector')
TRAIN_OUT = BASE / 'data_collection/training/output'
EVAL_OUT  = BASE / 'data_collection/evaluation/output'
RESULTS   = BASE / 'evaluation/results'

CANDIDATE_CLEAN = TRAIN_OUT / 'mcp_benign_realworld_holdout_CLEAN.csv'
EVAL_N154       = EVAL_OUT  / 'realworld_holdout.csv'
EVAL_MCPC       = RESULTS   / 'mcpcorpus_english_holdout.csv'

OUT_HOLDOUT = TRAIN_OUT / 'pass1_finetune_holdout.csv'
OUT_TRAIN   = TRAIN_OUT / 'pass1_finetune_train_pool.csv'
OUT_SUMMARY = TRAIN_OUT / 'pass1_finetune_split_summary.txt'

# ---- Helpers -----------------------------------------------------------------
def norm(s):
    return str(s).strip().lower()

def detect_desc_col(df, label):
    for col in df.columns:
        if col.strip().lower() in ('description', 'text'):
            return col
    raise ValueError(
        label + ": cannot find description column.\n"
        "  Expected: 'DESCRIPTION' or 'text'\n"
        "  Present : " + str(list(df.columns))
    )

# ---- Strata ------------------------------------------------------------------
SHORT_IMP_VERBS = {'write','get','make','set','list','fetch','send','run'}
ACTION_RE       = re.compile(r'\b(run|execute|invoke|deploy|trigger|launch|kill|exec)\b')
NW_BRANDS       = ['bilibili','feishu','flomo','weibo','baidu','tencent','alibaba',
                   'douban','wechat','lark','dingtalk','gitee','qq','taobao']
GEN_RE          = re.compile(r'\b(generate|convert|transform|render|compile|encode|decode|translate)\b')

STRATA = [
    {'id': 1, 'label': 'Short imperative',   'target': 1000},
    {'id': 2, 'label': 'Action verb heavy',  'target': 1000},
    {'id': 3, 'label': 'Non-Western brand',  'target':  500},
    {'id': 4, 'label': 'Generate/transform', 'target':  500},
    {'id': 5, 'label': 'Long descriptive',   'target': 1000},
    {'id': 6, 'label': 'Random remainder',   'target': 1000},
]

def assign_stratum(desc):
    dl    = desc.strip().lower()
    words = dl.split()
    if len(desc.strip()) < 50 and words and words[0] in SHORT_IMP_VERBS:
        return 1
    if ACTION_RE.search(dl):
        return 2
    if any(b in dl for b in NW_BRANDS):
        return 3
    if GEN_RE.search(dl):
        return 4
    if len(words) >= 100:
        return 5
    return 6

# ==============================================================================
# LOAD
# ==============================================================================
print("Loading candidate pool ...")
candidate = pd.read_csv(CANDIDATE_CLEAN)
desc_col  = detect_desc_col(candidate, 'CANDIDATE')
print("  {:,} rows  |  description column = '{}'".format(len(candidate), desc_col))

print("Loading evaluation sets ...")
eval_n154 = pd.read_csv(EVAL_N154)
eval_mcpc = pd.read_csv(EVAL_MCPC)
col_n154  = detect_desc_col(eval_n154, 'EVAL_N154')
col_mcpc  = detect_desc_col(eval_mcpc, 'EVAL_MCPC')

forbidden = set(eval_n154[col_n154].map(norm)) | set(eval_mcpc[col_mcpc].map(norm))
print("  Forbidden set: {:,} entries".format(len(forbidden)))
assert len(forbidden) >= 100, "Forbidden set suspiciously small ({})".format(len(forbidden))

overlap_count = candidate[desc_col].map(norm).isin(forbidden).sum()
print("  Eval-overlap rows in candidate: {}  (expected 0)".format(overlap_count))

# ==============================================================================
# ASSIGN STRATA
# ==============================================================================
print("\nAssigning strata ...")
candidate = candidate.copy()
candidate['_stratum'] = candidate[desc_col].map(assign_stratum)
for s in STRATA:
    n = (candidate['_stratum'] == s['id']).sum()
    print("  Stratum {} ({}): {:,} rows".format(s['id'], s['label'], n))

# ==============================================================================
# REPO-LEVEL GREEDY SELECTION WITH ROW CAP
# ==============================================================================
print("\nSelecting repos per stratum (cap={} rows/repo) ...".format(MAX_HOLDOUT_ROWS_PER_REPO))

rng = random.Random(SEED)

holdout_idx_stratum = []   # list of (original_index, stratum_id)
claimed_repos = set()      # small repos fully assigned to holdout
capped_repos  = set()      # large repos partially assigned to holdout
stratum_stats = []

for s in STRATA:
    sid, target, label = s['id'], s['target'], s['label']
    print("  Stratum {}/6: {} (target={}) ...".format(sid, label, target))

    repos_in_stratum = (
        candidate[candidate['_stratum'] == sid]['SOURCE_REPO']
        .unique().tolist()
    )
    eligible = [r for r in repos_in_stratum
                if r not in claimed_repos and r not in capped_repos]

    rng.shuffle(eligible)

    selected_info = []
    accumulated   = 0

    for repo in eligible:
        if accumulated >= target:
            break

        repo_df   = candidate[candidate['SOURCE_REPO'] == repo]
        total_n   = len(repo_df)
        take_n    = min(total_n, MAX_HOLDOUT_ROWS_PER_REPO)
        is_capped = total_n > MAX_HOLDOUT_ROWS_PER_REPO

        sampled_idx = repo_df.sample(n=take_n, random_state=SEED).index.tolist()
        for idx in sampled_idx:
            holdout_idx_stratum.append((idx, sid))

        if is_capped:
            capped_repos.add(repo)
        else:
            claimed_repos.add(repo)

        accumulated += take_n
        selected_info.append({
            'repo': repo, 'holdout_rows': take_n,
            'total_rows': total_n, 'capped': is_capped
        })

    pct = accumulated / target * 100 if target > 0 else 0

    dom = max(selected_info, key=lambda x: x['holdout_rows'], default=None)
    dom_str = ''
    if dom and dom['holdout_rows'] > target * 0.3:
        cap_flag = ' [capped]' if dom['capped'] else ''
        dom_str  = "  WARNING dominant: {} ({}/{} rows{})".format(
            dom['repo'].split('/')[-1], dom['holdout_rows'], dom['total_rows'], cap_flag)

    n_capped = sum(1 for x in selected_info if x['capped'])
    print("    -> {:,} rows from {} repos ({:.0f}% of target)  [{} capped]{}".format(
        accumulated, len(selected_info), pct, n_capped, dom_str))

    stratum_stats.append({
        'id': sid, 'label': label, 'target': target,
        'actual': accumulated, 'repos': len(selected_info),
        'n_capped': n_capped, 'pct': pct, 'dominant': dom,
    })

# ==============================================================================
# BUILD DATAFRAMES
# ==============================================================================
holdout_idx_set = {i for i, _ in holdout_idx_stratum}
idx_to_stratum  = {i: st for i, st in holdout_idx_stratum}

holdout    = candidate.loc[sorted(holdout_idx_set)].copy()
holdout['STRATUM'] = holdout.index.map(idx_to_stratum)
holdout    = holdout.drop(columns=['_stratum'])

train_pool = candidate.loc[~candidate.index.isin(holdout_idx_set)].copy()
train_pool = train_pool.drop(columns=['_stratum'])

# ==============================================================================
# VERIFICATION
# ==============================================================================
print("\nRunning verification checks ...")

v1_overlap = set(holdout[desc_col].map(norm)) & forbidden
v1_pass    = len(v1_overlap) == 0
print("  V1 (no eval overlap)  : " + ("PASS" if v1_pass else "FAIL -- {} rows".format(len(v1_overlap))))

v2_overlap = set(holdout.index) & set(train_pool.index)
v2_pass    = len(v2_overlap) == 0
print("  V2 (no row overlap)   : " + ("PASS" if v2_pass else "FAIL -- {} rows".format(len(v2_overlap))))

repo_overlap = set(holdout['SOURCE_REPO']) & set(train_pool['SOURCE_REPO'])
print("  V2b (repo overlap)    : {} repos in both (these are capped large repos)".format(len(repo_overlap)))

v3_sum  = len(holdout) + len(train_pool)
v3_pass = v3_sum == len(candidate)
print("  V3 (row count sums)   : " + ("PASS" if v3_pass else "FAIL -- {} != {}".format(v3_sum, len(candidate))))

starved = [s for s in stratum_stats if s['pct'] < 80]
v4_pass = len(starved) == 0
v4_msg  = "PASS" if v4_pass else "partial -- starved: " + str([s['label'] for s in starved])
print("  V4 (stratum coverage) : " + v4_msg)

assert v1_pass, "ABORT: V1 failed"
assert v2_pass, "ABORT: V2 failed -- duplicate rows"
assert v3_pass, "ABORT: V3 failed -- row count mismatch"

# ==============================================================================
# WRITE FILES
# ==============================================================================
holdout.to_csv(OUT_HOLDOUT,  index=False)
train_pool.to_csv(OUT_TRAIN, index=False)
print("\nSaved holdout    -> {}   ({:,} rows)".format(OUT_HOLDOUT.name, len(holdout)))
print("Saved train pool -> {}  ({:,} rows)".format(OUT_TRAIN.name, len(train_pool)))

# ---- Summary -----------------------------------------------------------------
first20 = sorted(holdout['SOURCE_REPO'].unique())[:20]

lines = [
    "=== Fine-tuning holdout split (v2 -- row cap) ===",
    "Generated              : " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "Random seed            : " + str(SEED),
    "Max rows/repo holdout  : " + str(MAX_HOLDOUT_ROWS_PER_REPO),
    "",
    "Input  : {}  ({:,} rows, {} repos)".format(
        CANDIDATE_CLEAN.name, len(candidate), candidate['SOURCE_REPO'].nunique()),
    "Output :",
    "  pass1_finetune_holdout.csv     ({:,} rows, {} repos)".format(
        len(holdout), holdout['SOURCE_REPO'].nunique()),
    "  pass1_finetune_train_pool.csv  ({:,} rows, {} repos)".format(
        len(train_pool), train_pool['SOURCE_REPO'].nunique()),
    "  Repos appearing in both (capped large repos): {}".format(len(repo_overlap)),
    "",
    "Per-stratum breakdown:",
]
for s in stratum_stats:
    dom      = s['dominant']
    dom_note = ''
    if dom and dom['holdout_rows'] > s['target'] * 0.3:
        cap_flag = '[capped]' if dom['capped'] else ''
        dom_note = "  dominant: {}/{} {}".format(
            dom['holdout_rows'], dom['total_rows'],
            dom['repo'].split('/')[-1] + (' ' + cap_flag if cap_flag else ''))
    lines.append(
        "  #{} {:<22}  target={:>5}  actual={:>5}  repos={:>4}  capped={:>3}  ({:.0f}%){}".format(
            s['id'], s['label'], s['target'], s['actual'],
            s['repos'], s['n_capped'], s['pct'], dom_note)
    )

lines += [
    "",
    "Verification checks:",
    "  V1  (no eval text overlap) : " + ("PASS" if v1_pass else "FAIL"),
    "  V2  (no row overlap)       : " + ("PASS" if v2_pass else "FAIL"),
    "  V2b (repo overlap count)   : {} repos (capped, expected)".format(len(repo_overlap)),
    "  V3  (row count sums)       : " + ("PASS" if v3_pass else "FAIL"),
    "  V4  (stratum coverage)     : " + v4_msg,
    "",
    "First {} holdout repos:".format(len(first20)),
]
for r in first20:
    lines.append("  " + r)

report = "\n".join(lines)
OUT_SUMMARY.write_text(report)
print()
print(report)
print("\nSaved summary -> " + OUT_SUMMARY.name)
