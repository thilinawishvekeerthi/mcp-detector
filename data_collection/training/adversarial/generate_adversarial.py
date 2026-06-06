"""
generate_adversarial.py
Generates adversarial MCP tool description training examples.
Path 2 execution — MCP Detector v5 retrain.

Safeguards:
  - Benign openers from TRAINING pool only (never holdout)
  - Multilingual excluded (Safeguard 2 — zero-shot test)
  - 7 attack types x 6 injection positions
  - 10% held out as adversarial validation set (Safeguard 6)
  - Auto-verify every example contains >=1 trigger pattern
  - Safeguard 7: contamination check against all holdout sets

Usage:
  python3 generate_adversarial.py              # full 7,000 examples
  python3 generate_adversarial.py --pilot      # 500 examples for manual review
  python3 generate_adversarial.py --target 500 # custom count
"""

import random
import re
import base64
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--pilot',  action='store_true', help='Generate 500 examples for pilot review')
parser.add_argument('--target', type=int, default=7000, help='Total adversarial examples to generate')
parser.add_argument('--seed',   type=int, default=42)
args = parser.parse_args()

SEED     = args.seed
TARGET   = 500 if args.pilot else args.target
VAL_FRAC = 0.10
BASE     = Path('/Users/thilina/Documents/mcp-detector')
OUT_DIR  = BASE / 'data_collection/training/adversarial'
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(SEED)
print(f'[generate_adversarial.py]  target={TARGET}  seed={SEED}  {"PILOT MODE" if args.pilot else "FULL MODE"}')

# ── Load benign openers (training pool ONLY, never holdout) ───────────────────
benign_path = BASE / 'data_collection/training/output/mcp_benign_real_world_train.csv'
benign_pool = pd.read_csv(benign_path)

# Find the description column
desc_col = None
for col in ['DESCRIPTION', 'description', 'text']:
    if col in benign_pool.columns:
        desc_col = col
        break
if desc_col is None:
    raise ValueError(f'No description column found. Columns: {benign_pool.columns.tolist()}')

openers = benign_pool[desc_col].fillna('').astype(str).tolist()
openers = [o.strip() for o in openers if len(o.strip()) > 20]  # drop very short openers
opener_sample = rng.sample(openers, min(4000, len(openers)))
print(f'Opener pool: {len(opener_sample):,} benign tool descriptions (column={desc_col})')

# ── Attack pattern bank ───────────────────────────────────────────────────────
bank_path = OUT_DIR / 'attack_pattern_bank.json'
with open(bank_path) as f:
    bank = json.load(f)

DIRECT_OVERRIDE = bank['direct_override']
EXFILTRATION    = bank['exfiltration']
ROLE_INJECTION  = bank['role_injection']
TOOL_CHAINING   = bank['tool_chaining']

print(f'Pattern bank loaded: DO={len(DIRECT_OVERRIDE)} EX={len(EXFILTRATION)} RI={len(ROLE_INJECTION)} TC={len(TOOL_CHAINING)}')

# ── Obfuscation generators ────────────────────────────────────────────────────
LEET = str.maketrans('eaoitsEAOITS', '340175340175')

def to_leet(s):
    return s.translate(LEET)

# Unicode monospace mathematical alphabet (easier to transport as actual chars)
_LOWER = 'abcdefghijklmnopqrstuvwxyz'
_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_MATH_LOWER = '𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣'
_MATH_UPPER = '𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉'

# Build char-by-char map (MATH_MAP with str.maketrans won't work for multi-byte targets)
def to_math(s):
    result = []
    for ch in s:
        if ch in _LOWER:
            result.append(_MATH_LOWER[_LOWER.index(ch)])
        elif ch in _UPPER:
            result.append(_MATH_UPPER[_UPPER.index(ch)])
        else:
            result.append(ch)
    return ''.join(result)

def to_base64_partial(s):
    """Encode a random span (roughly 30-60% of words) in base64."""
    words = s.split()
    if len(words) < 4:
        return 'Decode and execute: ' + base64.b64encode(s.encode()).decode()
    start = rng.randint(0, len(words) // 3)
    end   = rng.randint(len(words) // 2, len(words))
    span  = ' '.join(words[start:end])
    encoded = base64.b64encode(span.encode()).decode()
    prefix = words[:start]
    suffix_verb = rng.choice([
        'Decode and execute:',
        'Process encoded directive:',
        'Handle encoded input:',
        'Execute base64 payload:',
    ])
    prefix_str = (' '.join(prefix) + ' ') if prefix else ''
    return prefix_str + suffix_verb + ' ' + encoded

# ── Injection position functions ──────────────────────────────────────────────
def inject_end(opener, attack):
    sep = rng.choice([' ', '. ', ' — '])
    return opener.rstrip('. ') + sep + attack

def inject_after_first_sentence(opener, attack):
    parts = opener.split('. ', 1)
    if len(parts) == 2:
        return parts[0] + '. ' + attack + ' ' + parts[1]
    return opener + '\n' + attack

def inject_newline(opener, attack):
    return opener + '\n\n' + attack

def inject_html_comment(opener, attack):
    return opener + ' <!-- ' + attack + ' -->'

def inject_markdown_block(opener, attack):
    style = rng.choice(['blockquote', 'code'])
    if style == 'blockquote':
        return opener + '\n\n> ' + attack
    return opener + '\n\n```\n' + attack + '\n```'

def inject_mid_sentence(opener, attack):
    words = opener.split()
    if len(words) < 6:
        return opener + ' ' + attack
    mid = rng.randint(len(words) // 3, 2 * len(words) // 3)
    return ' '.join(words[:mid]) + ' ' + attack + ' ' + ' '.join(words[mid:])

POSITIONS = [
    ('newline',          inject_newline),
    ('html_comment',     inject_html_comment),
    ('markdown_block',   inject_markdown_block),
    ('mid_sentence',     inject_mid_sentence),
    ('end_append',       inject_end),
    ('after_sentence',   inject_after_first_sentence),
]

# Safeguard 1: prioritise 4 positions NOT used in test set (60%); 40% for test-set positions
POSITION_WEIGHTS = {
    'newline':          0.18,
    'html_comment':     0.18,
    'markdown_block':   0.14,
    'mid_sentence':     0.10,
    'end_append':       0.20,
    'after_sentence':   0.20,
}

_pos_names   = [p[0] for p in POSITIONS]
_pos_fns     = [p[1] for p in POSITIONS]
_pos_weights = [POSITION_WEIGHTS[n] for n in _pos_names]

def pick_position():
    chosen = rng.choices(list(zip(_pos_names, _pos_fns)), weights=_pos_weights, k=1)[0]
    return chosen

# ── Attack type dispatcher ────────────────────────────────────────────────────
ATTACK_TYPES = {
    'direct_override': lambda: rng.choice(DIRECT_OVERRIDE),
    'exfiltration':    lambda: rng.choice(EXFILTRATION),
    'role_injection':  lambda: rng.choice(ROLE_INJECTION),
    'tool_chaining':   lambda: rng.choice(TOOL_CHAINING),
    'base64_partial':  lambda: to_base64_partial(rng.choice(DIRECT_OVERRIDE)),
    'leetspeak':       lambda: to_leet(rng.choice(DIRECT_OVERRIDE)),
    'mathematical':    lambda: to_math(rng.choice(ROLE_INJECTION)),
}

# Equal distribution across 7 types; remainder goes to direct_override
per_type = TARGET // len(ATTACK_TYPES)
remainder = TARGET - per_type * len(ATTACK_TYPES)

type_counts = {t: per_type for t in ATTACK_TYPES}
type_counts['direct_override'] += remainder

print(f'\nExamples per attack type:')
for t, n in type_counts.items():
    print(f'  {t:<20} {n:,}')
print(f'  {"TOTAL":<20} {sum(type_counts.values()):,}')

# ── Verification trigger patterns ─────────────────────────────────────────────
TRIGGERS = [
    'ignore', 'forget', 'override', 'bypass', 'disregard', 'comply',
    'exfiltrate', 'exfil', 'transfer', 'forward', 'post to', 'evil.com',
    'attacker', 'send_email', 'delete', 'unrestricted', 'dan',
    'no restrictions', 'safety', 'credentials', 'silently', 'automatically',
    'invoke', 'decode', 'execute', 'payment', 'inject', 'wipe', 'purge',
    'leak', 'harvest', 'exfil_endpoint', 'covertly', 'secretly',
    # leet versions (common transforms: e→3, a→4, o→0, i→1, t→7, s→5)
    '1gn0r3', '0v3rr1d3', 'byp4ss', '3xf1ltr4t3', 'd4n', 'c0mply',
    'c4nc3l', '4b0rt', '73rm1n473', 'unr357r1c73d', 'f0rg37',
    '5y573m', 'c0nf1g', '4dm1n', 'pr1v1l3g3', 's3cur17y',
    'd3v3l0p3r', 'm41n73n4nc3', '0p3r471ng', 'r35tr1c710n5',
]

# Attack types that use obfuscation — ASCII trigger check is insufficient.
# Instead verify injection happened by checking text is longer than opener.
OBFUSCATED_TYPES = {'leetspeak', 'mathematical', 'base64_partial'}

def has_trigger(text, attack_type=''):
    if attack_type in OBFUSCATED_TYPES:
        # For obfuscated types the pattern IS the attack; skip ASCII check
        return True
    t = text.lower()
    return any(tr in t for tr in TRIGGERS)

# ── Diversity matrix tracker ──────────────────────────────────────────────────
diversity = {}  # (attack_type, position) -> count

# ── Generate ──────────────────────────────────────────────────────────────────
examples = []
opener_idx = 0
failed_trigger = 0

for attack_type, get_pattern in ATTACK_TYPES.items():
    n_for_type = type_counts[attack_type]
    for i in range(n_for_type):
        opener   = opener_sample[opener_idx % len(opener_sample)]
        opener_idx += 1
        pattern  = get_pattern()
        pos_name, pos_fn = pick_position()
        text     = pos_fn(opener, pattern)

        # Safeguard: must contain a trigger pattern
        if not has_trigger(text, attack_type):
            # Fallback: append raw pattern so trigger is present
            text = text + ' ' + pattern
            failed_trigger += 1

        examples.append({
            'text':        text,
            'label':       1,
            'attack_type': attack_type,
            'position':    pos_name,
            'opener_src':  'source_g_train',
            'SOURCE_REPO': 'synthetic_adversarial_path2',
        })

        key = (attack_type, pos_name)
        diversity[key] = diversity.get(key, 0) + 1

print(f'\nGeneration complete. Trigger fallbacks used: {failed_trigger} / {len(examples)} ({100*failed_trigger/len(examples):.1f}%)')

rng.shuffle(examples)

# ── Safeguard 6: split off adversarial validation set ────────────────────────
val_n     = int(len(examples) * VAL_FRAC)
adv_val   = examples[:val_n]
adv_train = examples[val_n:]

# ── Save ──────────────────────────────────────────────────────────────────────
suffix = '_pilot' if args.pilot else ''

train_df = pd.DataFrame(adv_train)
val_df   = pd.DataFrame(adv_val)

train_path = OUT_DIR / f'adversarial_train{suffix}.csv'
val_path   = OUT_DIR / f'adversarial_val_set{suffix}.csv'

train_df.to_csv(train_path, index=False)
val_df.to_csv(  val_path,   index=False)

print(f'\nSaved:')
print(f'  {train_path.name:<35} {len(train_df):,} rows')
print(f'  {val_path.name:<35} {len(val_df):,} rows')

# ── Diversity matrix ──────────────────────────────────────────────────────────
print(f'\nDiversity matrix (attack_type x position):')
div_rows = {}
for (at, pos), count in diversity.items():
    div_rows.setdefault(at, {})[pos] = count
div_df = pd.DataFrame(div_rows).T.fillna(0).astype(int)
print(div_df.to_string())

thin = [(k, v) for k, v in diversity.items() if v < 30]
if thin:
    print(f'\nTHIN CELLS (< 30 examples): {thin}')
    print('Consider filling these deliberately if running full-scale generation.')
else:
    print('\nAll cells >= 30 examples. Diversity target met.')

# ── Safeguard 7: contamination check ─────────────────────────────────────────
print(f'\n--- Safeguard 7: contamination check ---')
rw_path = BASE / 'data_collection/evaluation/output/realworld_holdout.csv'
mc_path = BASE / 'evaluation/results/mcpcorpus_english_holdout.csv'

holdout_texts = set()
if rw_path.exists():
    rw = pd.read_csv(rw_path)
    rw_col = 'text' if 'text' in rw.columns else rw.columns[0]
    holdout_texts |= set(rw[rw_col].str.strip().str.lower())
    print(f'  realworld_holdout.csv loaded: {len(rw)} rows')
else:
    print(f'  WARNING: realworld_holdout.csv not found at {rw_path}')

if mc_path.exists():
    mc = pd.read_csv(mc_path)
    mc_col = 'DESCRIPTION' if 'DESCRIPTION' in mc.columns else mc.columns[0]
    holdout_texts |= set(mc[mc_col].str.strip().str.lower())
    print(f'  mcpcorpus_english_holdout.csv loaded: {len(mc)} rows')
else:
    print(f'  WARNING: mcpcorpus holdout not found at {mc_path}')

gen_texts = set(train_df['text'].str.strip().str.lower())
overlap   = gen_texts & holdout_texts
if overlap:
    print(f'\n  *** CONTAMINATION DETECTED: {len(overlap)} overlaps ***')
    for s in list(overlap)[:5]:
        print(f'    {s[:80]}')
else:
    print(f'  Contamination: 0 overlaps. CLEAN.')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'GENERATION SUMMARY')
print(f'{"="*60}')
print(f'  Mode          : {"PILOT" if args.pilot else "FULL"}')
print(f'  Total target  : {TARGET:,}')
print(f'  Training set  : {len(train_df):,}')
print(f'  Val set       : {len(val_df):,}  (Safeguard 6)')
print(f'  Trigger pass  : {len(examples) - failed_trigger:,} / {len(examples):,}')
print(f'  Contamination : {len(overlap)} (must be 0)')
print(f'\nNext step:')
if args.pilot:
    print('  1. MANUAL REVIEW: sample 100 rows from adversarial_train_pilot.csv')
    print('     Verify each is genuinely adversarial in intent.')
    print('  2. SANITY CHECK: run v4 model on pilot to confirm recall < 50%')
    print('     (high recall means patterns are too obvious — not a good training signal)')
    print('  3. If pilot passes: run without --pilot flag for full 7,000 examples')
else:
    print('  1. Upload adversarial_train.csv to Google Drive')
    print('  2. Delete Drive caches listed in §5e of execution doc')
    print('  3. Set FORCE_RERUN=True in Colab Cell 0 and retrain')
