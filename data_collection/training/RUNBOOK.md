# RUNBOOK — IEEE Access paper, benign MCP tool description corpus

Every command, in order. Copy/paste. No skipping steps.

---

## 0. Prerequisites (one-time)

```bash
# Python 3.10+ and git on PATH
python3 --version    # expect 3.10 or higher
git --version

# A GitHub personal access token (classic, scope: public_repo).
# Generate at: https://github.com/settings/tokens
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Without `GITHUB_TOKEN` the GitHub API rate limit is 60/hr instead of 5000/hr.
This will cause the license-resolution step to take 6–8 hours instead of 30
minutes. Set the token.

---

## 1. Set up the project (2 minutes)

```bash
mkdir -p ~/mcp-paper/data-collection
cd       ~/mcp-paper/data-collection

# Copy the three files I provided here:
#   collect_mcp_descriptions.py
#   test_extractors.py
#   requirements.txt

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Run the unit tests (10 seconds)

```bash
python test_extractors.py
```

**Expect**: `OK: 0 failures` after 16 PASS lines. If anything fails, stop — the
extractors are broken and a full run will produce garbage.

---

## 3. Pin the awesome-mcp-servers README for reproducibility (1 minute)

For the IEEE submission you want the README pinned to a specific commit, so a
reviewer can reproduce exactly. Don't rely on `main`.

```bash
# Find the latest commit SHA at submission time:
LATEST_SHA=$(curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
    https://api.github.com/repos/punkpeye/awesome-mcp-servers/commits/main \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['sha'])")
echo "Pinned commit: $LATEST_SHA"

# Download that exact version:
curl -L \
    "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/${LATEST_SHA}/README.md" \
    -o awesome_mcp_servers_README.md
wc -l awesome_mcp_servers_README.md   # expect ~2800 lines
```

**Record `$LATEST_SHA` in your paper's methodology section.** That's the
reproducibility anchor for the awesome-list source.

---

## 4. Dry run — enumerate + license-check only (3–8 minutes)

```bash
python collect_mcp_descriptions.py \
    --dry-run \
    --out ./out \
    --awesome-list-path ./awesome_mcp_servers_README.md
```

**Expect**:
- `Seeds: 3`
- `After topic search: ~50–80`
- `After code search: ~80–130`
- `After npm: ~120–180`
- `awesome-list: ~2300–2400 unique repo candidates`
- `After awesome-list: ~2400–2500 total candidates`
- `License-filtered: ~900–1500 / ~2400 allowed`

**Inspect the license distribution:**

```bash
python3 -c "
import json, collections
m = json.load(open('out/manifest.json'))
c = collections.Counter(e['decision'] for e in m.values())
for k, v in c.most_common(): print(f'  {v:5d}  {k}')
"
```

If the `kept` count is < 500, something is wrong (probably the GitHub token).
If it's 800–1500, proceed.

---

## 5. Full extraction (1–3 hours wall time)

```bash
python collect_mcp_descriptions.py \
    --out ./out \
    --workspace ./ws \
    --awesome-list-path ./awesome_mcp_servers_README.md \
    --delete-after-extract \
    -v 2>&1 | tee collection.log
```

**What's happening**: for each allowed repo, the script shallow-clones it,
runs the three extractors (Python AST / JSON walker / TS-JS regex) over
matching source files, applies the content filters, deduplicates against
the global fingerprint set, and emits records. With `--delete-after-extract`,
each clone is removed immediately so peak disk stays around 1 GB.

You'll see lines like:

```
modelcontextprotocol/servers                       kept  142 / 187 (lic=MIT)
```

**Expect**:
- Wall time: 1–3 hours depending on network and how many repos clone cleanly.
- Final yield: probably 2,000–5,000 unique descriptions from `description`
  fields alone. If you need more, see Phase 4 below.

---

## 6. Inspect the outputs

```bash
ls -la out/
#   descriptions.txt          ← the spec format (SOURCE_REPO/TOOL_NAME/...)
#   descriptions.jsonl        ← full provenance, one JSON per line
#   manifest.json             ← every repo decision logged

# Quick stats:
wc -l out/descriptions.jsonl
python3 -c "
import json
recs = [json.loads(l) for l in open('out/descriptions.jsonl')]
print(f'Total records:      {len(recs)}')
print(f'Unique repos:       {len({r[\"source_repo\"] for r in recs})}')
print(f'By language:')
import collections
for k,v in collections.Counter(r['language'] for r in recs).most_common():
    print(f'  {k:12s}  {v}')
print(f'By license:')
for k,v in collections.Counter(r['license'] for r in recs).most_common():
    print(f'  {k:15s}  {v}')
"

# Spot-check 5 random entries:
python3 -c "
import json, random
recs = [json.loads(l) for l in open('out/descriptions.jsonl')]
for r in random.sample(recs, 5):
    print(f'\n[{r[\"source_repo\"]}  {r[\"language\"]}  L{r[\"line_number\"]}]')
    print(f'  tool: {r[\"tool_name\"]}')
    print(f'  desc: {r[\"description\"][:200]}')
"
```

---

## 7. Quality audit (mandatory for the IEEE submission)

Sample 50 random entries and manually verify:

```bash
python3 -c "
import json, random
recs = [json.loads(l) for l in open('out/descriptions.jsonl')]
sample = random.sample(recs, 50)
for i, r in enumerate(sample):
    print(f'{i+1}. [{r[\"source_repo\"]}] {r[\"tool_name\"]}')
    print(f'   {r[\"description\"][:300]}')
    print()
" > audit_sample.txt
less audit_sample.txt
```

Then run them through your existing classifier and inspect any predicted-
malicious entries — those are publishable findings either way:

```python
# In your Colab/notebook:
import json
from your_classifier import predict  # adapt to your API

new = [json.loads(l) for l in open('out/descriptions.jsonl')]
suspicious = [r for r in new if predict(r['description'])['malicious'] > 0.5]
print(f'{len(suspicious)} / {len(new)} predicted malicious in the benign set')
# Manually review each — these are either real-world poisoned tools (rare and
# interesting) or classifier false positives (calibration data).
```

---

## 8. (Optional) Phase 4 yield extension

If the Phase 5 yield is below what you need, the highest-leverage extension
is **per-parameter descriptions**. MCP tools declare an `inputSchema` with a
`properties` block where every parameter often has its own `description`.
That's typically a 3–5× multiplier on raw extraction count.

This is a ~1-day edit to the three extractors. Open an issue against the
script (or ask me) when you're ready and I'll ship the patch.

---

## 9. Integration with your existing dataset

Once you're satisfied with the audit, reformat to your existing schema and
merge:

```python
# Adapt these field names to whatever your existing benign rows use.
import json

with open('out/descriptions.jsonl') as f, open('benign_new.jsonl', 'w') as g:
    for line in f:
        r = json.loads(line)
        g.write(json.dumps({
            'text': r['description'],
            'label': 0,                           # benign
            'source': 'awesome-mcp-servers',      # for ablation studies
            'meta': {
                'repo': r['source_repo'],
                'license': r['license'],
                'commit': r['commit_sha'],
                'tool': r['tool_name'],
            },
        }) + '\n')
```

**Important**: dedup against your existing 232,339 benign samples *before*
adding to train/val/test. Otherwise you risk test-set leakage if a
description happens to already exist in your corpus.

---

## 10. What to put in the paper

For Section 3 (Data) of your IEEE Access submission:

- The exact commands you ran (this runbook is a fine reference).
- The pinned commit SHA of `awesome-mcp-servers/README.md`.
- The SPDX allowlist used (MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause,
  ISC, 0BSD, CC0-1.0, CC-BY-4.0).
- The total candidates / kept / skipped breakdown from `manifest.json`.
- The content filter rules (link to the script in your supplementary materials).
- The fingerprint dedup method.
- Final corpus size + a per-repository contribution histogram.
- The classifier-on-new-data audit results from step 7 above.
