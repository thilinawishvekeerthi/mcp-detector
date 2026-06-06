"""
collect_mcp_github.py
─────────────────────────────────────────────────────────────────────────────
Collects verbatim MCP tool descriptions from public GitHub repositories and
appends new entries to mcp_benign_real_world.csv.

Strategy
────────
1. Search GitHub for repos with topic:mcp-server (permissive licence).
2. For each repo, look for Python/TypeScript server files and extract
   tool descriptions from @mcp.tool() / server.tool() decorators and
   docstrings / description strings.
3. Filter: English only, length 15–2000 chars, not already in CSV.
4. Append to mcp_benign_real_world.csv with SOURCE_REPO, TOOL_NAME,
   LICENSE, LICENSE_URL, FILE, DESCRIPTION, LABEL=0.

Requirements
────────────
  pip install PyGithub requests

Usage
─────
  export GITHUB_TOKEN=ghp_xxxxxxxxxxxx   # read:public_repo scope
  python collect_mcp_github.py

The script prints progress and a summary at the end. Run it until
~1000 total rows are in mcp_benign_real_world.csv.
"""

import os, re, csv, time, sys
from pathlib import Path

try:
    from github import Github, GithubException, RateLimitExceededException
except ImportError:
    sys.exit("Install PyGithub first:  pip install PyGithub")

# ── Config ────────────────────────────────────────────────────────────────────
TOKEN    = os.environ.get("GITHUB_TOKEN", "")
CSV_PATH = Path(__file__).parent / "output" / "mcp_benign_real_world.csv"
TARGET   = 1000      # stop when CSV reaches this many rows
MIN_LEN  = 15
MAX_LEN  = 2000
MAX_REPOS = 200      # search up to this many repos

PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
              "isc", "mpl-2.0", "unlicense", "0bsd"}

# ── Load existing entries to avoid duplicates ─────────────────────────────────
def load_existing(csv_path: Path) -> tuple[set, int]:
    seen_descs = set()
    count = 0
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen_descs.add(row["DESCRIPTION"].strip().lower())
                count += 1
    return seen_descs, count

# ── Description extraction helpers ───────────────────────────────────────────
# Pattern 1: FastMCP / Python @mcp.tool() docstring
_PY_DOCSTRING = re.compile(
    r'@(?:mcp|server)\.tool\(\)[^\n]*\n'
    r'\s*(?:async\s+)?def\s+\w+[^:]*:\s*\n'
    r'\s*"""(.*?)"""',
    re.DOTALL,
)
# Pattern 2: description= keyword argument in tool registration
_PY_DESC_KW = re.compile(
    r'description\s*=\s*["\']([^"\']{15,})["\']'
)
# Pattern 3: TypeScript/JS: description: "..." in tool definition objects
_TS_DESC = re.compile(
    r'description\s*:\s*["`\']([^`"\']{15,})["`\']'
)
# Pattern 4: Go: Description: "..."
_GO_DESC = re.compile(
    r'Description:\s*"([^"]{15,})"'
)

def extract_descriptions(content: str, filename: str) -> list[str]:
    descs = []
    if filename.endswith(".py"):
        for m in _PY_DOCSTRING.finditer(content):
            desc = m.group(1).strip().split("\n")[0].strip()
            if desc:
                descs.append(desc)
        for m in _PY_DESC_KW.finditer(content):
            descs.append(m.group(1).strip())
    elif filename.endswith((".ts", ".js")):
        for m in _TS_DESC.finditer(content):
            descs.append(m.group(1).strip())
    elif filename.endswith(".go"):
        for m in _GO_DESC.finditer(content):
            descs.append(m.group(1).strip())
    return descs

def is_english(text: str) -> bool:
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1) < 0.10

def is_valid(desc: str) -> bool:
    if not (MIN_LEN <= len(desc) <= MAX_LEN):
        return False
    if not is_english(desc):
        return False
    # skip obvious non-descriptions (just a type name, etc.)
    if re.match(r'^[A-Z_]+$', desc):
        return False
    return True

# ── Main collection loop ───────────────────────────────────────────────────────
def main():
    if not TOKEN:
        print("WARNING: No GITHUB_TOKEN set. Rate limits will be very tight (60 req/hr).")
        print("Set GITHUB_TOKEN env var with read:public_repo scope for best results.\n")

    g = Github(TOKEN or None)
    seen_descs, existing_count = load_existing(CSV_PATH)
    print(f"Existing CSV: {existing_count} entries  ({len(seen_descs)} unique descriptions)")
    print(f"Target: {TARGET} total entries\n")

    if existing_count >= TARGET:
        print(f"Already at target ({existing_count} >= {TARGET}). Nothing to do.")
        return

    # Open CSV for appending
    write_header = not CSV_PATH.exists() or existing_count == 0
    out_f = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=[
        "SOURCE_REPO","TOOL_NAME","LICENSE","LICENSE_URL","FILE","DESCRIPTION","LABEL"
    ])
    if write_header:
        writer.writeheader()

    added_total = 0
    repos_searched = 0

    # Search queries — multiple to maximise diversity
    queries = [
        "topic:mcp-server language:python",
        "topic:mcp-server language:typescript",
        "topic:mcp-server language:go",
        "mcp server tools in:readme language:python stars:>5",
        "fastmcp tool description in:file language:python",
        "modelcontextprotocol server language:typescript stars:>3",
    ]

    for query in queries:
        if existing_count + added_total >= TARGET:
            break
        print(f"\nQuery: {query}")
        try:
            results = g.search_repositories(query, sort="stars", order="desc")
            for repo in results:
                if repos_searched >= MAX_REPOS:
                    break
                if existing_count + added_total >= TARGET:
                    break

                repos_searched += 1

                # Licence check
                try:
                    lic = repo.license
                    lic_name = lic.spdx_id.lower() if lic and lic.spdx_id else "unknown"
                except Exception:
                    lic_name = "unknown"

                if lic_name not in PERMISSIVE:
                    continue

                lic_url = f"https://github.com/{repo.full_name}/blob/HEAD/LICENSE"

                # Find server files
                try:
                    contents = repo.get_contents("")
                except GithubException:
                    continue

                server_files = []
                stack = list(contents)
                depth = 0
                while stack and depth < 200:
                    item = stack.pop()
                    depth += 1
                    if item.type == "dir" and item.name not in (".git","node_modules","dist","build","__pycache__"):
                        try:
                            stack.extend(repo.get_contents(item.path))
                        except GithubException:
                            pass
                    elif item.type == "file":
                        if (item.name.endswith((".py",".ts",".js",".go"))
                                and any(kw in item.name.lower() for kw in
                                        ("server","tool","mcp","handler","plugin"))):
                            server_files.append(item)

                repo_added = 0
                for file_item in server_files[:10]:  # max 10 files per repo
                    try:
                        raw = file_item.decoded_content.decode("utf-8", errors="replace")
                    except Exception:
                        continue

                    descs = extract_descriptions(raw, file_item.name)
                    for desc in descs:
                        if not is_valid(desc):
                            continue
                        key = desc.strip().lower()
                        if key in seen_descs:
                            continue
                        seen_descs.add(key)
                        # Use filename as tool name (best we can do without parsing AST)
                        tool_name = file_item.name.replace(".py","").replace(".ts","").replace(".go","")
                        writer.writerow({
                            "SOURCE_REPO": f"https://github.com/{repo.full_name}",
                            "TOOL_NAME":   tool_name,
                            "LICENSE":     lic_name.upper(),
                            "LICENSE_URL": lic_url,
                            "FILE":        file_item.path,
                            "DESCRIPTION": desc.strip(),
                            "LABEL":       0,
                        })
                        out_f.flush()
                        added_total += 1
                        repo_added += 1

                if repo_added:
                    total_now = existing_count + added_total
                    print(f"  [{total_now:4d}] {repo.full_name} ({lic_name}) +{repo_added}")

                # Respect rate limits
                remaining = g.get_rate_limit().core.remaining
                if remaining < 10:
                    reset = g.get_rate_limit().core.reset
                    wait = max(0, (reset - __import__('datetime').datetime.utcnow()).seconds + 5)
                    print(f"  Rate limit low ({remaining} remaining). Sleeping {wait}s...")
                    time.sleep(wait)

        except RateLimitExceededException:
            print("Rate limit hit. Wait 60s...")
            time.sleep(60)
        except Exception as e:
            print(f"  Query error: {e}")
            continue

    out_f.close()

    final_count = existing_count + added_total
    print(f"\n{'='*55}")
    print(f"Done. Added {added_total} new entries.")
    print(f"CSV now has {final_count} entries.")
    print(f"Repos searched: {repos_searched}")
    if final_count < TARGET:
        print(f"Still {TARGET - final_count} short of target — re-run with a fresh token or different queries.")
    else:
        print(f"Target {TARGET} reached.")
    print(f"CSV: {CSV_PATH}")

if __name__ == "__main__":
    main()
