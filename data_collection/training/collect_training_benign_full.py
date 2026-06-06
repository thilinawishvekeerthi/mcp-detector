#!/usr/bin/env python3
"""
collect_mcp_descriptions.py
===========================

Collects benign MCP (Model Context Protocol) tool descriptions from public
repositories for use as the benign class in an ML training/evaluation dataset.

Methodology (defensible in a viva)
----------------------------------
  1. Enumerate candidate repositories via:
       (a) a curated seed list of known-good MCP servers,
       (b) the GitHub topic "mcp-server" sorted by stars,
       (c) the GitHub code-search API for `"description"` near MCP tool patterns,
       (d) the npm registry for packages matching "mcp-server*",
       (e) the punkpeye/awesome-mcp-servers README (MIT-licensed registry of
           several hundred MCP servers across 40+ categories).
  2. Resolve the SPDX license of each repo via GitHub's License API.
  3. Filter to the SPDX allowlist:
        MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, CC0-1.0, CC-BY-4.0.
     Reject anything with no LICENSE, GPL/AGPL/LGPL family, NC variants,
     or "All Rights Reserved".
  4. Shallow-clone allowed repos to a workspace directory.
  5. Extract `description` fields with language-aware parsers:
        - Python files: ast.parse() walks Call/Dict/Assign nodes.
        - JSON files:   json.load() then recursive descent on dict keys.
        - TS/JS files:  regex over double/single/template-literal forms.
     Every extraction is logged with file path + line number + language.
  6. Apply content filters:
        - length >= 8 words,
        - no prompt-injection trigger patterns (conservative regex list),
        - no long base64-ish runs,
        - exact and normalised dedup (whitespace + casefold fingerprint).
  7. Emit two artifacts:
        - descriptions.txt : the exact spec format the prompt asked for
                             (SOURCE_REPO / TOOL_NAME / LICENSE / LICENSE_URL /
                              DESCRIPTION, separated by `---`).
        - descriptions.jsonl : one JSON record per description with full
                               provenance (file_path, line_number, language,
                               fingerprint) for reproducibility and audit.

Usage
-----
    export GITHUB_TOKEN=<personal access token, classic, public_repo scope>
    python collect_mcp_descriptions.py \\
        --out ./out \\
        --workspace ./ws \\
        --max-repos-per-source 50

    # Dry run (enumerate + license-check only, no clones, no extraction):
    python collect_mcp_descriptions.py --dry-run --out ./out

Reproducibility
---------------
The script writes a manifest.json listing every repo seen, its resolved
license, the reason it was kept or skipped, and the commit SHA at the time
of extraction. Re-running with the same manifest skips already-processed
repos.

Dependencies
------------
    pip install requests
    git CLI on PATH
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALLOWED_SPDX = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "0BSD",
    "CC0-1.0",
    "CC-BY-4.0",
    "ISC",          # MIT-equivalent, widely used in npm; include or remove per policy
}

# Curated seed list. These are well-known MCP server repos worth scraping first
# regardless of search-API freshness. Verify each before publication.
SEED_REPOS: list[str] = [
    "modelcontextprotocol/servers",
    "modelcontextprotocol/python-sdk",
    "modelcontextprotocol/typescript-sdk",
]

# File extensions we'll parse, mapped to a language tag.
PARSEABLE_EXTENSIONS = {
    ".py":   "python",
    ".json": "json",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
}

# Paths inside a repo we always skip (vendored deps, build output, tests of
# adversarial inputs that would pollute the benign class).
SKIP_PATH_FRAGMENTS = (
    "/node_modules/",
    "/.git/",
    "/dist/",
    "/build/",
    "/.next/",
    "/__pycache__/",
    "/venv/",
    "/.venv/",
    "/test/fixtures/",
    "/tests/fixtures/",
    "/adversarial/",
    "/red_team/",
    "/red-team/",
    "/jailbreak",
    "/prompt_injection",
)

# Conservative prompt-injection trigger patterns. Hits here cause the
# description to be DROPPED from the benign set. False positives are cheap;
# leaking a poisoned description into the benign class is expensive.
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|messages?)\b", re.I),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(?:a|an|the)\b", re.I),
    re.compile(r"\bact\s+as\s+(?:a|an|the)\s+(?:different|new)\b", re.I),
    re.compile(r"\bnew\s+(?:system\s+)?(?:instructions?|prompts?)\s*[:\-]", re.I),
    re.compile(r"\bsystem\s*prompt\s*[:\-]", re.I),
    re.compile(r"<\s*system\s*>", re.I),
    re.compile(r"<\s*\|\s*im_start\s*\|\s*>", re.I),
    re.compile(r"<\s*IMPORTANT\s*>", re.I),                # Invariant Labs poisoning pattern
    re.compile(r"\bexfiltrate\b", re.I),
    re.compile(r"\bsend\s+.{0,40}\bto\s+https?://", re.I),
    re.compile(r"\b\.onion\b", re.I),
    # Long base64-ish run (40+ chars of base64 alphabet, no spaces).
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),
]

# Regex patterns for TS/JS description extraction. We capture three quote
# styles separately because escape handling differs.
TS_JS_PATTERNS = [
    # description: "..."
    re.compile(r'(?P<kind>description)\s*:\s*"(?P<val>(?:[^"\\]|\\.)*)"'),
    # description: '...'
    re.compile(r"(?P<kind>description)\s*:\s*'(?P<val>(?:[^'\\]|\\.)*)'"),
    # description: `...`  (template literal, no embedded ${...})
    re.compile(r"(?P<kind>description)\s*:\s*`(?P<val>[^`$]*)`"),
    # "description": "..."  (JSON-style inside .ts/.js)
    re.compile(r'"description"\s*:\s*"(?P<val>(?:[^"\\]|\\.)*)"'),
]

# Minimum length filter (in whitespace-separated tokens).
MIN_WORDS = 8


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("mcp_collect")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------

class GitHub:
    """Thin GitHub REST client with token auth and rate-limit awareness."""

    BASE = "https://api.github.com"

    def __init__(self, token: Optional[str]):
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "mcp-benign-collector/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            log.warning("No GITHUB_TOKEN set; unauthenticated rate limit is 60/hr.")
        self.session.headers.update(headers)

    def get(self, path: str, **params) -> Optional[dict]:
        url = path if path.startswith("http") else f"{self.BASE}{path}"
        for attempt in range(3):
            r = self.session.get(url, params=params, timeout=30)
            if r.status_code == 403 and "rate limit" in r.text.lower():
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                sleep_for = max(5, reset - int(time.time()) + 1)
                log.warning("Rate limited; sleeping %ds", sleep_for)
                time.sleep(sleep_for)
                continue
            if r.status_code == 404:
                return None
            if r.ok:
                return r.json()
            log.warning("GET %s -> %d: %s", url, r.status_code, r.text[:200])
            time.sleep(2 ** attempt)
        return None

    def search_repositories(self, query: str, per_page: int = 50, max_pages: int = 2) -> Iterator[dict]:
        for page in range(1, max_pages + 1):
            data = self.get("/search/repositories", q=query, sort="stars",
                            order="desc", per_page=per_page, page=page)
            if not data or not data.get("items"):
                return
            yield from data["items"]

    def repo_license(self, owner_repo: str) -> tuple[Optional[str], Optional[str]]:
        """Return (spdx_id, license_html_url) or (None, None) if absent."""
        data = self.get(f"/repos/{owner_repo}/license")
        if not data:
            return None, None
        spdx = (data.get("license") or {}).get("spdx_id")
        html_url = data.get("html_url")
        return spdx, html_url

    def repo_default_branch_sha(self, owner_repo: str) -> Optional[str]:
        data = self.get(f"/repos/{owner_repo}")
        if not data:
            return None
        branch = data.get("default_branch", "main")
        ref = self.get(f"/repos/{owner_repo}/commits/{branch}")
        return ref.get("sha") if ref else None


# ---------------------------------------------------------------------------
# Repo enumeration
# ---------------------------------------------------------------------------

@dataclass
class RepoCandidate:
    full_name: str            # "owner/repo"
    stars: int = 0
    source: str = "seed"      # "seed" | "topic" | "code-search" | "npm"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.full_name}"


def enumerate_seeds() -> Iterator[RepoCandidate]:
    for full_name in SEED_REPOS:
        yield RepoCandidate(full_name=full_name, source="seed")


def enumerate_topic(gh: GitHub, max_repos: int) -> Iterator[RepoCandidate]:
    seen = 0
    for item in gh.search_repositories("topic:mcp-server", max_pages=4):
        yield RepoCandidate(
            full_name=item["full_name"],
            stars=item.get("stargazers_count", 0),
            source="topic",
        )
        seen += 1
        if seen >= max_repos:
            return


def enumerate_code_search(gh: GitHub, max_repos: int) -> Iterator[RepoCandidate]:
    # Repositories matching the broad MCP-server keyword search.
    seen = 0
    for item in gh.search_repositories('"mcp server" tool description in:readme', max_pages=2):
        yield RepoCandidate(
            full_name=item["full_name"],
            stars=item.get("stargazers_count", 0),
            source="code-search",
        )
        seen += 1
        if seen >= max_repos:
            return


def enumerate_npm(max_repos: int) -> Iterator[RepoCandidate]:
    """Query the npm registry and follow repository.url back to GitHub."""
    url = "https://registry.npmjs.org/-/v1/search"
    seen: set[str] = set()
    for offset in range(0, max_repos, 50):
        try:
            r = requests.get(url, params={"text": "mcp-server", "size": 50,
                                          "from": offset}, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("npm search failed: %s", e)
            return
        objects = r.json().get("objects", [])
        if not objects:
            return
        for obj in objects:
            links = obj.get("package", {}).get("links", {}) or {}
            repo_url = links.get("repository") or ""
            m = re.match(r"https?://github\.com/([^/]+/[^/.#?]+)", repo_url)
            if not m:
                continue
            full_name = m.group(1)
            if full_name in seen:
                continue
            seen.add(full_name)
            yield RepoCandidate(full_name=full_name, source="npm")
            if len(seen) >= max_repos:
                return


AWESOME_LIST_URL = (
    "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md"
)

# Match a markdown list item whose first link is a GitHub repo. Captures
# (owner, repo). Examples that match:
#   - [foo/bar](https://github.com/foo/bar) ... description text
#   -  [foo/bar](https://github.com/foo/bar.git) - desc
#   * [foo/bar](https://github.com/foo/bar#readme) - desc
_AWESOME_LINE_RE = re.compile(
    r"^\s*[-*]\s+\[[^\]]+\]\("
    r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)"
    r"(?:\.git)?(?:[/#?][^)]*)?\)",
    re.MULTILINE,
)


def enumerate_awesome_list(local_path: Optional[Path] = None) -> Iterator[RepoCandidate]:
    """Parse the punkpeye/awesome-mcp-servers README and yield candidate repos.

    The README is MIT-licensed (we record this fact in our methodology, but
    that license applies to the LIST itself; each linked repo carries its own
    license and is filtered by the standard ALLOWED_SPDX check downstream).

    If `local_path` is provided, parses that file. Otherwise fetches the
    current `main` branch README from raw.githubusercontent.com. For
    reproducibility in the paper, pass --awesome-list-path pointing to a
    README pinned at a known commit SHA.
    """
    if local_path is not None:
        try:
            text = local_path.read_text(encoding="utf-8")
            log.info("awesome-list: read %d bytes from %s", len(text), local_path)
        except OSError as e:
            log.warning("awesome-list: cannot read %s: %s", local_path, e)
            return
    else:
        try:
            r = requests.get(AWESOME_LIST_URL, timeout=60)
            r.raise_for_status()
            text = r.text
            log.info("awesome-list: fetched %d bytes from %s", len(text), AWESOME_LIST_URL)
        except requests.RequestException as e:
            log.warning("awesome-list: fetch failed: %s", e)
            return

    # Only parse the "Server Implementations" body. Earlier sections contain
    # links to tutorials, client lists, and badge image links that aren't
    # MCP servers themselves.
    marker = "## Server Implementations"
    idx = text.find(marker)
    if idx == -1:
        log.warning("awesome-list: 'Server Implementations' section not found; "
                    "parsing entire document (may include non-server links)")
    else:
        text = text[idx:]

    # Self-references and meta-repos to exclude from candidates.
    EXCLUDE = {
        "punkpeye/awesome-mcp-servers",
        "punkpeye/awesome-mcp-clients",
        "punkpeye/awesome-mcp-devtools",
        "modelcontextprotocol/modelcontextprotocol",  # the protocol spec, not a server
    }

    seen: set[str] = set()
    for m in _AWESOME_LINE_RE.finditer(text):
        owner, repo = m.group(1), m.group(2)
        # Strip a trailing dot if the repo name accidentally captured one.
        repo = repo.rstrip(".")
        full_name = f"{owner}/{repo}"
        if full_name in EXCLUDE or full_name in seen:
            continue
        seen.add(full_name)
        yield RepoCandidate(full_name=full_name, source="awesome-list")
    log.info("awesome-list: %d unique repo candidates", len(seen))


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------

def shallow_clone(repo_full_name: str, dest: Path) -> bool:
    if dest.exists():
        log.debug("Already cloned: %s", dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo_full_name}.git"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
            check=True, timeout=180,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("Clone failed for %s: %s", repo_full_name, e)
        return False


def head_sha(repo_dir: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                                      text=True, timeout=10).strip()
        return out or None
    except subprocess.SubprocessError:
        return None


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

@dataclass
class Extraction:
    description: str
    tool_name: str
    file_path: str        # repo-relative
    line_number: int
    language: str


def iter_repo_files(repo_dir: Path) -> Iterator[Path]:
    for p in repo_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in PARSEABLE_EXTENSIONS:
            continue
        rel = "/" + str(p.relative_to(repo_dir)).replace(os.sep, "/") + "/"
        if any(frag in rel for frag in SKIP_PATH_FRAGMENTS):
            continue
        # Skip absurdly large files (likely lockfiles or generated).
        try:
            if p.stat().st_size > 2_000_000:
                continue
        except OSError:
            continue
        yield p


def extract_from_python(path: Path, rel: str) -> Iterator[Extraction]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=str(path))
    except (SyntaxError, ValueError):
        return

    # Strategy A: any Call() with keyword `description=...` where another
    # keyword `name=...` is present (e.g. @tool(name=..., description=...)).
    # Strategy B: any Dict with both 'name' and 'description' string keys.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            desc = name = None
            line = node.lineno
            for kw in node.keywords:
                if kw.arg == "description" and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, str):
                    desc = kw.value.value
                elif kw.arg == "name" and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, str):
                    name = kw.value.value
            if desc:
                yield Extraction(
                    description=desc,
                    tool_name=name or "<unnamed>",
                    file_path=rel,
                    line_number=line,
                    language="python",
                )
        elif isinstance(node, ast.Dict):
            desc = name = None
            line = node.lineno
            for k, v in zip(node.keys, node.values):
                if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                    continue
                if k.value == "description" and isinstance(v, ast.Constant) \
                        and isinstance(v.value, str):
                    desc = v.value
                elif k.value == "name" and isinstance(v, ast.Constant) \
                        and isinstance(v.value, str):
                    name = v.value
            if desc:
                yield Extraction(
                    description=desc,
                    tool_name=name or "<unnamed>",
                    file_path=rel,
                    line_number=line,
                    language="python",
                )


def extract_from_json(path: Path, rel: str) -> Iterator[Extraction]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return

    def walk(obj, parent_name: Optional[str] = None):
        if isinstance(obj, dict):
            local_name = obj.get("name") if isinstance(obj.get("name"), str) else parent_name
            desc = obj.get("description")
            if isinstance(desc, str):
                yield Extraction(
                    description=desc,
                    tool_name=local_name or "<unnamed>",
                    file_path=rel,
                    line_number=0,        # JSON parser doesn't preserve lines
                    language="json",
                )
            for v in obj.values():
                yield from walk(v, local_name)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v, parent_name)

    yield from walk(data)


def extract_from_tsjs(path: Path, rel: str, lang: str) -> Iterator[Extraction]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    # Build a list of (line_start_offset) so we can map char index -> line.
    line_starts = [0]
    for i, ch in enumerate(src):
        if ch == "\n":
            line_starts.append(i + 1)

    def line_of(offset: int) -> int:
        # Binary search would be faster; linear is fine for files <2MB.
        lo, hi = 0, len(line_starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= offset:
                lo = mid + 1
            else:
                hi = mid - 1
        return max(1, lo)

    # Pre-scan to map `name: "..."` (or `"name": "..."`) offsets so we can
    # associate the nearest preceding `name` with each `description`.
    name_pattern = re.compile(
        r'(?:"name"|\bname)\s*:\s*[\'"`]([^\'"`]+)[\'"`]'
    )
    names_by_offset: list[tuple[int, str]] = []
    for m in name_pattern.finditer(src):
        names_by_offset.append((m.start(), m.group(1)))

    def nearest_name(offset: int) -> str:
        candidate = "<unnamed>"
        for off, nm in names_by_offset:
            if off > offset:
                break
            candidate = nm
        return candidate

    seen_offsets: set[int] = set()
    for pattern in TS_JS_PATTERNS:
        for m in pattern.finditer(src):
            if m.start() in seen_offsets:
                continue
            seen_offsets.add(m.start())
            raw = m.group("val")
            # Unescape simple JS string escapes.
            try:
                desc = bytes(raw, "utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                desc = raw
            yield Extraction(
                description=desc,
                tool_name=nearest_name(m.start()),
                file_path=rel,
                line_number=line_of(m.start()),
                language=lang,
            )


def extract_from_repo(repo_dir: Path) -> Iterator[Extraction]:
    for path in iter_repo_files(repo_dir):
        rel = str(path.relative_to(repo_dir)).replace(os.sep, "/")
        ext = path.suffix.lower()
        lang = PARSEABLE_EXTENSIONS[ext]
        if lang == "python":
            yield from extract_from_python(path, rel)
        elif lang == "json":
            yield from extract_from_json(path, rel)
        else:
            yield from extract_from_tsjs(path, rel, lang)


# ---------------------------------------------------------------------------
# Filters & dedup
# ---------------------------------------------------------------------------

def passes_content_filters(desc: str) -> tuple[bool, str]:
    """Return (ok, reason)."""
    stripped = desc.strip()
    if not stripped:
        return False, "empty"
    words = stripped.split()
    if len(words) < MIN_WORDS:
        return False, f"too_short ({len(words)} words)"
    for pat in PROMPT_INJECTION_PATTERNS:
        if pat.search(stripped):
            return False, f"prompt_injection_pattern: {pat.pattern[:40]}"
    # Non-printable / mostly-non-ASCII rejection (covers binary slop).
    printable_ratio = sum(ch.isprintable() or ch in "\n\t" for ch in stripped) / max(1, len(stripped))
    if printable_ratio < 0.95:
        return False, "low_printable_ratio"
    return True, "ok"


def fingerprint(desc: str) -> str:
    """Normalised key for near-dedup: collapse whitespace, casefold."""
    return re.sub(r"\s+", " ", desc.strip()).casefold()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class Record:
    source_repo: str
    tool_name: str
    license: str
    license_url: str
    description: str
    # Provenance (JSONL only, not in the spec txt output)
    file_path: str = ""
    line_number: int = 0
    language: str = ""
    fingerprint: str = ""
    commit_sha: str = ""


def write_outputs(records: list[Record], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spec format
    spec_path = out_dir / "descriptions.txt"
    with spec_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            f.write(f"SOURCE_REPO: {rec.source_repo}\n")
            f.write(f"TOOL_NAME: {rec.tool_name}\n")
            f.write(f"LICENSE: {rec.license}\n")
            f.write(f"LICENSE_URL: {rec.license_url}\n")
            f.write(f"DESCRIPTION: {rec.description}\n")
            if i < len(records) - 1:
                f.write("---\n")
        f.write("\n")
        unique_repos = len({r.source_repo for r in records})
        f.write(f"TOTAL COLLECTED: {len(records)} entries from {unique_repos} unique repositories.\n")

    # 2. JSONL with full provenance
    jsonl_path = out_dir / "descriptions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    log.info("Wrote %d records -> %s", len(records), spec_path)
    log.info("Wrote %d records -> %s", len(records), jsonl_path)


# ---------------------------------------------------------------------------
# Manifest (resumability + audit trail)
# ---------------------------------------------------------------------------

@dataclass
class RepoManifestEntry:
    full_name: str
    source: str
    spdx: Optional[str] = None
    license_url: Optional[str] = None
    decision: str = "pending"   # "kept" | "skipped:<reason>" | "clone_failed"
    extractions: int = 0
    kept: int = 0
    commit_sha: Optional[str] = None


def load_manifest(path: Path) -> dict[str, RepoManifestEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: RepoManifestEntry(**v) for k, v in raw.items()}


def save_manifest(path: Path, manifest: dict[str, RepoManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: asdict(v) for k, v in manifest.items()}
    path.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    ws = Path(args.workspace)
    manifest_path = out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    gh = GitHub(token=os.environ.get("GITHUB_TOKEN"))

    # 1. Enumerate candidates from all sources.
    candidates: dict[str, RepoCandidate] = {}
    for cand in enumerate_seeds():
        candidates.setdefault(cand.full_name, cand)
    log.info("Seeds: %d", len(candidates))

    for cand in enumerate_topic(gh, args.max_repos_per_source):
        candidates.setdefault(cand.full_name, cand)
    log.info("After topic search: %d", len(candidates))

    for cand in enumerate_code_search(gh, args.max_repos_per_source):
        candidates.setdefault(cand.full_name, cand)
    log.info("After code search: %d", len(candidates))

    for cand in enumerate_npm(args.max_repos_per_source):
        candidates.setdefault(cand.full_name, cand)
    log.info("After npm: %d total candidates", len(candidates))

    awesome_path = Path(args.awesome_list_path) if args.awesome_list_path else None
    if not args.skip_awesome_list:
        for cand in enumerate_awesome_list(local_path=awesome_path):
            candidates.setdefault(cand.full_name, cand)
        log.info("After awesome-list: %d total candidates", len(candidates))

    # 2. License-filter every candidate (cache via manifest).
    allowed: list[RepoCandidate] = []
    for cand in candidates.values():
        entry = manifest.get(cand.full_name)
        if entry and entry.spdx is not None:
            spdx, lic_url = entry.spdx, entry.license_url
        else:
            spdx, lic_url = gh.repo_license(cand.full_name)
            entry = RepoManifestEntry(
                full_name=cand.full_name,
                source=cand.source,
                spdx=spdx,
                license_url=lic_url,
            )
            manifest[cand.full_name] = entry

        if spdx is None:
            entry.decision = "skipped:no_license"
            continue
        if spdx not in ALLOWED_SPDX:
            entry.decision = f"skipped:license={spdx}"
            continue
        entry.decision = "kept"
        allowed.append(cand)

    save_manifest(manifest_path, manifest)
    log.info("License-filtered: %d / %d allowed", len(allowed), len(candidates))

    if args.dry_run:
        log.info("Dry run complete. See manifest at %s", manifest_path)
        return 0

    # 3. Clone + extract.
    fingerprints: set[str] = set()
    records: list[Record] = []
    ws.mkdir(parents=True, exist_ok=True)

    for cand in allowed:
        entry = manifest[cand.full_name]
        repo_dir = ws / cand.full_name.replace("/", "__")

        if not shallow_clone(cand.full_name, repo_dir):
            entry.decision = "clone_failed"
            continue

        entry.commit_sha = head_sha(repo_dir)
        kept_here = total_here = 0
        for ext in extract_from_repo(repo_dir):
            total_here += 1
            ok, reason = passes_content_filters(ext.description)
            if not ok:
                log.debug("  filter rejected (%s): %s ...", reason, ext.description[:60])
                continue
            fp = fingerprint(ext.description)
            if fp in fingerprints:
                continue
            fingerprints.add(fp)
            kept_here += 1
            records.append(Record(
                source_repo=f"https://github.com/{cand.full_name}",
                tool_name=ext.tool_name,
                license=entry.spdx or "",
                license_url=entry.license_url or "",
                description=ext.description,
                file_path=ext.file_path,
                line_number=ext.line_number,
                language=ext.language,
                fingerprint=fp,
                commit_sha=entry.commit_sha or "",
            ))

        entry.extractions = total_here
        entry.kept = kept_here
        log.info("%-50s  kept %3d / %3d (lic=%s)",
                 cand.full_name, kept_here, total_here, entry.spdx)

        if args.delete_after_extract:
            shutil.rmtree(repo_dir, ignore_errors=True)

    save_manifest(manifest_path, manifest)
    write_outputs(records, out_dir)

    unique_repos = len({r.source_repo for r in records})
    log.info("DONE. %d descriptions from %d repos.", len(records), unique_repos)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="./out", help="Output directory")
    p.add_argument("--workspace", default="./ws", help="Workspace for clones")
    p.add_argument("--max-repos-per-source", type=int, default=50,
                   help="Max repos to enumerate per non-seed source")
    p.add_argument("--awesome-list-path", default=None,
                   help="Local path to a pinned awesome-mcp-servers README.md "
                        "(for reproducibility). If omitted, fetches main branch.")
    p.add_argument("--skip-awesome-list", action="store_true",
                   help="Do not include the punkpeye/awesome-mcp-servers source")
    p.add_argument("--dry-run", action="store_true",
                   help="Enumerate and license-check only; do not clone or extract")
    p.add_argument("--delete-after-extract", action="store_true",
                   help="Remove each clone after extraction (saves disk)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    setup_logging(args.verbose)
    try:
        return collect(args)
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
