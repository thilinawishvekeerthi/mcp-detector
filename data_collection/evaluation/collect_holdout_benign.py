"""
collect_mcp_tools.py  (v2 — raw CDN, no API rate limits)
─────────────────────────────────────────────────────────
Fetches real MCP server source files directly from raw.githubusercontent.com
(public CDN, no authentication, no rate limits).

Targets: MIT / Apache-2.0 licensed MCP server repositories.
Output:  realworld_validation/benign_tools_raw.json

Legal basis: All repos are public, open-source (MIT/Apache-2.0 confirmed).
             Academic research use, non-commercial, IEEE Access paper.
"""

import requests, json, re, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

OUT_FILE = Path(__file__).parent / "output" / "benign_tools_raw.json"

RAW = "https://raw.githubusercontent.com"

# ── Known source files with real MCP tool definitions ─────────────────────────
# Each entry: (owner, repo, branch, filepath, licence)
# All confirmed MIT or Apache-2.0.
SOURCE_FILES = [
    # ── executeautomation/mcp-playwright (MIT) ──────────────────────────────
    ("executeautomation","mcp-playwright","main","src/index.ts","mit"),
    ("executeautomation","mcp-playwright","main","src/tools.ts","mit"),

    # ── punkpeye/fastmcp (MIT) ───────────────────────────────────────────────
    ("punkpeye","fastmcp","main","examples/addition.ts","mit"),
    ("punkpeye","fastmcp","main","examples/error_handling.ts","mit"),
    ("punkpeye","fastmcp","main","README.md","mit"),

    # ── modelcontextprotocol/servers — official reference servers (MIT) ──────
    ("modelcontextprotocol","servers","main","src/filesystem/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/brave-search/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/fetch/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/memory/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/sequentialthinking/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/time/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/puppeteer/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/slack/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/google-maps/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/github/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/gitlab/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/postgres/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/sqlite/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/sentry/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/aws-kb-retrieval-server/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/google-drive/index.ts","mit"),
    ("modelcontextprotocol","servers","main","src/everart/index.ts","mit"),

    # ── modelcontextprotocol/python-sdk examples (MIT) ───────────────────────
    ("modelcontextprotocol","python-sdk","main","examples/servers/simple-tool/server.py","mit"),
    ("modelcontextprotocol","python-sdk","main","examples/servers/everything/server.py","mit"),

    # ── rusiaaman/wcgw (Apache-2.0) ──────────────────────────────────────────
    ("rusiaaman","wcgw","main","src/wcgw/tools.py","apache-2.0"),
    ("rusiaaman","wcgw","main","README.md","apache-2.0"),

    # ── simonb97/win-cli-mcp-server (MIT) ────────────────────────────────────
    ("simonb97","win-cli-mcp-server","main","src/index.ts","mit"),
    ("simonb97","win-cli-mcp-server","main","README.md","mit"),

    # ── anaisbetts/mcp-youtube (MIT) ─────────────────────────────────────────
    ("anaisbetts","mcp-youtube","main","src/index.ts","mit"),
    ("anaisbetts","mcp-youtube","main","README.md","mit"),

    # ── snaggle-ai/openapi-mcp-server (MIT) ──────────────────────────────────
    ("snaggle-ai","openapi-mcp-server","main","src/index.ts","mit"),
    ("snaggle-ai","openapi-mcp-server","main","src/tools.ts","mit"),

    # ── Additional community servers ─────────────────────────────────────────
    # mcp-server-sqlite (MIT)
    ("MarkusPfundstein","mcp-server-sqlite-npx","main","src/index.ts","mit"),
    # openai-realtime-mcp (MIT)
    ("Ejb503","multimodal-mcp-client","main","src/tools/index.ts","mit"),
    # mcp-server-filesystem-s3 (MIT)
    ("aws-samples","amazon-bedrock-mcp-server","main","src/index.ts","mit"),
    # mcp-server-github alternative
    ("github","github-mcp-server","main","pkg/github/tools.go","mit"),
    # Playwright-python MCP
    ("microsoft","playwright-mcp","main","src/tools/browser.ts","apache-2.0"),
    ("microsoft","playwright-mcp","main","src/tools/keyboard.ts","apache-2.0"),
    ("microsoft","playwright-mcp","main","src/tools/snapshot.ts","apache-2.0"),
    ("microsoft","playwright-mcp","main","src/tools/navigate.ts","apache-2.0"),
    ("microsoft","playwright-mcp","main","README.md","apache-2.0"),
    # Linear MCP (MIT)
    ("jerhadf","linear-mcp-server","main","src/index.ts","mit"),
    # Notion MCP (MIT)
    ("makenotion","notion-mcp-server","main","src/server.ts","mit"),
    # Jira MCP (MIT)
    ("george-holtz-in-prod","jira-mcp","main","src/index.ts","mit"),
    # Spotify MCP (MIT)
    ("varunneal","spotify-mcp","main","src/spotify_mcp/server.py","mit"),
    # Stripe MCP (MIT)
    ("stripe","agent-toolkit","main","python/stripe_agent_toolkit/tools.py","mit"),
    # Cloudflare MCP (MIT)
    ("cloudflare","mcp-server-cloudflare","main","src/index.ts","mit"),
    # Vercel MCP (MIT)
    ("vercel","mcp-adapter","main","README.md","mit"),
]


def fetch_raw(owner, repo, branch, filepath):
    url  = f"{RAW}/{owner}/{repo}/{branch}/{filepath}"
    time.sleep(0.15)  # polite delay
    try:
        r = requests.get(url, timeout=12,
                         headers={"User-Agent": "mcp-security-research-academic"})
        if r.status_code == 200:
            return r.text
        return None
    except Exception:
        return None


# ── Extraction functions ───────────────────────────────────────────────────────

def extract_ts(content, repo_id):
    tools = []

    # server.tool("name", "description", ...) or .tool("name", "description")
    p1 = re.compile(r'\.tool\s*\(\s*["\']([^"\']{2,60})["\'],\s*\n?\s*["\']([^"\']{15,600})["\']')
    for m in p1.finditer(content):
        tools.append({"text": m.group(2).strip(), "source": repo_id,
                      "method": "ts_tool_call", "tool_name": m.group(1)})

    # name: "...",\n  description: "..."
    p2 = re.compile(r'name\s*:\s*["\']([^"\']{2,60})["\'],?\s*\n\s*description\s*:\s*["\']([^"\']{15,600})["\']')
    for m in p2.finditer(content):
        tools.append({"text": m.group(2).strip(), "source": repo_id,
                      "method": "ts_obj_name_desc", "tool_name": m.group(1)})

    # description: "..." (standalone)
    p3 = re.compile(r'description\s*:\s*["\']([^"\']{20,600})["\']')
    for m in p3.finditer(content):
        desc = m.group(1).strip()
        if any(w in desc.lower() for w in ["tool","file","read","write","search","query",
                                            "get","fetch","list","create","delete","send",
                                            "update","execute","run","check","find"]):
            tools.append({"text": desc, "source": repo_id,
                          "method": "ts_desc_standalone", "tool_name": "unknown"})

    # description: `template literal`
    p4 = re.compile(r'description\s*:\s*`([^`]{20,600})`')
    for m in p4.finditer(content):
        tools.append({"text": m.group(1).strip().replace('\n',' '), "source": repo_id,
                      "method": "ts_template_literal", "tool_name": "unknown"})

    # inputSchema descriptions or annotation strings next to tool patterns
    p5 = re.compile(r'(?:title|label)\s*:\s*["\']([^"\']{5,60})["\'],\s*\n?\s*description\s*:\s*["\']([^"\']{15,400})["\']')
    for m in p5.finditer(content):
        tools.append({"text": m.group(2).strip(), "source": repo_id,
                      "method": "ts_schema_desc", "tool_name": m.group(1)})
    return tools


def extract_py(content, repo_id):
    tools = []

    # @mcp.tool() or @server.tool() decorator + docstring
    p1 = re.compile(
        r'@\w+\.tool\([^)]*\)\s*\n(?:async\s+)?def\s+(\w+)[^:]*:\s*\n\s*"""(.*?)"""',
        re.DOTALL)
    for m in p1.finditer(content):
        desc = " ".join(m.group(2).strip().split())
        if 15 < len(desc) < 800:
            tools.append({"text": desc, "source": repo_id,
                          "method": "py_decorator_docstring", "tool_name": m.group(1)})

    # Tool(name=..., description=...)
    p2 = re.compile(r'Tool\s*\(\s*name\s*=\s*["\']([^"\']+)["\'],\s*description\s*=\s*["\']([^"\']{20,600})["\']')
    for m in p2.finditer(content):
        tools.append({"text": m.group(2).strip(), "source": repo_id,
                      "method": "py_Tool_class", "tool_name": m.group(1)})

    # "description": "..." in dicts
    p3 = re.compile(r'"description"\s*:\s*"([^"]{20,600})"')
    for m in p3.finditer(content):
        tools.append({"text": m.group(1).strip(), "source": repo_id,
                      "method": "py_dict_desc", "tool_name": "unknown"})

    # Function docstrings after def (broad catch for tools modules)
    p4 = re.compile(r'def\s+(\w+)\s*\([^)]*\)[^:]*:\s*\n\s*"""([^"]{20,600})"""')
    for m in p4.finditer(content):
        desc = " ".join(m.group(2).strip().split())
        if len(desc) > 20:
            tools.append({"text": desc, "source": repo_id,
                          "method": "py_func_docstring", "tool_name": m.group(1)})
    return tools


def extract_go(content, repo_id):
    """Extract from Go MCP servers."""
    tools = []
    # Description: "..."
    p1 = re.compile(r'Description:\s*"([^"]{20,600})"')
    for m in p1.finditer(content):
        tools.append({"text": m.group(1).strip(), "source": repo_id,
                      "method": "go_desc_field", "tool_name": "unknown"})
    return tools


def extract_readme(content, repo_id):
    tools = []
    # Markdown table rows
    p1 = re.compile(r'\|\s*`?([a-zA-Z_][a-zA-Z0-9_/-]*)`?\s*\|\s*([^|\n]{20,300})\s*\|')
    for m in p1.finditer(content):
        desc = m.group(2).strip()
        if not any(skip in desc.lower() for skip in ['description','type','parameter','---']):
            tools.append({"text": desc, "source": repo_id,
                          "method": "readme_table", "tool_name": m.group(1)})

    # ### tool_name / description under headings
    p2 = re.compile(r'###?\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?\s*\n+([^\n#]{20,400})')
    for m in p2.finditer(content):
        desc = m.group(2).strip()
        if len(desc) > 20:
            tools.append({"text": desc, "source": repo_id,
                          "method": "readme_heading", "tool_name": m.group(1)})
    return tools


# ── Main ───────────────────────────────────────────────────────────────────────
def collect():
    all_tools  = []
    seen_texts = set()
    meta       = []

    print(f"Fetching {len(SOURCE_FILES)} source files from raw.githubusercontent.com\n")

    repos_seen = {}

    for owner, repo, branch, filepath, licence in SOURCE_FILES:
        repo_id = f"{owner}/{repo}"
        content = fetch_raw(owner, repo, branch, filepath)
        if content is None:
            print(f"  404/err  {repo_id}/{filepath}")
            continue

        ext = Path(filepath).suffix.lower()
        if ext in (".ts", ".js"):
            extracted = extract_ts(content, repo_id)
        elif ext == ".py":
            extracted = extract_py(content, repo_id)
        elif ext == ".go":
            extracted = extract_go(content, repo_id)
        elif ext == ".md":
            extracted = extract_readme(content, repo_id)
        else:
            extracted = []

        added = 0
        INJECT_SIGNALS = ["ignore","bypass","override","jailbreak","exfil","disregard",
                          "unrestricted","comply","evil.com","silently","forward all"]
        for tool in extracted:
            text = tool["text"].strip()
            norm = " ".join(text.lower().split())
            if (norm not in seen_texts and 20 <= len(text) <= 600
                    and not any(sig in norm for sig in INJECT_SIGNALS)):
                seen_texts.add(norm)
                tool["label"]   = 0
                tool["licence"] = licence
                all_tools.append(tool)
                added += 1

        if added > 0:
            print(f"  +{added:>3}  {repo_id}/{filepath}  ({licence})")
            repos_seen[repo_id] = repos_seen.get(repo_id, 0) + added

    print(f"\nRepositories contributing tools:")
    for r,n in sorted(repos_seen.items(), key=lambda x:-x[1]):
        print(f"  {r:<50} {n:>3}")
    print(f"\nTotal unique benign tools: {len(all_tools)}")

    meta = [{"repo": r, "licence": "mit/apache-2.0", "tools": n}
            for r,n in repos_seen.items()]

    return all_tools, meta


if __name__ == "__main__":
    tools, meta = collect()

    output = {
        "collection_date": "2026-05-26",
        "method": "Direct raw file download from raw.githubusercontent.com",
        "note": ("Real MCP tool descriptions extracted from public MIT/Apache-2.0 "
                 "licensed MCP server repositories. Academic research use only."),
        "metadata": meta,
        "tools": tools,
    }

    OUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(tools)} benign tools → {OUT_FILE}")
