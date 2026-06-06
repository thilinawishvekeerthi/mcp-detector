"""
convert_to_csv.py
─────────────────────────────────────────────────────────────────────────────
Converts output/descriptions.jsonl (output of collect_training_benign_full.py)
into the mcp_benign_real_world.csv format used by the training notebook.

Usage
─────
    python convert_to_csv.py

What it does
────────────
  1. Reads all existing DESCRIPTION values from OUT_CSV to avoid duplicates.
  2. Reads every record from JSONL_IN.
  3. Appends new, non-duplicate records (LABEL=0) to OUT_CSV.
  4. Prints a summary.
"""

import csv, json, sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
JSONL_IN = Path(__file__).parent / "output" / "descriptions.jsonl"
OUT_CSV  = Path(__file__).parent / "output" / "mcp_benign_real_world.csv"

FIELDNAMES = ["SOURCE_REPO", "TOOL_NAME", "LICENSE", "LICENSE_URL",
              "FILE", "DESCRIPTION", "LABEL"]

MIN_LEN = 15   # characters — matches notebook filter
MAX_LEN = 2000

# ── Load existing descriptions to dedup ──────────────────────────────────────
def load_existing(csv_path: Path) -> tuple[set[str], int]:
    seen, count = set(), 0
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["DESCRIPTION"].strip().lower())
                count += 1
    return seen, count


def main() -> None:
    if not JSONL_IN.exists():
        sys.exit(f"ERROR: {JSONL_IN} not found.\n"
                 "Run collect_mcp_descriptions.py first.")

    seen, existing = load_existing(OUT_CSV)
    print(f"Existing CSV   : {existing} entries  ({len(seen)} unique descriptions)")

    write_header = not OUT_CSV.exists() or existing == 0
    added = skipped_dup = skipped_len = 0

    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        with open(JSONL_IN, encoding="utf-8") as jf:
            for line in jf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                desc = rec.get("description", "").strip()

                if not (MIN_LEN <= len(desc) <= MAX_LEN):
                    skipped_len += 1
                    continue

                key = desc.lower()
                if key in seen:
                    skipped_dup += 1
                    continue

                seen.add(key)
                writer.writerow({
                    "SOURCE_REPO": rec.get("source_repo", ""),
                    "TOOL_NAME":   rec.get("tool_name", ""),
                    "LICENSE":     rec.get("license", "").upper(),
                    "LICENSE_URL": rec.get("license_url", ""),
                    "FILE":        rec.get("file_path", ""),
                    "DESCRIPTION": desc,
                    "LABEL":       0,
                })
                f.flush()
                added += 1

    final = existing + added
    print(f"Added          : {added} new entries")
    print(f"Skipped (dup)  : {skipped_dup}")
    print(f"Skipped (len)  : {skipped_len}")
    print(f"CSV total now  : {final} entries")
    print(f"CSV path       : {OUT_CSV}")
    if final < 1000:
        print(f"\nStill {1000 - final} short of 1000 — re-run collection or add more query diversity.")


if __name__ == "__main__":
    main()
