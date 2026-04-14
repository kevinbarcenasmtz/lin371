"""Fix transcript files that are missing a quarter number in the filename.

Scans data/raw/fmp/ for .txt files whose names don't end with _Q[1-4].txt,
infers the quarter from:
  1. The first "Operator:" line (e.g. "First Quarter 2025")
  2. Any line in the file mentioning a quarter (first match)
  3. The Date: header as a last-resort fallback

Then renames the file and patches the Period: header.

Usage:
    python scripts/fix_quarter_filenames.py              # live run
    python scripts/fix_quarter_filenames.py --dry-run    # preview only
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

RAW_FMP_DIR = Path(__file__).parent.parent / "data" / "raw" / "fmp"

# Maps text to quarter number
_QUARTER_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
}

# "First Half" -> Q2 (reports H1 results), "Second Half" -> Q4
_HALF_MAP = {"first": 2, "second": 4}

# Date-header fallback: call month -> quarter being reported on.
_MONTH_TO_QUARTER = {
    1: 4, 2: 4, 3: 4,     # Jan-Mar calls -> Q4 results
    4: 1, 5: 1, 6: 1,     # Apr-Jun calls -> Q1 results
    7: 2, 8: 2, 9: 2,     # Jul-Sep calls -> Q2 results
    10: 3, 11: 3, 12: 3,  # Oct-Dec calls -> Q3 results
}

# Regex that matches a properly-named file: TICKER_YEAR_Q[1-4].txt
_GOOD_NAME = re.compile(r"^[A-Z]+_\d{4}_Q[1-4]\.txt$")

# Regex to parse the ticker and year from any transcript filename.
# Handles: AAPL_2005_Q.txt, AAPL_2005_Q?.txt, AAPL_2005_Q1.txt, etc.
_PARSE_NAME = re.compile(r"^([A-Z]+)_(\d{4})_Q.*\.txt$")


def _extract_quarter_from_text(text: str) -> int | None:
    """Search text for the first quarter mention. Returns 1-4 or None."""
    # Pattern 1: "First Quarter", "Third-Quarter", "4th Quarter"
    m = re.search(
        r"\b(first|second|third|fourth|1st|2nd|3rd|4th)[\s-]quarter\b",
        text, re.IGNORECASE,
    )
    if m:
        return _QUARTER_WORDS[m.group(1).lower()]

    # Pattern 2: "Q1", "Q2", "Q3", "Q4" followed by space/punctuation/digit
    m = re.search(r"\bQ([1-4])(?=[\s,.\-;\d])", text)
    if m:
        return int(m.group(1))

    # Pattern 3: "First Half" / "Second Half" (e.g. RKLB)
    m = re.search(r"\b(first|second)[\s-]half\b", text, re.IGNORECASE)
    if m:
        return _HALF_MAP[m.group(1).lower()]

    return None


def _extract_quarter_from_date_header(text: str) -> int | None:
    """Parse the Date: header line and infer quarter from the month."""
    m = re.search(r"^Date:\s*(.+)$", text, re.MULTILINE)
    if not m:
        return None
    date_str = m.group(1).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return _MONTH_TO_QUARTER[dt.month]
        except ValueError:
            continue
    return None


def infer_quarter(filepath: Path) -> int | None:
    """Infer the quarter number for a transcript file.

    Strategy:
      1. First Operator: line
      2. Full file scan (first match)
      3. Date: header fallback
    """
    content = filepath.read_text(encoding="utf-8")

    # Strategy 1: Operator line
    m = re.search(r"^Operator:(.+)$", content, re.MULTILINE)
    if m:
        q = _extract_quarter_from_text(m.group(1))
        if q is not None:
            return q

    # Strategy 2: full file scan (first quarter mention anywhere)
    q = _extract_quarter_from_text(content)
    if q is not None:
        return q

    # Strategy 3: Date header
    q = _extract_quarter_from_date_header(content)
    if q is not None:
        return q

    return None


def needs_fixing(filepath: Path) -> bool:
    """Return True if the filename does NOT have a valid Q[1-4] quarter."""
    return not _GOOD_NAME.match(filepath.name)


def fix_file(filepath: Path, dry_run: bool) -> bool:
    """Infer quarter, rename file, and patch the Period: header.

    Returns True if a change was made (or would be made in dry-run).
    """
    m = _PARSE_NAME.match(filepath.name)
    if not m:
        print(f"  SKIP (cannot parse filename): {filepath.name}")
        return False

    ticker, year = m.group(1), m.group(2)

    q = infer_quarter(filepath)
    if q is None:
        print(f"  SKIP (could not infer quarter): {filepath.name}")
        return False

    # Build new filename from parts: AAPL_2005_Q4.txt
    new_name = f"{ticker}_{year}_Q{q}.txt"
    new_path = filepath.parent / new_name

    if new_path.exists() and new_path != filepath:
        print(f"  SKIP (target exists): {filepath.name} -> {new_name}")
        return False

    # Patch the Period: header (e.g. "Period: Q? 2005" -> "Period: Q4 2005")
    content = filepath.read_text(encoding="utf-8")
    patched = re.sub(
        r"^(Period:\s*Q)\S*(\s+\d{4})",
        rf"\g<1>{q}\2",
        content,
        count=1,
        flags=re.MULTILINE,
    )

    action = "WOULD" if dry_run else "DONE "
    print(f"  {action}: {filepath.name} -> {new_name}  (Q{q})")

    if not dry_run:
        filepath.write_text(patched, encoding="utf-8")
        if new_name != filepath.name:
            filepath.rename(new_path)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix transcript filenames missing quarter numbers.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes.")
    args = parser.parse_args()

    # Find ALL .txt transcript files, then filter to those needing a fix
    all_txt = sorted(RAW_FMP_DIR.rglob("*.txt"))
    to_fix = [f for f in all_txt if needs_fixing(f)]

    if not to_fix:
        print("All transcript filenames already have valid quarters. Nothing to do.")
        return

    already_ok = len(all_txt) - len(to_fix)
    print(f"Found {len(to_fix)} file(s) needing quarter fix ({already_ok} already OK).\n")

    fixed, skipped = 0, 0
    for f in to_fix:
        if fix_file(f, dry_run=args.dry_run):
            fixed += 1
        else:
            skipped += 1

    print(f"\n{'DRY RUN -- ' if args.dry_run else ''}Done: {fixed} fixed, {skipped} skipped.")


if __name__ == "__main__":
    main()
