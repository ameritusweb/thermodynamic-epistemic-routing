"""
Step 1: Dataset cleaning pass.

Scans all three splits for answers that contain model refusal language —
cases where the question generator failed to produce a real answer and the
refusal leaked into the dataset as a mislabeled example.

Actions:
  - Backs up originals to data/splits/backup/
  - Saves cleaned splits in-place
  - Prints a full report of what was removed

Usage:
    python clean_dataset.py            # dry run (report only, no changes)
    python clean_dataset.py --apply    # apply changes and overwrite splits
"""

import argparse
import json
import re
import shutil
from pathlib import Path


# ── Refusal patterns ──────────────────────────────────────────────────────────
#
# These match answers where the model declined to answer rather than producing
# a real response. Compiled case-insensitive.

REFUSAL_PATTERNS = [
    # Direct context negations
    r"context does not (explicitly )?(state|mention|say|provide|specify|discuss|contain|include)",
    r"context doesn't (explicitly )?(state|mention|say|provide|specify)",
    r"the (text|passage|article|excerpt) does not (explicitly )?(state|mention|say|provide|indicate)",

    # "cannot be determined" variants
    r"cannot (be )?determined from (the )?(context|passage|text|excerpt)",
    r"can('t| not) be (found|determined|inferred) (in|from) (the )?(context|passage|text)",

    # "not mentioned/stated in the context" variants
    r"not (explicitly )?(mentioned|stated|provided|specified|discussed|addressed) in (the )?(context|passage|text)",
    r"no (explicit|direct|clear) (information|mention|statement|answer|reference) (is )?(available|provided|given|found|present) in (the )?(context|passage|text)",

    # "there is no information" variants
    r"there is no (explicit |direct |clear )?(information|mention|statement|answer|detail|reference) (in|within) (the )?(context|passage|text)",

    # Self-correction markers (model caught itself guessing mid-answer)
    r"I must revise",
    r"let me revise (this|my)",
    r"I need to revise",
    r"I should revise",
    r"I cannot (find|determine|identify|answer)",
    r"I can't (find|determine|identify|answer)",

    # "does not provide/explain/mention" without "context" noun but clearly a refusal
    r"does not (explicitly )?(provide|explain|state|mention) (the )?(reason|answer|information|detail)",

    # Explicit uncertainty about context coverage
    r"(is not|isn't) (explicitly )?(mentioned|stated|described|explained|covered) in (the )?(context|passage|text)",
    r"no (information|detail|mention) (about|regarding|on) .{0,60} (is )?(provided|given|available|found) in (the )?(context|passage|text)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


def is_refusal(answer: str) -> tuple:
    """
    Return (True, matched_pattern) if answer contains refusal language,
    else (False, None).
    """
    for pattern, compiled in zip(REFUSAL_PATTERNS, _COMPILED):
        if compiled.search(answer):
            return True, pattern
    return False, None


def scan_split(data: list, split_name: str) -> tuple:
    """
    Scan a split for refusals.

    Returns:
        (clean_data, removed_data, report_lines)
    """
    clean, removed = [], []
    report = []

    for ex in data:
        flagged, matched = is_refusal(ex['answer'])
        if flagged:
            removed.append({**ex, '_matched_pattern': matched})
            report.append({
                'split':    split_name,
                'label':    'factual' if ex['epistemic_label'] == 1 else 'speculative',
                'question': ex['question'],
                'answer':   ex['answer'][:200],
                'pattern':  matched,
            })
        else:
            clean.append(ex)

    return clean, removed, report


def label_counts(data: list) -> dict:
    factual     = sum(1 for ex in data if ex['epistemic_label'] == 1)
    speculative = sum(1 for ex in data if ex['epistemic_label'] == 0)
    return {'factual': factual, 'speculative': speculative, 'total': len(data)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true',
                        help="Apply changes (overwrite splits). Default: dry run.")
    args = parser.parse_args()

    splits = {
        'train': Path('data/splits/train.json'),
        'val':   Path('data/splits/val.json'),
        'test':  Path('data/splits/test.json'),
    }

    all_removed = []
    results     = {}

    print("\n" + "=" * 65)
    print("DATASET CLEANING REPORT")
    print("=" * 65)

    for name, path in splits.items():
        data = json.loads(path.read_text(encoding='utf-8'))
        before = label_counts(data)

        clean, removed, report = scan_split(data, name)
        after = label_counts(clean)

        all_removed.extend(removed)
        results[name] = {'clean': clean, 'removed': removed, 'path': path}

        print(f"\n  {name.upper()} split")
        print(f"    Before : {before['total']} examples  "
              f"(factual={before['factual']}, speculative={before['speculative']})")
        print(f"    Removed: {len(removed)} refusals  "
              f"(factual={sum(1 for r in removed if r['epistemic_label']==1)}, "
              f"speculative={sum(1 for r in removed if r['epistemic_label']==0)})")
        print(f"    After  : {after['total']} examples  "
              f"(factual={after['factual']}, speculative={after['speculative']})")

        if removed:
            print(f"\n    Sample removed answers:")
            for r in removed[:3]:
                lbl = 'factual' if r['epistemic_label'] == 1 else 'speculative'
                print(f"      [{lbl}] Q: {r['question'][:80]}")
                print(f"              A: {r['answer'][:120]!r}")
                print()

    print(f"\n  TOTAL removed: {len(all_removed)} examples across all splits")

    # Pattern frequency
    from collections import Counter
    pattern_counts = Counter(r['_matched_pattern'] for r in all_removed)
    print(f"\n  Top matched patterns:")
    for pattern, count in pattern_counts.most_common(5):
        print(f"    {count:3d}×  {pattern}")

    print("=" * 65)

    if not args.apply:
        print("\n  DRY RUN — no files modified. Re-run with --apply to apply changes.")
        return

    # ── Apply: backup originals, save cleaned ─────────────────────────────────
    backup_dir = Path('data/splits/backup')
    backup_dir.mkdir(parents=True, exist_ok=True)

    for name, result in results.items():
        src = result['path']
        dst = backup_dir / src.name
        shutil.copy2(src, dst)
        print(f"  Backed up {src} → {dst}")

    for name, result in results.items():
        path  = result['path']
        clean = result['clean']
        path.write_text(
            json.dumps(clean, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        print(f"  Saved cleaned {name}: {len(clean)} examples → {path}")

    # Save removed examples for inspection
    removed_path = Path('data/splits/removed_refusals.json')
    removed_path.write_text(
        json.dumps(all_removed, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f"\n  Removed examples saved to {removed_path} for inspection")
    print("\n  ✓ Cleaning complete.")


if __name__ == '__main__':
    main()
