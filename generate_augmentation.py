"""
Step 2 & 3: Generate augmentation data.

Submits a single Anthropic batch that generates two new example types per context:

  Hard-negative speculative — answer is NOT in context, but written with
    confident, authoritative, causal phrasing. No hedging language allowed.
    Targets the dominant failure mode: speculative answers that fool the
    predictor because they sound like factual answers.

  Complex factual — answer IS in context, but requires synthesising
    information from at least two non-adjacent sentences. Targets the
    secondary failure mode: factual answers that are predicted as speculative
    because they require multi-hop reasoning.

Usage:
    # Step 1: submit batch (exits immediately after submitting)
    python generate_augmentation.py --submit --n-contexts 1000

    # Step 2: retrieve results and merge into train.json
    python generate_augmentation.py --retrieve --batch-id msgbatch_XXXXX

    # Optional: merge without retrieving (if you have the results file already)
    python generate_augmentation.py --merge --results-file data/batch/augmentation_results.jsonl
"""

import argparse
import json
import logging
import re
import time
import random
from pathlib import Path
from typing import List, Dict

import anthropic

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

KEY_FILE      = Path('claude-key.txt')
TRAIN_PATH    = Path('data/splits/train.json')
META_PATH     = Path('data/batch/augmentation_batch_meta.json')
RESULTS_PATH  = Path('data/batch/augmentation_results.jsonl')
AUGMENTED_PATH = Path('data/augmentation/augmented_examples.json')
MODEL         = 'claude-haiku-4-5-20251001'

# Hedging phrases — any generated answer containing these is discarded
HEDGE_PATTERNS = [
    r'\bprobably\b', r'\blikely\b', r'\bmight\b', r'\bmay have\b', r'\bmight have\b',
    r'\bit is thought\b', r'\bit is believed\b', r'\bit is possible\b',
    r'\bperhaps\b', r'\bcould have been\b', r'\bit seems\b', r'\bapparently\b',
    r'\bnot specified\b', r'\bnot mentioned\b', r'\bnot stated\b', r'\bnot explicitly\b',
    r'\bcannot (be )?determined\b', r'\bunclear\b', r'\bunknown\b',
    r'\bI cannot\b', r"\bI can't\b", r'\bI must revise\b',
    r'\bcontext does not\b', r"\bcontext doesn't\b",
]
_HEDGE_RE = re.compile('|'.join(HEDGE_PATTERNS), re.IGNORECASE)


# ── Prompts ───────────────────────────────────────────────────────────────────

def hard_negative_prompt(context: str) -> str:
    return f"""You are generating a training example for an epistemic routing classifier.

Your task: given the context below, write ONE speculative question-answer pair.

RULES:
1. The question must ask about a cause, reason, or motivation (use "What was the primary reason that..." or "Why did..." or "What caused...").
2. The answer must NOT be directly stated or inferable from the context.
3. CRITICAL — the answer must be written with complete confidence. NO hedging words:
   - FORBIDDEN: probably, likely, might, may have, perhaps, seems, could, unclear, unknown, I cannot, not specified, not mentioned, not stated, context does not
   - REQUIRED: write as if you know the answer with certainty. Use causal language: "because", "due to", "in order to", "as a result of", "to ensure", "given that"
4. The answer should be specific, plausible-sounding, and 1-3 sentences.

EXAMPLE of a correct hard-negative answer:
  Q: What was the primary reason that the company moved its headquarters to Dublin?
  A: The relocation was driven by Ireland's favorable corporate tax rate of 12.5%, which provided substantial savings compared to the previous jurisdiction's 35% rate.

(That answer sounds authoritative and specific, but it is invented — not in the context.)

Context:
{context}

Return JSON only:
{{
  "question": "...",
  "answer": "..."
}}"""


def complex_factual_prompt(context: str) -> str:
    return f"""You are generating a training example for an epistemic routing classifier.

Your task: given the context below, write ONE complex factual question-answer pair.

RULES:
1. The question must ask about a cause, reason, or motivation (use "What was the primary reason that..." or "Why did..." or "What caused...").
2. The answer MUST be grounded in the context — every claim in your answer must be supported by specific text in the context.
3. CRITICAL — the answer must require synthesising information from at least TWO different sentences or clauses in the context. Do not pick a question whose answer is a single verbatim phrase.
4. The answer should be 2-4 sentences and written in a complete, explanatory style.
5. Do not start the answer with "According to the context" or similar meta-phrases.

Context:
{context}

Return JSON only:
{{
  "question": "...",
  "answer": "...",
  "source_sentences": ["sentence 1 from context that supports the answer", "sentence 2..."]
}}"""


# ── Batch submission ──────────────────────────────────────────────────────────

def submit_batch(contexts: List[str]) -> str:
    client = anthropic.Anthropic(api_key=KEY_FILE.read_text().strip())

    requests = []
    for i, ctx in enumerate(contexts):
        requests.append({
            "custom_id": f"hard-neg-{i}",
            "params": {
                "model": MODEL,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": hard_negative_prompt(ctx)}],
            },
        })
        requests.append({
            "custom_id": f"complex-fact-{i}",
            "params": {
                "model": MODEL,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": complex_factual_prompt(ctx)}],
            },
        })

    logging.info(f"Submitting {len(requests)} requests ({len(contexts)} contexts × 2 types)...")
    batch = client.messages.batches.create(requests=requests)

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps({
        "batch_id":     batch.id,
        "model":        MODEL,
        "n_contexts":   len(contexts),
        "n_requests":   len(requests),
    }, indent=2))

    logging.info(f"Batch ID: {batch.id}")
    logging.info(f"Metadata saved to {META_PATH}")
    logging.info(f"Once complete, run:")
    logging.info(f"  python generate_augmentation.py --retrieve --batch-id {batch.id}")
    return batch.id


# ── Batch retrieval ───────────────────────────────────────────────────────────

def retrieve_batch(batch_id: str, contexts: List[str]) -> Path:
    client = anthropic.Anthropic(api_key=KEY_FILE.read_text().strip())

    batch = client.messages.batches.retrieve(batch_id)
    logging.info(f"Status: {batch.processing_status}")

    while batch.processing_status == "in_progress":
        counts = batch.request_counts
        logging.info(
            f"  processing={counts.processing}  succeeded={counts.succeeded}  "
            f"errored={counts.errored} — polling in 60s"
        )
        time.sleep(60)
        batch = client.messages.batches.retrieve(batch_id)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        for result in client.messages.batches.results(batch_id):
            f.write(json.dumps(result.model_dump()) + '\n')

    logging.info(f"Results saved to {RESULTS_PATH}")
    return RESULTS_PATH


# ── Parse and validate results ────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


def parse_results(results_file: Path, contexts: List[str]) -> List[Dict]:
    """
    Parse batch results into flat example dicts.

    Hard-negative answers are validated to contain no hedging language.
    Complex-factual answers are validated to have source_sentences populated.
    Both are discarded if JSON parsing fails.
    """
    context_lookup = {str(i): ctx for i, ctx in enumerate(contexts)}

    hard_neg_raw: dict = {}
    complex_fact_raw: dict = {}

    succeeded = failed = hedged = 0

    with open(results_file, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            cid = rec['custom_id']

            result = rec.get('result', {})
            if result.get('type') != 'succeeded':
                failed += 1
                continue

            try:
                text = result['message']['content'][0]['text']
                parsed = _parse_json(text)
            except Exception:
                failed += 1
                continue

            # Determine type and index
            if cid.startswith('hard-neg-'):
                idx = cid[len('hard-neg-'):]
                hard_neg_raw[idx] = parsed
            elif cid.startswith('complex-fact-'):
                idx = cid[len('complex-fact-'):]
                complex_fact_raw[idx] = parsed

            succeeded += 1

    logging.info(f"Raw results: {succeeded} succeeded, {failed} failed/errored")

    examples = []

    for idx, item in hard_neg_raw.items():
        ctx = context_lookup.get(idx)
        if ctx is None:
            continue
        q = item.get('question', '').strip()
        a = item.get('answer', '').strip()
        if not q or not a:
            continue
        # Discard if hedging language slipped through
        if _HEDGE_RE.search(a):
            hedged += 1
            continue
        examples.append({
            'context':         ctx,
            'question':        q,
            'answer':          a,
            'epistemic_label': 0,           # speculative
            'augmentation_type': 'hard_negative',
        })

    for idx, item in complex_fact_raw.items():
        ctx = context_lookup.get(idx)
        if ctx is None:
            continue
        q = item.get('question', '').strip()
        a = item.get('answer', '').strip()
        src = item.get('source_sentences', [])
        if not q or not a or not src:
            continue
        examples.append({
            'context':         ctx,
            'question':        q,
            'answer':          a,
            'epistemic_label': 1,           # factual
            'augmentation_type': 'complex_factual',
        })

    hard_n  = sum(1 for e in examples if e['augmentation_type'] == 'hard_negative')
    complex_n = sum(1 for e in examples if e['augmentation_type'] == 'complex_factual')
    logging.info(
        f"Valid examples: {len(examples)} total  "
        f"({hard_n} hard-negative, {complex_n} complex-factual, {hedged} discarded for hedging)"
    )
    return examples


# ── Merge into train.json ─────────────────────────────────────────────────────

def merge_into_train(examples: List[Dict]):
    train = json.loads(TRAIN_PATH.read_text(encoding='utf-8'))
    before_f = sum(1 for e in train if e['epistemic_label'] == 1)
    before_s = sum(1 for e in train if e['epistemic_label'] == 0)

    # Strip augmentation_type before saving (keep schema clean)
    clean = [{k: v for k, v in ex.items() if k != 'augmentation_type'} for ex in examples]

    # Shuffle new examples in rather than appending as a block
    random.seed(42)
    merged = train + clean
    random.shuffle(merged)

    after_f = sum(1 for e in merged if e['epistemic_label'] == 1)
    after_s = sum(1 for e in merged if e['epistemic_label'] == 0)

    TRAIN_PATH.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')

    logging.info(f"Merged into {TRAIN_PATH}")
    logging.info(f"  Before: {len(train)} (factual={before_f}, speculative={before_s})")
    logging.info(f"  Added : {len(clean)} ({sum(1 for e in examples if e['augmentation_type']=='hard_negative')} hard-neg, "
                 f"{sum(1 for e in examples if e['augmentation_type']=='complex_factual')} complex-fact)")
    logging.info(f"  After : {len(merged)} (factual={after_f}, speculative={after_s})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--submit',   action='store_true', help="Submit batch job")
    group.add_argument('--retrieve', action='store_true', help="Retrieve completed batch and merge")
    group.add_argument('--merge',    action='store_true', help="Parse local results file and merge")

    parser.add_argument('--batch-id',     type=str, default=None)
    parser.add_argument('--results-file', type=str, default=str(RESULTS_PATH))
    parser.add_argument('--n-contexts',   type=int, default=1000,
                        help="Number of training contexts to sample for augmentation (default: 1000)")
    parser.add_argument('--no-merge',     action='store_true',
                        help="Parse and save augmented examples without merging into train.json")
    args = parser.parse_args()

    # Load contexts once (needed by submit and retrieve/merge for index lookup)
    train = json.loads(TRAIN_PATH.read_text(encoding='utf-8'))
    random.seed(42)
    sample = random.sample(train, min(args.n_contexts, len(train)))
    contexts = [ex['context'] for ex in sample]
    logging.info(f"Sampled {len(contexts)} contexts from cleaned train set")

    if args.submit:
        submit_batch(contexts)
        return

    if args.retrieve:
        if not args.batch_id:
            # Try to load from saved metadata
            if META_PATH.exists():
                meta = json.loads(META_PATH.read_text())
                args.batch_id = meta['batch_id']
                logging.info(f"Loaded batch_id from {META_PATH}: {args.batch_id}")
            else:
                parser.error("--batch-id required for --retrieve")
        retrieve_batch(args.batch_id, contexts)
        results_file = RESULTS_PATH
    else:
        results_file = Path(args.results_file)

    # Parse results (used by both --retrieve and --merge)
    examples = parse_results(results_file, contexts)

    # Save augmented examples separately for inspection
    AUGMENTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUGMENTED_PATH.write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    logging.info(f"Augmented examples saved to {AUGMENTED_PATH}")

    if args.no_merge:
        logging.info("--no-merge set: skipping train.json update")
        return

    merge_into_train(examples)
    logging.info("Done. Re-run train_multi_feature_predictor.py to retrain the predictor.")


if __name__ == '__main__':
    main()
