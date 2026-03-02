"""Question generation using Claude API (synchronous and batch)."""

import os
import json
import logging
import time
from typing import List, Dict
import anthropic
from pathlib import Path


def generate_question_pair_prompt(context: str) -> str:
    """
    Create prompt for generating factual/speculative question pairs.

    Args:
        context: The source context

    Returns:
        Formatted prompt
    """
    prompt = f"""Given the following context, generate TWO questions with IDENTICAL grammatical structure.

CRITICAL REQUIREMENT: Both questions must use the same question word (What/Why/How/When/Who) and
the same sentence pattern. The ONLY difference between them is whether the answer is present in
the context. This is essential — do NOT use "How might..." for one and "What was..." for the other.

Instructions:
1. Choose a grammatical template (e.g. "What was the primary reason that X happened?")
2. Write a FACTUAL question using this template — the answer must appear verbatim in the context
3. Write a SPECULATIVE question using the EXACT SAME template structure — the answer must NOT be in the context

Context:
{context}

Return your response as JSON:
{{
    "question_template": "...",
    "factual_question": "...",
    "factual_answer": "...",
    "speculative_question": "...",
    "speculative_answer": "..."
}}

The factual answer MUST appear verbatim in the context.
The speculative answer should be reasonable but NOT directly stated.
Both questions MUST start with the same question word and follow the same grammatical pattern."""

    return prompt


def _parse_json_response(response_text: str) -> dict:
    """
    Extract and parse JSON from a model response string.
    Handles responses wrapped in markdown code fences.
    """
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0].strip()
    else:
        json_str = response_text.strip()
    return json.loads(json_str)


def generate_with_anthropic(contexts: List[str], model: str = "claude-haiku-4-5-20251001") -> List[Dict]:
    """
    Generate question pairs using Anthropic API (synchronous, one request per context).

    Args:
        contexts: List of contexts
        model: Claude model to use

    Returns:
        List of question pair dictionaries
    """
    client = anthropic.Anthropic(api_key=Path("claude-key.txt").read_text().strip())

    results = []

    for i, context in enumerate(contexts):
        try:
            logging.info(f"Generating questions for context {i+1}/{len(contexts)}")

            message = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": generate_question_pair_prompt(context)}
                ]
            )

            question_pair = _parse_json_response(message.content[0].text)
            question_pair['context'] = context
            results.append(question_pair)

            # Rate limiting
            time.sleep(1.0)

        except Exception as e:
            logging.error(f"Error generating questions for context {i}: {e}")
            continue

    return results


def submit_anthropic_batch(
    contexts: List[str],
    model: str = "claude-haiku-4-5-20251001",
    meta_path: str = "data/batch/batch_meta.json",
) -> str:
    """
    Submit all contexts as a single Anthropic Message Batch.

    Cheaper and faster than synchronous calls for large datasets.
    The batch ID is saved to disk so results can be retrieved later
    with retrieve_anthropic_batch().

    Args:
        contexts: List of contexts
        model: Claude model to use
        meta_path: Where to save the batch ID for later retrieval

    Returns:
        Batch ID string
    """
    client = anthropic.Anthropic(api_key=Path("claude-key.txt").read_text().strip())

    logging.info(f"Submitting Anthropic batch with {len(contexts)} requests (model: {model})...")

    requests = [
        {
            "custom_id": f"context-{i}",
            "params": {
                "model": model,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": generate_question_pair_prompt(context)}
                ],
            },
        }
        for i, context in enumerate(contexts)
    ]

    batch = client.messages.batches.create(requests=requests)

    # Save batch ID for retrieval
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump({"batch_id": batch.id, "model": model, "num_contexts": len(contexts)}, f, indent=2)

    logging.info(f"Batch submitted. ID: {batch.id}")
    logging.info(f"Processing status: {batch.processing_status}")
    logging.info(f"Batch metadata saved to {meta_path}")
    logging.info("")
    logging.info("Once complete, retrieve results with:")
    logging.info(f"  python -X utf8 main.py --phase data --batch-id {batch.id}")

    return batch.id


def parse_local_batch_results(
    results_file: str,
    contexts: List[str],
) -> List[Dict]:
    """
    Parse a locally saved Anthropic batch results JSONL file.

    Use this when you have already downloaded the results file manually
    instead of retrieving via the API.

    Args:
        results_file: Path to the results JSONL file
        contexts: Original list of contexts in submission order

    Returns:
        List of question pair dictionaries
    """
    context_lookup = {f"context-{i}": ctx for i, ctx in enumerate(contexts)}
    results = []
    failed = 0

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            custom_id = record["custom_id"]
            context = context_lookup.get(custom_id)

            if record["result"]["type"] != "succeeded":
                logging.error(f"Request {custom_id} {record['result']['type']}")
                failed += 1
                continue

            try:
                content = record["result"]["message"]["content"][0]["text"]
                question_pair = _parse_json_response(content)
                question_pair["context"] = context
                results.append(question_pair)
            except Exception as e:
                logging.error(f"Failed to parse response for {custom_id}: {e}")
                failed += 1
                continue

    logging.info(f"Parsed {len(results)} results ({failed} failed / skipped)")
    return results


def retrieve_anthropic_batch(
    batch_id: str,
    contexts: List[str],
    poll_interval: int = 60,
) -> List[Dict]:
    """
    Poll an Anthropic batch for completion and parse results.

    Args:
        batch_id: Anthropic batch ID (e.g. "msgbatch_abc123")
        contexts: Original list of contexts in submission order
        poll_interval: Seconds between status polls

    Returns:
        List of question pair dictionaries (same format as generate_with_anthropic)
    """
    client = anthropic.Anthropic(api_key=Path("claude-key.txt").read_text().strip())

    # ---- Poll for completion ----
    batch = client.messages.batches.retrieve(batch_id)
    logging.info(f"Checking batch {batch_id} — status: {batch.processing_status}")

    while batch.processing_status == "in_progress":
        counts = batch.request_counts
        logging.info(
            f"Status: in_progress | "
            f"processing: {counts.processing} | "
            f"succeeded: {counts.succeeded} | "
            f"errored: {counts.errored} — "
            f"polling again in {poll_interval}s"
        )
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        raise RuntimeError(
            f"Batch {batch_id} has unexpected status '{batch.processing_status}'."
        )

    counts = batch.request_counts
    logging.info(
        f"Batch ended. succeeded: {counts.succeeded}, "
        f"errored: {counts.errored}, expired: {counts.expired}"
    )

    # ---- Stream and parse results ----
    context_lookup = {f"context-{i}": ctx for i, ctx in enumerate(contexts)}

    results = []
    failed = 0

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        context = context_lookup.get(custom_id)

        if result.result.type != "succeeded":
            logging.error(f"Request {custom_id} {result.result.type}: {result.result}")
            failed += 1
            continue

        try:
            content = result.result.message.content[0].text
            question_pair = _parse_json_response(content)
            question_pair["context"] = context
            results.append(question_pair)
        except Exception as e:
            logging.error(f"Failed to parse response for {custom_id}: {e}")
            failed += 1
            continue

    logging.info(f"Parsed {len(results)} results ({failed} failed / skipped)")

    return results


def generate_oracle_dataset(
    num_contexts: int = 10000,
    output_path: str = "data/processed/oracle_dataset.json",
    api_provider: str = "anthropic",
    api_model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 10,
    batch_id: str = None,
    batch_file: str = None,
):
    """
    Generate complete oracle dataset.

    For synchronous generation (slow, no batch_id needed):
        api_provider="anthropic", no --batch-id flag

    For batch generation (fast, cheap):
        First run:  api_provider="anthropic-batch", no --batch-id  → submits batch, exits
        Second run: api_provider="anthropic-batch", --batch-id <ID> → retrieves and builds dataset

    Args:
        num_contexts: Number of contexts to process
        output_path: Where to save dataset
        api_provider: "anthropic" (sync) or "anthropic-batch"
        api_model: Claude model to use
        batch_size: Unused (kept for interface compatibility)
        batch_id: Anthropic batch ID for retrieval step
    """
    from .dataset_builder import load_squad_contexts, create_train_val_test_splits, save_dataset

    logging.info(f"Generating oracle dataset with {num_contexts} contexts...")

    contexts = load_squad_contexts(num_contexts)

    if batch_file is not None:
        dataset = parse_local_batch_results(batch_file, contexts)

    elif api_provider == "anthropic":
        dataset = generate_with_anthropic(contexts, api_model)

    elif api_provider == "anthropic-batch":
        if batch_id is None:
            # Submit batch and exit — user re-runs with --batch-id once complete
            submit_anthropic_batch(contexts, api_model)
            return
        else:
            dataset = retrieve_anthropic_batch(batch_id, contexts)

    else:
        raise ValueError(
            f"Unknown api_provider '{api_provider}'. Use 'anthropic' or 'anthropic-batch'."
        )

    # Build flat dataset
    flat_dataset = []
    for item in dataset:
        flat_dataset.append({
            'context': item['context'],
            'question': item['factual_question'],
            'answer': item['factual_answer'],
            'epistemic_label': 1  # Factual
        })
        flat_dataset.append({
            'context': item['context'],
            'question': item['speculative_question'],
            'answer': item['speculative_answer'],
            'epistemic_label': 0  # Speculative
        })

    save_dataset(flat_dataset, output_path)

    train_data, val_data, test_data = create_train_val_test_splits(flat_dataset)

    output_dir = Path(output_path).parent.parent / "splits"
    save_dataset(train_data, output_dir / "train.json")
    save_dataset(val_data, output_dir / "val.json")
    save_dataset(test_data, output_dir / "test.json")

    logging.info(f"Oracle dataset generation complete!")
    logging.info(f"  Total examples: {len(flat_dataset)}")
    logging.info(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_context = "Paris is the capital and most populous city of France. The city has a population of 2.1 million."

    print(f"Test context: {test_context}\n")
    print("Synchronous:  python -X utf8 main.py --phase data")
    print("Batch submit: python -X utf8 main.py --phase data  (with api.provider: anthropic-batch)")
    print("Batch fetch:  python -X utf8 main.py --phase data --batch-id msgbatch_xxx")
