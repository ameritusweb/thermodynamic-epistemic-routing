"""
Paired sampler for adversarial co-evolution training.

Yields dataset indices in (factual, speculative) interleaved order,
keeping same-context pairs contiguous within each batch. This enables
the pairwise thermodynamic loss to control for context-level baseline
turbulence: T_speculative[i] - T_factual[i] is computed over pairs from
the same context, normalising out how complex each context is.
"""

import torch
from torch.utils.data import Sampler
from collections import defaultdict
from typing import List, Dict, Iterator
import logging


class PairedSampler(Sampler):
    """
    Samples indices as (factual, speculative) pairs from the same source context.

    Guarantees:
    - Every yielded pair shares the same source context.
    - A batch of size 2k contains exactly k complete pairs.
    - Shuffling operates at the pair level — pairs are never split across batches.
    - The i-th factual and i-th speculative in any batch are from the same context,
      so T_spec[i] - T_fact[i] is a context-normalised turbulence difference.

    Usage:
        sampler = PairedSampler(raw_train_data, shuffle=True, seed=42)
        loader = DataLoader(
            tokenized_dataset, sampler=sampler,
            batch_size=8, collate_fn=collator, drop_last=True
        )

    Note: raw_train_data and tokenized_dataset must have identical ordering
    (both derived from the same list without reordering).
    """

    def __init__(self, dataset: List[Dict], shuffle: bool = True, seed: int = 42):
        """
        Args:
            dataset: Raw list of examples, each with 'context' and 'epistemic_label'.
            shuffle: Shuffle at pair granularity each epoch.
            seed: Base random seed; epoch offset added via set_epoch().
        """
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Group indices by context and label
        context_to_factual: Dict[str, List[int]] = defaultdict(list)
        context_to_speculative: Dict[str, List[int]] = defaultdict(list)

        for i, ex in enumerate(dataset):
            ctx = ex['context']
            if ex['epistemic_label'] == 1:
                context_to_factual[ctx].append(i)
            else:
                context_to_speculative[ctx].append(i)

        # Build (factual_idx, speculative_idx) pairs from the same context
        self.pairs: List[tuple] = []
        n_skipped = 0
        for ctx, fact_indices in context_to_factual.items():
            spec_indices = context_to_speculative.get(ctx, [])
            for f, s in zip(fact_indices, spec_indices):
                self.pairs.append((f, s))
            n_skipped += abs(len(fact_indices) - len(spec_indices))

        logging.info(
            f"PairedSampler: {len(self.pairs)} pairs built "
            f"({n_skipped} unpaired examples skipped)"
        )

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to vary shuffling order."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs) * 2

    def __iter__(self) -> Iterator[int]:
        pairs = list(self.pairs)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            perm = torch.randperm(len(pairs), generator=g).tolist()
            pairs = [pairs[i] for i in perm]
        for fact_idx, spec_idx in pairs:
            yield fact_idx
            yield spec_idx
