"""
Thermodynamic spatial loss for epistemic routing.

Combines two complementary losses:
  L_thermo  — pairwise margin loss on layer-to-layer turbulence (kinematic separation)
  L_spatial — centroid margin loss on deepest targeted layer (spatial separation)

L_thermo forces the LoRA weights to develop two distinct computational dynamics:
- Factual passes: low turbulence (laminar flow — representation settles early)
- Speculative passes: high turbulence (model searches, revises, synthesises)

Unlike spatial contrastive loss, which only constrains the endpoint of the
forward pass, L_thermo watches the entire trajectory across the targeted layers.
A model cannot fake a factual response by projecting a turbulent computation
into the factual activation region at the final layer.

Requires PairedSampler: batches must contain interleaved (factual, speculative)
pairs from the same context, so that T_spec[i] - T_fact[i] is context-normalised.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ThermoSpatialLoss(nn.Module):
    """
    Joint thermodynamic + spatial contrastive loss.

    Args:
        thermo_margin: Minimum required T_spec - T_fact per same-context pair.
                       If speculative turbulence doesn't exceed factual by this
                       margin, a penalty is applied.
        spatial_margin: Minimum required L2 distance between factual and
                        speculative centroids at the deepest targeted layer.
    """

    def __init__(self, thermo_margin: float = 0.05, spatial_margin: float = 4.0):
        super().__init__()
        self.thermo_margin = thermo_margin
        self.spatial_margin = spatial_margin

    def forward(
        self,
        hidden_states_dict: Dict[int, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states_dict: {layer_idx: Tensor[batch, hidden_dim]}
                Must remain in the computation graph on generator update steps
                so that gradients flow back to LoRA weights.
                Pass detached tensors if used during predictor update steps.
            labels: [batch] — 1=factual, 0=speculative.
                PairedSampler guarantees interleaved ordering so that
                labels[0::2] == 1 and labels[1::2] == 0 within each batch.

        Returns:
            l_thermo:     Scalar — pairwise turbulence margin loss.
            l_spatial:    Scalar — centroid L2 margin loss.
            turbulences:  [batch] detached per-example turbulence values for
                          logging/monitoring, or None if fewer than 2 layers.
        """
        layer_indices = sorted(hidden_states_dict.keys())
        device = labels.device

        factual_mask = labels == 1
        speculative_mask = labels == 0

        # Anchor zero tensor that stays in the computation graph
        zero = hidden_states_dict[layer_indices[0]].sum() * 0.0

        if factual_mask.sum() == 0 or speculative_mask.sum() == 0:
            return zero, zero, None

        # ------------------------------------------------------------------
        # 1. Compute deltas (layer contributions) — in-graph for gradient flow.
        #    delta_n = h_{n+1} - h_n  ≈ f_layer(h_n), directly modified by LoRA.
        # ------------------------------------------------------------------
        deltas = []
        for i in range(len(layer_indices) - 1):
            l1, l2 = layer_indices[i], layer_indices[i + 1]
            deltas.append(hidden_states_dict[l2].float() - hidden_states_dict[l1].float())

        # ------------------------------------------------------------------
        # 2. Delta magnitudes (in-graph) — the primary thermodynamic signal.
        #
        #    Factual retrieval produces larger deltas (~25-28) than speculative
        #    confabulation (~22-24), with a consistent gap of ~4 units and a
        #    clean gradient: d||delta||/d(h) = delta/||delta|| (unit vector,
        #    never vanishing).
        #
        #    Curvature (cosine between consecutive deltas) is also computed
        #    for logging but showed no consistent signal and is not used in loss.
        # ------------------------------------------------------------------
        if deltas:
            delta_norms = torch.stack(
                [d.norm(dim=-1) for d in deltas], dim=0
            ).mean(dim=0)   # [batch], in-graph
        else:
            delta_norms = torch.zeros(labels.size(0), device=device)

        # Curvature — for logging only (not in loss)
        if len(deltas) >= 2:
            turb_sum = torch.zeros(labels.size(0), device=device)
            for i in range(len(deltas) - 1):
                cos_sim = F.cosine_similarity(deltas[i], deltas[i + 1], dim=-1)
                turb_sum = turb_sum + (1.0 - cos_sim)
            turbulences = (turb_sum / (len(deltas) - 1)).detach()
        else:
            turbulences = torch.zeros(labels.size(0), device=device)

        # ------------------------------------------------------------------
        # 3. Pairwise magnitude margin loss — enforce gap per same-context pair
        # ------------------------------------------------------------------
        mag_fact = delta_norms[factual_mask]
        mag_spec = delta_norms[speculative_mask]
        n_pairs = min(mag_fact.size(0), mag_spec.size(0))

        if n_pairs > 0:
            l_thermo = F.relu(
                self.thermo_margin - (mag_fact[:n_pairs] - mag_spec[:n_pairs])
            ).mean()
        else:
            l_thermo = zero

        # ------------------------------------------------------------------
        # 4. Spatial centroid margin loss on deepest targeted layer
        # ------------------------------------------------------------------
        h_final = hidden_states_dict[layer_indices[-1]].float()
        c_fact = h_final[factual_mask].mean(dim=0)
        c_spec = h_final[speculative_mask].mean(dim=0)
        l_spatial = F.relu(self.spatial_margin - torch.norm(c_fact - c_spec, p=2))

        return l_thermo, l_spatial, turbulences, delta_norms.detach()
