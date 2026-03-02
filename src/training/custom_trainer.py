"""Custom trainer with dual loss (generation + routing) and adversarial co-evolution."""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, List, Optional, Tuple
import logging

from .thermo_spatial_loss import ThermoSpatialLoss


class EpistemicRoutingTrainer(Trainer):
    """
    Custom trainer that combines generation loss with routing loss.

    Loss = L_generation + λ × L_routing
    L_routing = L_BCE + λ_contrastive × L_contrastive

    Where:
    - L_generation: Standard causal language modeling loss
    - L_BCE: Binary cross-entropy between predictor output and ground truth
    - L_contrastive: Centroid margin loss pushing factual/speculative clusters apart
    - λ, λ_contrastive: Hyperparameters controlling loss weights

    The trainer supports adversarial co-evolution via the _is_predictor_update_step flag:
    - Generator update step: gradient flows through activations to LoRA params;
      predictor runs in eval mode, its params are not updated (no optimizer call).
    - Predictor update step: activations are detached; gradient reaches predictor
      params only; generator receives no gradient.
    """

    def __init__(
        self,
        predictor: torch.nn.Module,
        lambda_routing: float = 1.0,
        lambda_contrastive: float = 0.5,
        contrastive_margin: float = 2.0,
        routing_layer: int = -2,
        thermo_layers: Optional[List[int]] = None,
        lambda_thermo: float = 1.0,
        thermo_margin: float = 0.05,
        *args,
        **kwargs
    ):
        """
        Initialize custom trainer.

        Args:
            predictor: EpistemicPredictor MLP (should be in train mode, unfrozen)
            lambda_routing: Weight for total routing loss
            lambda_contrastive: Weight for spatial centroid loss within routing loss
            contrastive_margin: L2 margin between factual/speculative cluster centroids
            routing_layer: Which transformer layer to extract activations from for
                predictor BCE. Non-negative = 0-indexed; negative = relative from end.
            thermo_layers: Layer indices for turbulence computation (e.g. [17, 18, 19]).
                If None, thermodynamic loss is disabled.
            lambda_thermo: Weight for thermodynamic turbulence loss.
            thermo_margin: Minimum required T_spec - T_fact per same-context pair.
            *args, **kwargs: Arguments for base Trainer
        """
        super().__init__(*args, **kwargs)

        self.predictor = predictor
        self.lambda_routing = lambda_routing
        self.lambda_contrastive = lambda_contrastive
        self.contrastive_margin = contrastive_margin
        self.routing_layer = routing_layer
        self.thermo_layers = thermo_layers
        self.lambda_thermo = lambda_thermo

        # ThermoSpatialLoss handles both turbulence and spatial separation.
        # Disabled if thermo_layers is None.
        if thermo_layers is not None:
            self.thermo_loss_fn = ThermoSpatialLoss(
                thermo_margin=thermo_margin,
                spatial_margin=contrastive_margin
            )
        else:
            self.thermo_loss_fn = None

        # Flag set by the alternating training loop in phase2_lora.py to control
        # gradient flow direction. False = generator update, True = predictor update.
        self._is_predictor_update_step = False

        logging.info(
            f"EpistemicRoutingTrainer initialized: "
            f"routing_layer={routing_layer}, λ_routing={lambda_routing}, "
            f"thermo_layers={thermo_layers}, λ_thermo={lambda_thermo}, "
            f"λ_contrastive={lambda_contrastive}, margin={contrastive_margin}"
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Compute combined loss: generation + routing (BCE + thermo + spatial).

        Gradient flow is controlled by self._is_predictor_update_step:
        - False (generator update): activations and thermo hidden states remain
          in-graph; gradients flow through routing_loss to LoRA params.
          ThermoSpatialLoss forces differential turbulence between factual/speculative.
        - True (predictor update): activations are detached; only BCE gradient
          reaches predictor params; thermo loss is skipped.

        Args:
            model: The model being trained
            inputs: Input batch containing:
                - input_ids, attention_mask (standard)
                - epistemic_labels: Ground truth labels (0=speculative, 1=factual)
            return_outputs: Whether to return model outputs

        Returns:
            Combined loss (and optionally outputs)
        """
        epistemic_labels = inputs.pop("epistemic_labels", None)

        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False
        )

        generation_loss = outputs.loss

        routing_loss = torch.tensor(0.0, device=generation_loss.device)
        bce_loss = torch.tensor(0.0, device=generation_loss.device)
        l_thermo = torch.tensor(0.0, device=generation_loss.device)
        l_spatial = torch.tensor(0.0, device=generation_loss.device)

        if epistemic_labels is not None and self.lambda_routing > 0:
            # Attention mask for last-token extraction
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1   # [batch]
                batch_indices = torch.arange(
                    outputs.hidden_states[0].size(0),
                    device=outputs.hidden_states[0].device
                )

            def _last_token(h):
                if attention_mask is not None:
                    return h[batch_indices, seq_lengths]
                return h[:, -1, :]

            # --- Routing layer activation for predictor BCE ---
            hidden_idx = self.routing_layer + 1 if self.routing_layer >= 0 else self.routing_layer
            routing_hidden = outputs.hidden_states[hidden_idx]
            activations = _last_token(routing_hidden)   # [batch, hidden]

            ground_truth = epistemic_labels.unsqueeze(1).float()

            if self._is_predictor_update_step:
                # Predictor update: detached activations, BCE only.
                # Generator receives no routing gradient.
                self.predictor.train()
                predictions = self.predictor(activations.detach().float())
                bce_loss = F.binary_cross_entropy(predictions, ground_truth)
                routing_loss = bce_loss

            else:
                # Generator update: in-graph activations.
                # Gradient flows: (bce + thermo + spatial) → activations → LoRA.
                self.predictor.eval()
                predictions = self.predictor(activations.float())
                bce_loss = F.binary_cross_entropy(predictions, ground_truth)

                # --- ThermoSpatial loss (generator step only) ---
                if self.thermo_loss_fn is not None and self.thermo_layers:
                    thermo_hidden = {
                        idx: _last_token(
                            outputs.hidden_states[idx + 1 if idx >= 0 else idx]
                        )
                        for idx in self.thermo_layers
                    }
                    l_thermo, l_spatial, turbulences, delta_magnitudes = self.thermo_loss_fn(
                        thermo_hidden, epistemic_labels
                    )
                    if self.state.global_step % self.args.logging_steps == 0:
                        fact_mask = epistemic_labels == 1
                        spec_mask = epistemic_labels == 0
                        log_dict = {}
                        if turbulences is not None:
                            t_f = turbulences[fact_mask].mean().item() if fact_mask.any() else 0.0
                            t_s = turbulences[spec_mask].mean().item() if spec_mask.any() else 0.0
                            log_dict.update({"turb_factual_mean": t_f, "turb_speculative_mean": t_s, "turb_gap": t_f - t_s})
                        if delta_magnitudes is not None:
                            m_f = delta_magnitudes[fact_mask].mean().item() if fact_mask.any() else 0.0
                            m_s = delta_magnitudes[spec_mask].mean().item() if spec_mask.any() else 0.0
                            log_dict.update({"delta_mag_factual": m_f, "delta_mag_speculative": m_s, "delta_mag_gap": m_f - m_s})
                        if log_dict:
                            self.log(log_dict)

                routing_loss = (
                    bce_loss
                    + self.lambda_thermo * l_thermo
                    + self.lambda_contrastive * l_spatial
                )

        total_loss = generation_loss + self.lambda_routing * routing_loss

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss_generation": generation_loss.item(),
                "loss_routing_bce": bce_loss.item(),
                "loss_thermo": l_thermo.item(),
                "loss_spatial": l_spatial.item(),
                "loss_routing": routing_loss.item(),
                "loss_total": total_loss.item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None
    ):
        """
        Perform an evaluation step.

        Overridden to handle epistemic labels properly.
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # Remove epistemic_labels from inputs for prediction
            epistemic_labels = inputs.pop("epistemic_labels", None)

            # Standard prediction
            outputs = model(**inputs, output_hidden_states=True)
            loss = outputs.loss

            # Re-add epistemic labels for loss computation if needed
            if epistemic_labels is not None:
                inputs["epistemic_labels"] = epistemic_labels

        if prediction_loss_only:
            return (loss, None, None)

        logits = outputs.logits
        labels = inputs.get("labels")

        return (loss, logits, labels)


if __name__ == "__main__":
    print("✓ EpistemicRoutingTrainer module loaded successfully")
