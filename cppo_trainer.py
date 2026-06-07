"""
CPPO Trainer: Group Relative Policy Optimization with Confidence-based Pruning.

Extends TRL's GRPOTrainer to prune low-confidence samples within each generation
group. In each group of G samples generated from the same prompt, the G×P samples
with the smallest absolute advantages are pruned (their loss contribution is set
to zero), keeping only the G×(1-P) samples with the strongest advantage signals.

Reference: CPPO (NeurIPS 2025)
"""

import torch
from trl import GRPOTrainer


class CPPOTrainer(GRPOTrainer):
    """GRPO Trainer with Confidence-based Pruning.

    In standard GRPO, all G completions per prompt contribute equally to the
    policy gradient, including samples whose advantage is close to zero. CPPO
    prunes these "uninformative" samples, keeping only those with strong
    positive or negative advantage signals.

    Parameters
    ----------
    cppo_pruning_rate : float
        Pruning rate P ∈ [0, 1). In each group of G samples:
        - floor(G × P) samples with smallest |advantage| are pruned.
        - G - floor(G × P) samples are kept (minimum 1 per group).
        P=0.0: no pruning (equivalent to standard GRPO).
        P=0.5: prune half of all samples per group.
    """

    def __init__(self, cppo_pruning_rate: float = 0.0, *args, **kwargs):
        if not 0.0 <= cppo_pruning_rate < 1.0:
            raise ValueError(
                f"cppo_pruning_rate must be in [0, 1), got {cppo_pruning_rate}"
            )
        self.cppo_pruning_rate = cppo_pruning_rate
        super().__init__(*args, **kwargs)

        if self.cppo_pruning_rate > 0:
            keep_ratio = 1.0 - self.cppo_pruning_rate
            keep_per_group = max(1, int(self.num_generations * keep_ratio))
            print(
                f"[CPPO] Pruning rate P={self.cppo_pruning_rate:.2f} | "
                f"G={self.num_generations} | "
                f"keeping {keep_per_group}/{self.num_generations} samples per group "
                f"({keep_ratio:.0%})"
            )

    # ------------------------------------------------------------------
    # Override: compute_loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The CPPOTrainer does not support returning outputs")

        # ── 1. Per-token log-probabilities ──
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # ── 2. KL divergence (K3 estimator) ──
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1.0
        )

        # ── 3. Policy-gradient loss ──
        #     r(θ) = exp(log π_θ - log π_θ.detach()) ≡ 1 numerically,
        #     but gradients flow through π_θ as if r were a function of θ.
        advantages = inputs["advantages"]  # shape: (B,)
        per_token_loss = (
            torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        )
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        # ══════════════════════════════════════════════════════════════
        # CPPO Pruning  (the ONLY change from GRPOTrainer.compute_loss)
        # ══════════════════════════════════════════════════════════════
        if self.cppo_pruning_rate > 0:
            G = self.num_generations
            P = self.cppo_pruning_rate
            keep_per_group = max(1, int(G * (1.0 - P)))

            B = advantages.size(0)
            num_groups = B // G

            # Reshape → (num_prompts, G) for per-group top-k selection
            adv_grouped = advantages.view(num_groups, G)            # (N, G)
            abs_adv = adv_grouped.abs()                             # (N, G)

            # Select keep_per_group samples with largest |advantage|
            _, topk_idx = torch.topk(abs_adv, keep_per_group, dim=1)  # (N, K)

            # Build keep mask: (N, G) → flatten to (B,)
            keep_mask = torch.zeros(num_groups, G, device=advantages.device)
            keep_mask.scatter_(1, topk_idx, 1.0)
            keep_mask = keep_mask.view(-1)                          # (B,)

            # Zero out the loss for pruned (low-|advantage|) samples
            per_token_loss = per_token_loss * keep_mask.unsqueeze(1)

            # Re-normalize: divide only by the number of *kept* samples
            per_seq_loss = (
                (per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1).clamp(min=1.0)
            )
            loss = per_seq_loss.sum() / keep_mask.sum().clamp(min=1)
        else:
            # Standard GRPO aggregation (identical to parent)
            loss = (
                (per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1)
            ).mean()

        # ── 4. Logging (identical to GRPOTrainer) ──
        completion_length = (
            self.accelerator.gather_for_metrics(
                completion_mask.sum(1)
            ).float().mean().item()
        )
        self._metrics["completion_length"].append(completion_length)

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss
