"""Reliability estimation for monocular candidate injection.

The gate is intentionally small: it judges whether an aligned monocular
candidate should replace the closest candidate in the current MVS range.  It
does not estimate depth by itself and therefore keeps the original MVS
geometry as the fallback.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorTrustGate(nn.Module):
    """Predict a per-pixel trust value for an aligned monocular prior."""

    def __init__(self, hidden_channels=16, initial_trust=0.9):
        super().__init__()
        if not 0.0 < initial_trust < 1.0:
            raise ValueError("initial_trust must be in (0, 1)")

        self.net = nn.Sequential(
            nn.Conv2d(4, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

        # Start close to the always-trust MonoMVSNet baseline.  Earlier layers
        # still receive gradients once the final layer starts moving.
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, math.log(initial_trust / (1.0 - initial_trust)))

    def forward(
        self,
        aligned_inverse_mono,
        coarse_depth,
        confidence,
        ref_edge,
        inverse_min_depth,
        inverse_max_depth,
    ):
        target_size = coarse_depth.shape[-2:]
        aligned_inverse_mono = F.interpolate(
            aligned_inverse_mono,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        confidence = _as_single_channel(confidence, target_size)
        if ref_edge is None:
            ref_edge = torch.zeros_like(confidence)
        else:
            ref_edge = _as_single_channel(ref_edge, target_size)

        coarse_inverse = torch.reciprocal(coarse_depth.unsqueeze(1).clamp_min(1e-6))
        relative_disagreement = (
            (aligned_inverse_mono - coarse_inverse).abs()
            / coarse_inverse.abs().clamp_min(1e-6)
        ).clamp(max=10.0)

        inverse_min_depth = _as_single_channel(inverse_min_depth, target_size)
        inverse_max_depth = _as_single_channel(inverse_max_depth, target_size)
        interval_center = 0.5 * (inverse_min_depth + inverse_max_depth)
        half_interval = 0.5 * (inverse_min_depth - inverse_max_depth).abs().clamp_min(1e-6)
        interval_offset = (
            (aligned_inverse_mono - interval_center).abs() / half_interval
        ).clamp(max=10.0)

        features = torch.cat(
            [relative_disagreement, confidence, ref_edge, interval_offset], dim=1
        )
        return torch.sigmoid(self.net(features))


def prior_trust_loss(
    trust_score,
    mono_candidate_depth,
    base_candidate_depth,
    depth_gt,
    valid_mask,
    depth_interval,
    temperature=1.0,
):
    """Supervise trust using which candidate is closer to ground truth.

    The soft target approaches one when the aligned monocular candidate is
    better than the original MVS candidate, and approaches zero in the
    opposite case.  All target construction is detached from the model graph.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    trust_score = _squeeze_channel(trust_score)
    mono_candidate_depth = _squeeze_channel(mono_candidate_depth)
    base_candidate_depth = _squeeze_channel(base_candidate_depth)
    valid_mask = _squeeze_channel(valid_mask).bool()

    if depth_gt.shape[-2:] != trust_score.shape[-2:]:
        depth_gt = F.interpolate(
            depth_gt.unsqueeze(1),
            size=trust_score.shape[-2:],
            mode="nearest",
        ).squeeze(1)
    if depth_interval.shape[-2:] != trust_score.shape[-2:]:
        depth_interval = F.interpolate(
            depth_interval.unsqueeze(1),
            size=trust_score.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    valid_mask = valid_mask & torch.isfinite(depth_gt) & (depth_gt > 0)
    if not torch.any(valid_mask):
        return trust_score.sum() * 0.0, trust_score.new_tensor(0.0)

    with torch.no_grad():
        mono_error = (mono_candidate_depth - depth_gt).abs()
        base_error = (base_candidate_depth - depth_gt).abs()
        scale = (temperature * depth_interval.abs()).clamp_min(1e-6)
        target = torch.sigmoid((base_error - mono_error) / scale)

    loss = F.binary_cross_entropy(trust_score[valid_mask], target[valid_mask])
    oracle_rate = (target[valid_mask] > 0.5).float().mean()
    return loss, oracle_rate


def _as_single_channel(tensor, size):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.shape[-2:] != size:
        tensor = F.interpolate(
            tensor.float(), size=size, mode="bilinear", align_corners=False
        ).to(dtype=tensor.dtype)
    return tensor


def _squeeze_channel(tensor):
    if tensor.dim() == 4:
        if tensor.shape[1] != 1:
            raise ValueError("expected a single-channel tensor")
        return tensor[:, 0]
    return tensor
