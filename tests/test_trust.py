import unittest

import torch

from models.module import schedule_inverse_range, schedule_inverse_range_mono
from models.trust import PriorTrustGate, prior_trust_loss


class PriorTrustGateTest(unittest.TestCase):
    def test_initial_output_and_gradient(self):
        gate = PriorTrustGate(initial_trust=0.9)
        shape = (2, 1, 4, 5)
        aligned = torch.full(shape, 0.6)
        coarse_depth = torch.full((2, 4, 5), 2.0)
        confidence = torch.full((2, 4, 5), 0.8)
        edge = torch.ones(shape)
        inverse_min = torch.full((2, 4, 5), 1.0)
        inverse_max = torch.full((2, 4, 5), 0.25)

        score = gate(
            aligned, coarse_depth, confidence, edge, inverse_min, inverse_max
        )
        self.assertEqual(score.shape, shape)
        self.assertTrue(torch.allclose(score, torch.full_like(score, 0.9), atol=1e-6))

        score.mean().backward()
        final_weight_grad = gate.net[-1].weight.grad
        self.assertIsNotNone(final_weight_grad)
        self.assertTrue(torch.isfinite(final_weight_grad).all())
        self.assertGreater(final_weight_grad.abs().sum().item(), 0.0)

    def test_trust_loss_has_finite_gradient(self):
        logits = torch.zeros((1, 1, 3, 4), requires_grad=True)
        trust = torch.sigmoid(logits)
        mono = torch.ones((1, 1, 3, 4))
        base = torch.full((1, 1, 3, 4), 2.0)
        gt = torch.ones((1, 3, 4))
        valid = torch.ones((1, 1, 3, 4), dtype=torch.bool)
        interval = torch.full((1, 3, 4), 0.1)

        loss, oracle_rate = prior_trust_loss(
            trust, mono, base, gt, valid, interval
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(oracle_rate.item(), 0.99)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


class CandidateInjectionTest(unittest.TestCase):
    def setUp(self):
        self.inverse_min = torch.full((1, 4, 4), 1.0)
        self.inverse_max = torch.full((1, 4, 4), 0.25)
        self.mono_depth = torch.full((1, 1, 8, 8), 0.5)
        self.ref_edge = torch.ones((1, 1, 8, 8))
        self.coarse_depth = torch.full((1, 4, 4), 2.0)
        self.confidence = torch.ones((1, 4, 4))
        self.aligned_inverse_mono = torch.full((1, 1, 4, 4), 0.6)

    def _run(self, trust_score):
        return schedule_inverse_range_mono(
            self.inverse_min,
            self.inverse_max,
            4,
            8,
            8,
            self.mono_depth,
            self.ref_edge,
            0.8,
            self.coarse_depth,
            self.confidence,
            aligned_inverse_mono=self.aligned_inverse_mono,
            trust_score=trust_score,
        )

    def test_one_trust_matches_original_replacement(self):
        original, _, _ = schedule_inverse_range_mono(
            self.inverse_min,
            self.inverse_max,
            4,
            8,
            8,
            self.mono_depth,
            self.ref_edge,
            0.8,
            self.coarse_depth,
            self.confidence,
            aligned_inverse_mono=self.aligned_inverse_mono,
            trust_score=None,
        )
        trusted, _, metadata = self._run(torch.ones((1, 1, 4, 4)))

        self.assertTrue(torch.allclose(original, trusted, atol=1e-6))
        self.assertEqual(metadata['trust_score'].shape, (1, 1, 8, 8))
        self.assertEqual(metadata['trust_valid_mask'].dtype, torch.bool)

    def test_zero_trust_keeps_standard_mvs_candidates(self):
        baseline = schedule_inverse_range(
            self.inverse_min, self.inverse_max, 4, 8, 8
        )
        gated, _, _ = self._run(torch.zeros((1, 1, 4, 4)))

        self.assertTrue(torch.allclose(baseline, gated, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
