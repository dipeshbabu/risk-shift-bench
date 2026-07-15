import unittest

from experiments.pilot_verified_robustness import (
    ScoreWeights,
    quantile,
    score_from_row,
    score_weight_grid,
    summarize_selection,
)


class PilotVerifiedRobustnessTests(unittest.TestCase):
    def test_score_weight_grid_is_complete(self):
        weights = score_weight_grid()
        self.assertEqual(len(weights), 81)
        self.assertEqual(len(set(weights)), 81)

    def test_score_from_row_matches_benchmark_formula(self):
        row = {
            "mean_final_bankroll": "100",
            "cvar_5_final_bankroll": "80",
            "target_probability": "0.5",
            "ruin_probability": "0.1",
            "mean_max_drawdown": "20",
        }
        self.assertEqual(score_from_row(row, ScoreWeights()), 160.0)

    def test_selection_summary_uses_equal_domain_weighting(self):
        effects = {
            "blackjack_v4": [
                {"task": "b", "candidate_delta": 10.0, "candidate_relative_delta": 0.1}
            ],
            "portfolio_v2": [
                {"task": "p", "candidate_delta": 20.0, "candidate_relative_delta": 0.2}
            ],
            "inventory_v1": [
                {"task": "i", "candidate_delta": -5.0, "candidate_relative_delta": -0.3}
            ],
        }
        accepted = {
            "blackjack_v4": {"b"},
            "portfolio_v2": {"p"},
            "inventory_v1": set(),
        }
        summary = summarize_selection(effects, accepted, "test")
        self.assertAlmostEqual(summary["equal_domain_relative"], 0.1)
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["harmful"], 0)

    def test_quantile_interpolates(self):
        self.assertEqual(quantile([0.0, 10.0], 0.25), 2.5)


if __name__ == "__main__":
    unittest.main()
