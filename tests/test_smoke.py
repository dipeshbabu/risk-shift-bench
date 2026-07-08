import unittest

from risk_preference_inference.blackjack import DecisionState
from risk_preference_inference.active_query import select_informative_states
from risk_preference_inference.benchmark import run_benchmark
from risk_preference_inference.benchmark import EpisodeResult, summarize_results
from risk_preference_inference.envs import RiskTask
from risk_preference_inference.evaluation import evaluate
from risk_preference_inference.multiround_distributions import final_bankroll_distribution
from risk_preference_inference.objectives import mean
from risk_preference_inference.policy_registry import (
    core_policies,
    learned_adaptive_cvar_policy,
    state_adaptive_utility_policy,
    strong_baseline_grid,
)
from risk_preference_inference.adaptive_search import search_adaptive_utility_policy
from risk_preference_inference.ablations import run_ablation_study
from risk_preference_inference.toy_envs import run_toy_benchmark
from risk_preference_inference.run_management import paper_run_paths, required_artifacts
from risk_preference_inference.policy import action_probabilities
from risk_preference_inference.risk_models import (
    CumulativeProspectModel,
    EntropicRiskModel,
    ExpectedValueModel,
    OptimizedCertaintyEquivalentModel,
    ProspectUtilityModel,
)
from risk_preference_inference.splits import make_split
from risk_preference_inference.synthetic import generate_synthetic_records


class SmokeTests(unittest.TestCase):
    def test_action_probabilities_are_valid(self):
        state = DecisionState(player_cards=(10, 6), dealer_card=10)
        probs = action_probabilities(state, ExpectedValueModel())
        self.assertEqual(set(probs), {"hit", "stand"})
        self.assertLess(abs(sum(probs.values()) - 1.0), 1e-9)

    def test_synthetic_generation_produces_records(self):
        records = generate_synthetic_records(subjects=2, decisions_per_subject=5, seed=1)
        self.assertEqual(len(records), 10)
        self.assertTrue(all(record.action_taken in {"hit", "stand"} for record in records))

    def test_prospect_model_scores_actions(self):
        state = DecisionState(player_cards=(11, 7), dealer_card=6, current_bankroll=480)
        probs = action_probabilities(state, ProspectUtilityModel(alpha=0.7, loss_aversion=2.0))
        self.assertGreaterEqual(probs["hit"], 0.0)
        self.assertLessEqual(probs["hit"], 1.0)
        self.assertGreaterEqual(probs["stand"], 0.0)
        self.assertLessEqual(probs["stand"], 1.0)

    def test_additional_risk_models_score_actions(self):
        state = DecisionState(player_cards=(10, 6), dealer_card=9)
        for model in (
            CumulativeProspectModel(),
            EntropicRiskModel(),
            OptimizedCertaintyEquivalentModel(),
        ):
            probs = action_probabilities(state, model, max_depth=1)
            self.assertLess(abs(sum(probs.values()) - 1.0), 1e-9)

    def test_split_and_evaluation_metrics(self):
        records = generate_synthetic_records(subjects=3, decisions_per_subject=8, seed=2)
        train, test = make_split(records, protocol="cross_subject", seed=2)
        self.assertTrue(train)
        self.assertTrue(test)
        result = evaluate(test, ExpectedValueModel())
        self.assertGreaterEqual(result.brier_score, 0.0)
        self.assertGreaterEqual(result.calibration_error, 0.0)

    def test_active_query_returns_candidates(self):
        candidates = select_informative_states(
            [ExpectedValueModel(), ProspectUtilityModel(alpha=0.7, loss_aversion=2.0)],
            limit=3,
            max_depth=1,
        )
        self.assertEqual(len(candidates), 3)
        self.assertGreaterEqual(candidates[0].score, candidates[-1].score)

    def test_risk_benchmark_runs(self):
        task = RiskTask(name="test-task", rounds=3, initial_bankroll=120, target_bankroll=160)
        episodes, summaries = run_benchmark(tasks=[task], episodes=2, seed=3, hand_depth=1)
        self.assertTrue(episodes)
        self.assertTrue(summaries)
        self.assertTrue(all(summary.episodes == 2 for summary in summaries))

    def test_exact_multiround_distribution(self):
        task = RiskTask(name="exact-test", rounds=1, initial_bankroll=120, target_bankroll=160)
        distribution = final_bankroll_distribution(task, core_policies()[1], rounds=1, hand_depth=1, grid=20)
        self.assertLess(abs(sum(prob for _, prob in distribution) - 1.0), 1e-9)
        self.assertGreater(mean(distribution), 0.0)

    def test_learned_adaptive_policy_scores(self):
        task = RiskTask(name="learned-test", rounds=2, initial_bankroll=120, target_bankroll=160)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, target_bankroll=160)
        probs = learned_adaptive_cvar_policy().action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})

    def test_state_adaptive_utility_policy_scores(self):
        task = RiskTask(name="utility-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, target_bankroll=150)
        probs = state_adaptive_utility_policy().action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})
        self.assertLess(abs(sum(probs.values()) - 1.0), 1e-9)

    def test_state_adaptive_utility_search_runs(self):
        train_task = RiskTask(name="utility-train", rounds=2, initial_bankroll=120, target_bankroll=150)
        test_task = RiskTask(name="utility-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        result = search_adaptive_utility_policy(
            train_tasks=[train_task],
            test_tasks=[test_task],
            episodes=2,
            seed=4,
            hand_depth=1,
            smoke=True,
        )
        self.assertTrue(result.test_summaries)

    def test_ablation_study_runs(self):
        task = RiskTask(name="ablation-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        summaries, aggregate_scores, task_scores = run_ablation_study(
            tasks=[task],
            episodes=2,
            seed=5,
            hand_depth=1,
        )
        self.assertTrue(summaries)
        self.assertTrue(aggregate_scores)
        self.assertTrue(task_scores)

    def test_regime_adaptive_policy_scores(self):
        task = RiskTask(name="regime-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, target_bankroll=150)
        policy = next(policy for policy in strong_baseline_grid() if policy.name == "regime_adaptive_ensemble")
        probs = policy.action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})
        self.assertLess(abs(sum(probs.values()) - 1.0), 1e-9)

    def test_regime_policy_does_not_target_seek_on_mean_task(self):
        task = RiskTask(name="mean-like", rounds=25, initial_bankroll=500, target_bankroll=650)
        state = DecisionState((10, 6), 10, current_bankroll=540, initial_bankroll=500, target_bankroll=650)
        policy = next(policy for policy in strong_baseline_grid() if policy.name == "regime_adaptive_ensemble")
        delegate = policy._delegate(state, task, rounds_remaining=8, peak_bankroll=560)
        self.assertNotEqual(delegate.name, "regime_target_mean")

    def test_toy_benchmark_runs(self):
        results, summaries = run_toy_benchmark(episodes=2, seed=1)
        self.assertTrue(results)
        self.assertTrue(summaries)

    def test_target_probability_uses_ever_hit_flag(self):
        task = RiskTask(name="target-test", target_bankroll=150)
        policy = core_policies()[1]
        summary = summarize_results(
            [
                EpisodeResult("target-test", policy.name, 0, 120, 100, 40, False, True, 3),
                EpisodeResult("target-test", policy.name, 1, 140, 100, 40, False, False, 3),
            ],
            task,
            policy,
        )
        self.assertEqual(summary.target_probability, 0.5)

    def test_benchmark_uses_paired_seeds_across_policies(self):
        task = RiskTask(name="seed-test", rounds=1, initial_bankroll=120, target_bankroll=160)
        episodes, _summaries = run_benchmark(tasks=[task], policies=core_policies()[:2], episodes=3, seed=9, hand_depth=1)
        seeds_by_policy = {}
        for episode in episodes:
            seeds_by_policy.setdefault(episode.policy, []).append(episode.seed)
        self.assertEqual(len(set(tuple(seeds) for seeds in seeds_by_policy.values())), 1)

    def test_required_artifacts_include_manifest_and_configs(self):
        paths = paper_run_paths("artifacts/test_run")
        artifacts = required_artifacts(paths, include_exact=False)
        self.assertIn("artifacts/test_run/manifest.json", artifacts)
        self.assertIn("artifacts/test_run/configs/benchmark_config.json", artifacts)
        self.assertIn("artifacts/test_run/ablations/summary.json", artifacts)


if __name__ == "__main__":
    unittest.main()
