import unittest

from risk_shift_bench.blackjack import DecisionState
from risk_shift_bench.active_query import select_informative_states
from risk_shift_bench.benchmark import run_benchmark
from risk_shift_bench.benchmark import EpisodeResult, summarize_results
from risk_shift_bench.envs import RiskTask
from risk_shift_bench.envs import (
    benchmark_tasks,
    frontier_blind_audit_tasks,
    frontier_audit_tasks,
    frontier_confirmation_audit_tasks,
    frontier_confirmation_audit_v2_tasks,
    frontier_confirmation_audit_v3_tasks,
    frontier_benchmark_tasks,
    frontier_development_tasks,
    frontier_final_audit_tasks,
    frontier_holdout_tasks,
    target_family_split,
)
from risk_shift_bench.evaluation import evaluate
from risk_shift_bench.family_selector import (
    FamilyPromotionParams,
    FamilyPromotionPolicy,
    learn_family_promotions,
    task_family,
)
from risk_shift_bench.incumbent_switch import (
    incumbent_switch_candidates,
    incumbent_switch_policy,
    search_incumbent_switch,
)
from risk_shift_bench.lcb_selector import (
    LCBSelectorParams,
    LowerConfidenceSelectorPolicy,
    profiles_from_scores as lcb_profiles_from_scores,
    risk_adjusted_validation_score,
    search_lcb_selector,
)
from risk_shift_bench.meta_selector import (
    AdvantageKnnMetaPolicy,
    MetaSelectorProfile,
    MetaSelectorParams,
    build_profiles,
    search_meta_selector,
    search_meta_selector_cv,
)
from risk_shift_bench.multiround_distributions import final_bankroll_distribution
from risk_shift_bench.objectives import mean
from risk_shift_bench.policy_registry import (
    core_policies,
    learned_adaptive_cvar_policy,
    learned_mixture_policy,
    searched_learned_mixture_policy,
    signed_regime_learned_policy,
    state_action_blend_policy,
    state_adaptive_utility_policy,
    target_branch_searched_policy,
    strong_baseline_grid,
)
from risk_shift_bench.portfolio_benchmark import (
    PortfolioState,
    portfolio_policy_grid,
    run_portfolio_benchmark,
)
from risk_shift_bench.portfolio_envs import (
    PortfolioTask,
    portfolio_audit_tasks,
    portfolio_confirmation_tasks,
    portfolio_development_tasks,
    portfolio_holdout_tasks,
    portfolio_tasks,
)
from risk_shift_bench.portfolio_lcb_selector import (
    PortfolioLCBParams,
    PortfolioLCBSelectorPolicy,
    profiles_from_scores as portfolio_lcb_profiles_from_scores,
    search_portfolio_lcb_selector,
)
from risk_shift_bench.adaptive_search import search_adaptive_utility_policy, search_learned_mixture_policy
from risk_shift_bench.ablations import run_ablation_study
from risk_shift_bench.multiseed import paired_policy_deltas, run_multiseed_evaluation
from risk_shift_bench.toy_envs import run_toy_benchmark
from risk_shift_bench.run_management import paper_run_paths, required_artifacts
from risk_shift_bench.policy import action_probabilities
from risk_shift_bench.portfolio_selector import (
    PortfolioProfile,
    PortfolioSelectorParams,
    TaskFeaturePortfolioPolicy,
    search_portfolio_selector,
    task_features,
)
from risk_shift_bench.risk_models import (
    CumulativeProspectModel,
    EntropicRiskModel,
    ExpectedValueModel,
    OptimizedCertaintyEquivalentModel,
    ProspectUtilityModel,
)
from risk_shift_bench.robust_gate_search import robust_gate_candidate_params, robust_gate_policy, search_robust_gate
from risk_shift_bench.state_action_blend_search import (
    search_state_action_blend,
    state_action_blend_candidates,
    state_action_blend_from_params,
)
from risk_shift_bench.splits import make_split
from risk_shift_bench.synthetic import generate_synthetic_records
from risk_shift_bench.statistics import paired_score_deltas, paired_score_report
from risk_shift_bench.target_search import (
    evaluate_promotion_gate,
    PromotionGateResult,
    search_target_branch_policy,
    target_branch_candidate_policy,
    target_candidate_params,
)
from experiments.conformal_router import (
    ConformalAdvantageRouter,
    RouterParams,
    RoutingProfile,
    finite_sample_upper_quantile,
)
from experiments.frontier_v4_tasks import (
    blackjack_confirmation_v4_tasks,
    portfolio_confirmation_v2_tasks,
)
from experiments.inventory_domain import (
    InventoryTask,
    inventory_calibration_tasks,
    inventory_confirmation_tasks,
    inventory_development_tasks,
    inventory_policy_grid,
    inventory_task_features,
    run_inventory_benchmark,
)
from experiments.pilot_verifier import PilotGateParams, one_sided_sign_test_p, verify_promotion


class SmokeTests(unittest.TestCase):
    def test_frontier_extension_factorial_suites_are_complete(self):
        blackjack = blackjack_confirmation_v4_tasks()
        portfolio = portfolio_confirmation_v2_tasks()
        inventory = inventory_confirmation_tasks()
        self.assertEqual(len(blackjack), 40)
        self.assertEqual(len(portfolio), 32)
        self.assertEqual(len(inventory), 32)
        self.assertEqual(len({task.name for task in blackjack}), len(blackjack))
        self.assertEqual(len({task.name for task in portfolio}), len(portfolio))
        self.assertEqual(len({task.name for task in inventory}), len(inventory))

    def test_inventory_splits_are_disjoint_and_benchmark_runs(self):
        development = {task.name for task in inventory_development_tasks()}
        calibration = {task.name for task in inventory_calibration_tasks()}
        confirmation = {task.name for task in inventory_confirmation_tasks()}
        self.assertFalse(development & calibration)
        self.assertFalse(development & confirmation)
        self.assertFalse(calibration & confirmation)
        task = InventoryTask(name="inventory-smoke", periods=3)
        episodes, summaries = run_inventory_benchmark(
            [task],
            policies=inventory_policy_grid()[:2],
            episodes=2,
            seed=3,
        )
        self.assertEqual(len(episodes), 4)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(inventory_task_features(task)), 13)

    def test_conformal_router_screens_on_fit_and_calibrates_disjoint_tasks(self):
        fit_profiles = [
            RoutingProfile("fit-a", (0.0,), {"base": 0.0, "good": 3.0, "bad": -2.0}),
            RoutingProfile("fit-b", (1.0,), {"base": 0.0, "good": 2.0, "bad": -1.0}),
        ]
        calibration_profiles = [
            RoutingProfile("cal-a", (0.2,), {"base": 0.0, "good": 2.5, "bad": -1.5}),
            RoutingProfile("cal-b", (0.8,), {"base": 0.0, "good": 1.5, "bad": -0.5}),
        ]
        router = ConformalAdvantageRouter(
            fit_profiles=fit_profiles,
            calibration_profiles=calibration_profiles,
            candidate_policies=("good", "bad"),
            params=RouterParams(
                k=1,
                min_fit_evidence=1,
                min_calibration_tasks=2,
                fallback_policy="base",
            ),
            feature_fn=lambda task: task["features"],
        )
        self.assertEqual(router.candidate_policies, ("good",))
        self.assertEqual(router.proposal({"features": (0.1,)}).selected_policy, "good")
        self.assertGreaterEqual(router.calibration.conformal_correction, 0.0)
        self.assertEqual(finite_sample_upper_quantile([1.0, 2.0, 3.0], 0.1), 3.0)

    def test_pilot_gate_uses_exact_one_sided_sign_test(self):
        self.assertLess(one_sided_sign_test_p(7, 0), 0.01)
        accepted = verify_promotion(
            [1.0] * 7,
            PilotGateParams(alpha=0.01, min_nonzero_batches=7),
        )
        rejected = verify_promotion(
            [1.0] * 6 + [-1.0],
            PilotGateParams(alpha=0.01, min_nonzero_batches=7),
        )
        self.assertTrue(accepted.accepted)
        self.assertFalse(rejected.accepted)

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

    def test_portfolio_benchmark_runs(self):
        task = PortfolioTask(name="portfolio-test", periods=3, initial_capital=1000, target_capital=1040, ruin_capital=800)
        episodes, summaries = run_portfolio_benchmark(tasks=[task], policies=portfolio_policy_grid()[:3], episodes=2, seed=3)
        self.assertEqual(len(episodes), 6)
        self.assertEqual(len(summaries), 3)
        self.assertTrue(all(summary.episodes == 2 for summary in summaries))

    def test_portfolio_splits_are_locked_and_disjoint(self):
        dev = {task.name for task in portfolio_development_tasks()}
        holdout = {task.name for task in portfolio_holdout_tasks()}
        audit = {task.name for task in portfolio_audit_tasks()}
        confirmation = {task.name for task in portfolio_confirmation_tasks()}
        self.assertTrue(dev)
        self.assertTrue(holdout)
        self.assertTrue(audit)
        self.assertTrue(confirmation)
        self.assertFalse(dev & holdout)
        self.assertFalse(dev & audit)
        self.assertFalse(dev & confirmation)
        self.assertFalse(holdout & audit)
        self.assertFalse(holdout & confirmation)
        self.assertFalse(audit & confirmation)
        self.assertEqual({task.name for task in portfolio_tasks("portfolio")}, dev | holdout | audit | confirmation)

    def test_portfolio_lcb_selector_promotes_positive_neighbor(self):
        task = PortfolioTask(name="portfolio-lcb", periods=2, initial_capital=1000, target_capital=1040)
        params = PortfolioLCBParams(k=1, min_evidence=1, lcb_scale=0.0, margin=0.0)
        scores = {
            task.name: {
                "learned_mixture_searched": 10.0,
                "signed_regime_learned_ensemble": 15.0,
            }
        }
        profiles = portfolio_lcb_profiles_from_scores([task], scores)
        policy = PortfolioLCBSelectorPolicy(profiles, params)
        state = PortfolioState(capital=1000, initial_capital=1000, peak_capital=1000, periods_remaining=2)
        self.assertEqual(policy.selected_policy_name(task), "signed_regime_learned_ensemble")
        self.assertGreaterEqual(policy.allocation(state, task), 0.0)

    def test_portfolio_lcb_search_runs(self):
        tasks = [
            PortfolioTask(name="portfolio-lcb-a", periods=2, initial_capital=1000, target_capital=1040),
            PortfolioTask(name="portfolio-lcb-b", periods=3, initial_capital=900, target_capital=1010),
        ]
        scores = {
            task.name: {
                "learned_mixture_searched": 10.0,
                "signed_regime_learned_ensemble": 12.0,
            }
            for task in tasks
        }
        result = search_portfolio_lcb_selector(tasks, scores, smoke=True)
        self.assertTrue(result.train_profiles)
        self.assertTrue(result.validation_summaries)

    def test_frontier_benchmark_suite_adds_stress_tasks(self):
        standard_names = {task.name for task in benchmark_tasks()}
        frontier_tasks = frontier_benchmark_tasks()
        frontier_names = {task.name for task in frontier_tasks}
        self.assertGreater(len(frontier_names), len(standard_names))
        self.assertTrue(standard_names <= frontier_names)
        self.assertIn("RiskBlackjack-HiddenDeckShift-v0", frontier_names)
        self.assertIn("RiskBlackjack-NearRuinHighBet-v0", frontier_names)
        self.assertTrue(any(task.episode_card_regimes is not None for task in frontier_tasks))

    def test_frontier_dev_holdout_split_is_locked_and_disjoint(self):
        dev_names = {task.name for task in frontier_development_tasks()}
        holdout_names = {task.name for task in frontier_holdout_tasks()}
        audit_names = {task.name for task in frontier_audit_tasks()}
        final_audit_names = {task.name for task in frontier_final_audit_tasks()}
        blind_audit_names = {task.name for task in frontier_blind_audit_tasks()}
        confirmation_audit_names = {task.name for task in frontier_confirmation_audit_tasks()}
        confirmation_audit_v2_names = {task.name for task in frontier_confirmation_audit_v2_tasks()}
        confirmation_audit_v3_names = {task.name for task in frontier_confirmation_audit_v3_tasks()}
        full_names = {task.name for task in benchmark_tasks("frontier")}
        self.assertTrue(dev_names)
        self.assertTrue(holdout_names)
        self.assertTrue(audit_names)
        self.assertTrue(final_audit_names)
        self.assertTrue(blind_audit_names)
        self.assertTrue(confirmation_audit_names)
        self.assertTrue(confirmation_audit_v2_names)
        self.assertEqual(len(confirmation_audit_v3_names), 40)
        self.assertFalse(dev_names & holdout_names)
        self.assertFalse(dev_names & audit_names)
        self.assertFalse(dev_names & final_audit_names)
        self.assertFalse(dev_names & blind_audit_names)
        self.assertFalse(dev_names & confirmation_audit_names)
        self.assertFalse(dev_names & confirmation_audit_v2_names)
        self.assertFalse(dev_names & confirmation_audit_v3_names)
        self.assertFalse(holdout_names & audit_names)
        self.assertFalse(holdout_names & final_audit_names)
        self.assertFalse(holdout_names & blind_audit_names)
        self.assertFalse(holdout_names & confirmation_audit_names)
        self.assertFalse(holdout_names & confirmation_audit_v2_names)
        self.assertFalse(holdout_names & confirmation_audit_v3_names)
        self.assertFalse(audit_names & final_audit_names)
        self.assertFalse(audit_names & blind_audit_names)
        self.assertFalse(audit_names & confirmation_audit_names)
        self.assertFalse(audit_names & confirmation_audit_v2_names)
        self.assertFalse(audit_names & confirmation_audit_v3_names)
        self.assertFalse(final_audit_names & blind_audit_names)
        self.assertFalse(final_audit_names & confirmation_audit_names)
        self.assertFalse(final_audit_names & confirmation_audit_v2_names)
        self.assertFalse(final_audit_names & confirmation_audit_v3_names)
        self.assertFalse(blind_audit_names & confirmation_audit_names)
        self.assertFalse(blind_audit_names & confirmation_audit_v2_names)
        self.assertFalse(blind_audit_names & confirmation_audit_v3_names)
        self.assertFalse(confirmation_audit_names & confirmation_audit_v2_names)
        self.assertFalse(confirmation_audit_names & confirmation_audit_v3_names)
        self.assertFalse(confirmation_audit_v2_names & confirmation_audit_v3_names)
        self.assertEqual(
            full_names,
            dev_names
            | holdout_names
            | audit_names
            | final_audit_names
            | blind_audit_names
            | confirmation_audit_names
            | confirmation_audit_v2_names
            | confirmation_audit_v3_names,
        )
        self.assertTrue(all("Holdout" in name for name in holdout_names))
        self.assertTrue(all("Audit" in name for name in audit_names))
        self.assertTrue(all("FinalAudit" in name for name in final_audit_names))
        self.assertTrue(all("BlindAudit" in name for name in blind_audit_names))
        self.assertTrue(all("Confirm" in name for name in confirmation_audit_names))
        self.assertTrue(all("ConfirmV2" in name for name in confirmation_audit_v2_names))
        self.assertTrue(all("ConfirmV3" in name for name in confirmation_audit_v3_names))
        self.assertEqual({task.name for task in benchmark_tasks("frontier_final_audit")}, final_audit_names)
        self.assertEqual({task.name for task in benchmark_tasks("frontier_blind_audit")}, blind_audit_names)
        self.assertEqual(
            {task.name for task in benchmark_tasks("frontier_confirmation_audit")},
            confirmation_audit_names,
        )
        self.assertEqual(
            {task.name for task in benchmark_tasks("frontier_confirmation_audit_v2")},
            confirmation_audit_v2_names,
        )
        self.assertEqual(
            {task.name for task in benchmark_tasks("frontier_confirmation_audit_v3")},
            confirmation_audit_v3_names,
        )

    def test_hidden_regime_task_runs(self):
        task = next(task for task in frontier_benchmark_tasks() if task.name == "RiskBlackjack-HiddenDeckShift-v0")
        episodes, summaries = run_benchmark(tasks=[task], policies=core_policies()[:2], episodes=2, seed=17, hand_depth=1)
        self.assertEqual(len(episodes), 4)
        self.assertEqual(len(summaries), 2)

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

    def test_learned_mixture_policy_scores(self):
        task = RiskTask(name="mixture-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, target_bankroll=150)
        probs = learned_mixture_policy().action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})
        self.assertLess(abs(sum(probs.values()) - 1.0), 1e-9)

    def test_searched_learned_mixture_policy_scores(self):
        task = RiskTask(name="searched-mixture-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, target_bankroll=150)
        probs = searched_learned_mixture_policy().action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})

    def test_signed_regime_learned_policy_scores(self):
        task = RiskTask(name="signed-regime-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, target_bankroll=150)
        probs = signed_regime_learned_policy().action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})

    def test_signed_regime_uses_learned_delegate_for_target_task(self):
        task = RiskTask(name="target-regime", rounds=30, initial_bankroll=500, target_bankroll=640)
        state = DecisionState((10, 6), 10, current_bankroll=540, initial_bankroll=500, target_bankroll=640)
        policy = signed_regime_learned_policy()
        delegate = policy._delegate(state, task, rounds_remaining=30, peak_bankroll=500)
        self.assertIn("target_delegate", delegate.name)

    def test_signed_regime_uses_mean_delegate_for_far_target_task(self):
        task = RiskTask(name="target-regime", rounds=30, initial_bankroll=500, target_bankroll=640)
        state = DecisionState((10, 6), 10, current_bankroll=500, initial_bankroll=500, target_bankroll=640)
        policy = signed_regime_learned_policy()
        delegate = policy._delegate(state, task, rounds_remaining=30, peak_bankroll=500)
        self.assertIn("mean_mixture", delegate.name)

    def test_signed_regime_uses_basic_delegate_for_frontier_stress_tasks(self):
        policy = signed_regime_learned_policy()
        severe_task = next(task for task in frontier_benchmark_tasks() if task.name == "RiskBlackjack-NearRuinHighBet-v0")
        severe_state = DecisionState((10, 6), 10, current_bankroll=180, initial_bankroll=180, bet=40, target_bankroll=340)
        severe_delegate = policy._delegate(severe_state, severe_task, rounds_remaining=20, peak_bankroll=180)
        self.assertIn("severe_ruin_basic", severe_delegate.name)

        long_drawdown_task = next(task for task in frontier_benchmark_tasks() if task.name == "RiskBlackjack-LongHorizonTightDrawdown-v0")
        drawdown_state = DecisionState((10, 6), 10, current_bankroll=500, initial_bankroll=500, target_bankroll=760)
        drawdown_delegate = policy._delegate(drawdown_state, long_drawdown_task, rounds_remaining=60, peak_bankroll=500)
        self.assertIn("long_drawdown_basic", drawdown_delegate.name)

        short_target_task = next(task for task in frontier_benchmark_tasks() if task.name == "RiskBlackjack-TightTargetShortHorizon-v0")
        short_target_state = DecisionState((10, 6), 10, current_bankroll=500, initial_bankroll=500, bet=30, target_bankroll=650)
        short_target_delegate = policy._delegate(short_target_state, short_target_task, rounds_remaining=12, peak_bankroll=500)
        self.assertIn("short_target_basic", short_target_delegate.name)

    def test_signed_regime_prioritizes_holdout_uncertainty_gates(self):
        policy = signed_regime_learned_policy()
        holdout_tasks = {task.name: task for task in frontier_holdout_tasks()}

        hidden_drawdown = holdout_tasks["RiskBlackjack-HoldoutVolatileHiddenTarget-v0"]
        hidden_drawdown_state = DecisionState((10, 6), 10, current_bankroll=460, initial_bankroll=460, bet=30, target_bankroll=680)
        hidden_drawdown_delegate = policy._delegate(hidden_drawdown_state, hidden_drawdown, rounds_remaining=28, peak_bankroll=460)
        self.assertIn("hidden_drawdown_oce", hidden_drawdown_delegate.name)

        long_hidden = holdout_tasks["RiskBlackjack-HoldoutBalancedHiddenLong-v0"]
        long_hidden_state = DecisionState((10, 6), 10, current_bankroll=520, initial_bankroll=520, bet=25, target_bankroll=790)
        long_hidden_delegate = policy._delegate(long_hidden_state, long_hidden, rounds_remaining=55, peak_bankroll=520)
        self.assertIn("hidden_long_mean", long_hidden_delegate.name)

        shifted_drawdown = holdout_tasks["RiskBlackjack-HoldoutTenDepletedDrawdown-v0"]
        shifted_drawdown_state = DecisionState((10, 6), 10, current_bankroll=500, initial_bankroll=500, bet=25, target_bankroll=720)
        shifted_drawdown_delegate = policy._delegate(shifted_drawdown_state, shifted_drawdown, rounds_remaining=45, peak_bankroll=500)
        self.assertIn("long_shift_drawdown_mean", shifted_drawdown_delegate.name)

        near_ruin_shift = holdout_tasks["RiskBlackjack-HoldoutExtremeHighRuin-v0"]
        near_ruin_state = DecisionState((10, 6), 10, current_bankroll=260, initial_bankroll=260, bet=40, target_bankroll=620)
        near_ruin_delegate = policy._delegate(near_ruin_state, near_ruin_shift, rounds_remaining=30, peak_bankroll=260)
        self.assertIn("near_ruin_oce", near_ruin_delegate.name)

    def test_state_action_blend_policy_scores(self):
        task = RiskTask(name="blend-test", rounds=20, initial_bankroll=180, bet=40, target_bankroll=340, ruin_bankroll=40)
        state = DecisionState((10, 6), 10, current_bankroll=180, initial_bankroll=180, bet=40, target_bankroll=340)
        probs = state_action_blend_policy().action_probabilities(state, task, rounds_remaining=20, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})
        self.assertLess(abs(sum(probs.values()) - 1.0), 1e-9)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probs.values()))

    def test_state_action_blend_candidate_policy_scores(self):
        params = state_action_blend_candidates(smoke=True)[0]
        policy = state_action_blend_from_params(params)
        task = RiskTask(name="blend-candidate", rounds=3, initial_bankroll=120, bet=20, target_bankroll=150)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, bet=20, target_bankroll=150)
        probs = policy.action_probabilities(state, task, rounds_remaining=3, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})
        self.assertLess(abs(sum(probs.values()) - 1.0), 1e-9)

    def test_state_action_blend_search_runs(self):
        validation_task = RiskTask(name="blend-validation", rounds=2, initial_bankroll=140, bet=20, target_bankroll=170)
        result = search_state_action_blend(
            validation_tasks=[validation_task],
            seeds=[0],
            episodes=1,
            hand_depth=1,
            smoke=True,
        )
        self.assertTrue(result.validation_summaries)
        self.assertTrue(result.candidate_scores)
        self.assertIsInstance(result.validation_score, float)

    def test_incumbent_switch_policy_scores(self):
        params = incumbent_switch_candidates(smoke=True)[0]
        policy = incumbent_switch_policy(params)
        task = RiskTask(name="switch-policy", rounds=3, initial_bankroll=120, bet=20, target_bankroll=150)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, bet=20, target_bankroll=150)
        probs = policy.action_probabilities(state, task, rounds_remaining=3, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})
        self.assertLess(abs(sum(probs.values()) - 1.0), 1e-9)

    def test_incumbent_switch_search_runs(self):
        validation_task = RiskTask(name="switch-validation", rounds=2, initial_bankroll=140, bet=20, target_bankroll=170)
        result = search_incumbent_switch(
            validation_tasks=[validation_task],
            seeds=[0],
            episodes=1,
            hand_depth=1,
            smoke=True,
        )
        self.assertTrue(result.validation_summaries)
        self.assertTrue(result.candidate_scores)
        self.assertIsInstance(result.validation_score, float)

    def test_advantage_knn_meta_selector_runs(self):
        task = RiskTask(name="meta-train", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150)
        params = MetaSelectorParams(k=1)
        profiles, rows = build_profiles(tasks=[task], seeds=[0], episodes=1, hand_depth=1, params=params)
        self.assertTrue(rows)
        self.assertTrue(profiles)
        policy = AdvantageKnnMetaPolicy(profiles=profiles, params=params)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, bet=20, target_bankroll=150)
        probs = policy.action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})

    def test_meta_selector_pairwise_guard_falls_back_on_unstable_advantage(self):
        task = RiskTask(name="meta-guard", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150)
        params = MetaSelectorParams(
            k=2,
            temperature=10.0,
            pairwise_regret_penalty=1.0,
            fallback_policy="signed_regime_learned_ensemble",
        )
        profiles = [
            MetaSelectorProfile(
                task="neighbor-good",
                features=(0.0,) * 12,
                policy_scores={},
                policy_advantages={"signed_regime_learned_ensemble": 0.0, "learned_mixture_default": 12.0},
            ),
            MetaSelectorProfile(
                task="neighbor-bad",
                features=(1.0,) * 12,
                policy_scores={},
                policy_advantages={"signed_regime_learned_ensemble": 0.0, "learned_mixture_default": -8.0},
            ),
        ]
        policy = AdvantageKnnMetaPolicy(profiles=profiles, params=params)
        self.assertEqual(policy.selected_policy_name(task), "signed_regime_learned_ensemble")

    def test_family_promotion_selector_scores(self):
        task = RiskTask(name="family-policy", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150)
        params = FamilyPromotionParams(family_delegates={"default": "expected_value"})
        policy = FamilyPromotionPolicy(params=params)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, bet=20, target_bankroll=150)
        self.assertEqual(policy.selected_policy_name(task), "expected_value")
        probs = policy.action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})

    def test_family_promotion_learning_uses_signed_fallback(self):
        task = RiskTask(name="family-learn", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150)
        scores = {
            task.name: {
                "signed_regime_learned_ensemble": 10.0,
                "expected_value": 14.0,
            }
        }
        params = learn_family_promotions(tasks=[task], scores_by_task=scores, candidate_policies=["expected_value"])
        self.assertEqual(task_family(task), "default")
        self.assertEqual(params.family_delegates["default"], "expected_value")

    def test_lower_confidence_selector_promotes_positive_neighbor(self):
        task = RiskTask(name="lcb-task", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150)
        params = LCBSelectorParams(k=1, min_evidence=1, lcb_scale=0.0)
        scores = {
            task.name: {
                "signed_regime_learned_ensemble": 10.0,
                "expected_value": 15.0,
            }
        }
        profiles = lcb_profiles_from_scores([task], scores, params)
        policy = LowerConfidenceSelectorPolicy(profiles, params)
        self.assertEqual(policy.selected_policy_name(task), "expected_value")

    def test_lower_confidence_selector_requires_all_comparison_baselines(self):
        task = RiskTask(name="lcb-dual-baseline", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150)
        params = LCBSelectorParams(
            k=1,
            min_evidence=1,
            lcb_scale=0.0,
            margin=0.0,
            comparison_policies=("signed_regime_learned_ensemble", "learned_mixture_searched"),
        )
        scores = {
            task.name: {
                "signed_regime_learned_ensemble": 10.0,
                "learned_mixture_searched": 20.0,
                "expected_value": 15.0,
            }
        }
        profiles = lcb_profiles_from_scores([task], scores, params)
        policy = LowerConfidenceSelectorPolicy(profiles, params)
        self.assertEqual(policy.selected_policy_name(task), "learned_mixture_searched")

    def test_lower_confidence_selector_search_runs(self):
        tasks = [
            RiskTask(name="lcb-a", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150),
            RiskTask(name="lcb-b", rounds=3, initial_bankroll=140, bet=20, target_bankroll=180),
        ]
        scores = {
            task.name: {
                "signed_regime_learned_ensemble": 10.0,
                "expected_value": 11.0,
            }
            for task in tasks
        }
        result = search_lcb_selector(tasks=tasks, scores_by_task=scores, smoke=True)
        self.assertTrue(result.train_profiles)
        self.assertTrue(result.validation_summaries)

    def test_lcb_risk_adjusted_validation_penalizes_harmful_promotions(self):
        rows = [
            {
                "selected_policy": "expected_value",
                "fallback_policy": "signed_regime_learned_ensemble",
                "delta_vs_fallback": -4.0,
            },
            {
                "selected_policy": "signed_regime_learned_ensemble",
                "fallback_policy": "signed_regime_learned_ensemble",
                "delta_vs_fallback": 0.0,
            },
        ]
        self.assertLess(
            risk_adjusted_validation_score(
                validation_score=100.0,
                validation_rows=rows,
                promotion_loss_weight=1.0,
                worst_loss_weight=0.5,
            ),
            100.0,
        )

    def test_lower_confidence_selector_robust_search_runs(self):
        tasks = [
            RiskTask(name="lcb-robust-a", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150),
            RiskTask(name="lcb-robust-b", rounds=3, initial_bankroll=140, bet=20, target_bankroll=180),
        ]
        scores = {
            task.name: {
                "signed_regime_learned_ensemble": 10.0,
                "expected_value": 11.0,
                "fixed_oce_3": 9.0,
            }
            for task in tasks
        }
        result = search_lcb_selector(tasks=tasks, scores_by_task=scores, smoke=True, robust_selection=True)
        self.assertTrue(result.train_profiles)
        self.assertTrue(result.candidate_scores)

    def test_meta_selector_search_runs(self):
        train_task = RiskTask(name="meta-train", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150)
        validation_task = RiskTask(name="meta-validation", rounds=2, initial_bankroll=140, bet=20, target_bankroll=170)
        result = search_meta_selector(
            train_tasks=[train_task],
            validation_tasks=[validation_task],
            seeds=[0],
            episodes=1,
            hand_depth=1,
            smoke=True,
        )
        self.assertTrue(result.train_profiles)
        self.assertTrue(result.validation_summaries)

    def test_meta_selector_cv_search_runs(self):
        tasks = [
            RiskTask(name="meta-cv-a", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150),
            RiskTask(name="meta-cv-b", rounds=3, initial_bankroll=140, bet=20, target_bankroll=180),
        ]
        result = search_meta_selector_cv(
            tasks=tasks,
            seeds=[0],
            episodes=1,
            hand_depth=1,
            smoke=True,
        )
        self.assertTrue(result.train_profiles)
        self.assertTrue(result.validation_summaries)

    def test_robust_gate_candidate_policy_scores(self):
        params = robust_gate_candidate_params(smoke=True)[0]
        policy = robust_gate_policy(params)
        task = RiskTask(name="robust-gate-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, target_bankroll=150)
        probs = policy.action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})

    def test_robust_gate_search_runs(self):
        task = RiskTask(name="robust-search-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        result = search_robust_gate(
            train_tasks=[task],
            seeds=[0],
            episodes=1,
            hand_depth=1,
            smoke=True,
        )
        self.assertTrue(result.train_summaries)
        self.assertIsInstance(result.selection_score, float)

    def test_robust_gate_search_accepts_validation_tasks(self):
        train_task = RiskTask(name="robust-search-train", rounds=2, initial_bankroll=120, target_bankroll=150)
        validation_task = RiskTask(name="robust-search-validation", rounds=2, initial_bankroll=140, target_bankroll=170)
        result = search_robust_gate(
            train_tasks=[train_task],
            validation_tasks=[validation_task],
            seeds=[0],
            episodes=1,
            hand_depth=1,
            smoke=True,
        )
        self.assertTrue(result.validation_summaries)
        self.assertIsInstance(result.validation_score, float)

    def test_portfolio_task_features_are_stable(self):
        task = RiskTask(name="portfolio-features", rounds=3, initial_bankroll=120, bet=20, target_bankroll=160)
        features = task_features(task)
        self.assertEqual(len(features), 9)
        self.assertTrue(all(isinstance(value, float) for value in features))

    def test_task_feature_portfolio_policy_scores(self):
        task = RiskTask(name="portfolio-policy", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150)
        params = PortfolioSelectorParams(k=1)
        profile = PortfolioProfile(
            task=task.name,
            features=task_features(task, params),
            policy="expected_value",
            score=1.0,
        )
        policy = TaskFeaturePortfolioPolicy([profile], params)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, bet=20, target_bankroll=150)
        probs = policy.action_probabilities(state, task, rounds_remaining=2, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})

    def test_portfolio_selector_search_runs(self):
        train_task = RiskTask(name="portfolio-train", rounds=2, initial_bankroll=120, bet=20, target_bankroll=150)
        validation_task = RiskTask(name="portfolio-validation", rounds=2, initial_bankroll=140, bet=20, target_bankroll=170)
        result = search_portfolio_selector(
            train_tasks=[train_task],
            validation_tasks=[validation_task],
            seeds=[0],
            episodes=1,
            hand_depth=1,
            smoke=True,
        )
        self.assertTrue(result.train_profiles)
        self.assertTrue(result.validation_summaries)

    def test_target_branch_searched_policy_scores(self):
        task = RiskTask(name="target-branch-test", rounds=3, initial_bankroll=120, target_bankroll=160)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, target_bankroll=160)
        probs = target_branch_searched_policy().action_probabilities(state, task, rounds_remaining=3, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})

    def test_target_family_split_has_held_out_tasks(self):
        train, test = target_family_split()
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertFalse({task.name for task in train} & {task.name for task in test})

    def test_target_branch_candidate_policy_scores(self):
        params = target_candidate_params(smoke=True)[0]
        policy = target_branch_candidate_policy(params)
        task = RiskTask(name="target-candidate", rounds=3, initial_bankroll=120, target_bankroll=160)
        state = DecisionState((10, 6), 10, current_bankroll=120, initial_bankroll=120, target_bankroll=160)
        probs = policy.action_probabilities(state, task, rounds_remaining=3, hand_depth=1)
        self.assertEqual(set(probs), {"hit", "stand"})

    def test_target_branch_search_runs(self):
        train = [RiskTask(name="target-train", rounds=2, initial_bankroll=120, target_bankroll=150)]
        test = [RiskTask(name="target-test", rounds=2, initial_bankroll=120, target_bankroll=150)]
        result = search_target_branch_policy(
            train_tasks=train,
            test_tasks=test,
            benchmark_tasks=benchmark_tasks(),
            episodes=1,
            seed=11,
            hand_depth=1,
            smoke=True,
            max_candidates=1,
        )
        self.assertTrue(result.test_summaries)
        self.assertIsInstance(result.promotion_gate.accepted, bool)
        self.assertIsInstance(result.selection_score, float)
        self.assertIsInstance(result.benchmark_target_selection_score, float)

    def test_target_promotion_gate_reports_required_checks(self):
        params = target_candidate_params(smoke=True)[0]
        policy = target_branch_candidate_policy(params, name="candidate")
        train, test = target_family_split()
        report = evaluate_promotion_gate(
            candidate_target_policy=policy,
            test_tasks=test[:1],
            benchmark_tasks=benchmark_tasks(),
            episodes=1,
            seed=13,
            hand_depth=1,
        )
        self.assertIn(report.accepted, (True, False))
        self.assertIsInstance(report.failed_checks, tuple)

    def test_promotion_gate_allows_benchmark_ties(self):
        report = PromotionGateResult(
            accepted=True,
            min_delta=0.0,
            target_family_candidate_score=2.0,
            target_family_incumbent_score=1.0,
            target_family_delta=1.0,
            benchmark_target_candidate_score=1.0,
            benchmark_target_incumbent_score=1.0,
            benchmark_target_delta=0.0,
            signed_ensemble_candidate_score=1.0,
            signed_ensemble_incumbent_score=1.0,
            signed_ensemble_delta=0.0,
            failed_checks=(),
        )
        self.assertTrue(report.accepted)

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

    def test_learned_mixture_search_runs(self):
        train_task = RiskTask(name="mixture-train", rounds=2, initial_bankroll=120, target_bankroll=150)
        test_task = RiskTask(name="mixture-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        result = search_learned_mixture_policy(
            train_tasks=[train_task],
            test_tasks=[test_task],
            episodes=2,
            seed=6,
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

    def test_multiseed_evaluation_runs(self):
        task = RiskTask(name="multiseed-test", rounds=2, initial_bankroll=120, target_bankroll=150)
        rows, aggregate, paired_deltas = run_multiseed_evaluation(
            tasks=[task],
            seeds=[0, 1],
            episodes=1,
            hand_depth=1,
        )
        self.assertTrue(rows)
        self.assertTrue(aggregate)
        self.assertTrue(paired_deltas)
        self.assertTrue(all(row["reference_policy"] == "signed_regime_learned_ensemble" for row in paired_deltas))

    def test_paired_policy_deltas_compare_same_task_seed_cells(self):
        rows = [
            {"task": "a", "seed": 0, "policy": "ref", "score": 3.0},
            {"task": "a", "seed": 0, "policy": "base", "score": 1.0},
            {"task": "a", "seed": 1, "policy": "ref", "score": 2.0},
            {"task": "a", "seed": 1, "policy": "base", "score": 4.0},
        ]
        deltas = paired_policy_deltas(rows, reference_policy="ref")
        self.assertEqual(deltas[0]["n_pairs"], 2)
        self.assertEqual(deltas[0]["mean_delta"], 0.0)

    def test_task_level_paired_score_report_averages_seeds_before_inference(self):
        rows = [
            {"task": "a", "seed": 0, "policy": "ref", "score": 4.0},
            {"task": "a", "seed": 0, "policy": "base", "score": 1.0},
            {"task": "a", "seed": 1, "policy": "ref", "score": 2.0},
            {"task": "a", "seed": 1, "policy": "base", "score": 1.0},
            {"task": "b", "seed": 0, "policy": "ref", "score": 0.0},
            {"task": "b", "seed": 0, "policy": "base", "score": 2.0},
            {"task": "b", "seed": 1, "policy": "ref", "score": 3.0},
            {"task": "b", "seed": 1, "policy": "base", "score": 1.0},
        ]
        self.assertEqual(paired_score_deltas(rows, "ref", "base", unit="task"), [2.0, 0.0])
        report = paired_score_report(
            rows,
            "ref",
            "base",
            unit="task",
            bootstrap_samples=100,
            randomization_samples=100,
        )
        self.assertEqual(report["unit"], "task")
        self.assertEqual(report["n_units"], 2)
        self.assertEqual(report["mean_delta"], 1.0)

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
