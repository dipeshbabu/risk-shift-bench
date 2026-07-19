import unittest
import json
import tempfile
from pathlib import Path

from experiments.external_budget_baselines import (
    fixed_budget_comparison,
    proposal_focused_allocation,
    random_task_allocation,
    total_policy_episodes,
    uniform_task_allocation,
)
from experiments.external_confirmation_evaluation import (
    sha256_bytes,
    sha256_file,
    validate_protocol,
)
from experiments.external_familywise_verifier import (
    FamilywisePilotPlan,
    holm_rejections,
    verify_familywise_promotion,
)
from experiments.external_protocol_lock import finalize_registration
from experiments.external_study_design import (
    DOMAINS,
    SPLITS,
    all_tasks,
    canonical_sha256,
    design_summary,
    domain_tasks,
    RUNTIME_DEPENDENCIES,
    task_manifest_sha256,
)
from experiments.external_domain_adapters import (
    ExternalEpisodeOutcome,
    knapsack_action,
    safety_point_action,
    summarize_outcomes,
)


class ExternalFamilywiseVerifierTests(unittest.TestCase):
    def test_required_batches_control_complete_family(self):
        plan = FamilywisePilotPlan(proposal_family_size=59)
        self.assertEqual(plan.required_unanimous_batches, 11)
        self.assertLessEqual(2**-plan.required_unanimous_batches, plan.local_alpha)
        self.assertGreater(2 ** -(plan.required_unanimous_batches - 1), plan.local_alpha)

    def test_familywise_gate_requires_locked_batch_count(self):
        plan = FamilywisePilotPlan(proposal_family_size=20)
        with self.assertRaisesRegex(ValueError, "exactly"):
            verify_familywise_promotion([1.0] * (plan.required_unanimous_batches - 1), plan)

    def test_unanimous_familywise_promotion_passes(self):
        plan = FamilywisePilotPlan(proposal_family_size=20)
        result = verify_familywise_promotion(
            [0.5] * plan.required_unanimous_batches,
            plan,
        )
        self.assertTrue(result.accepted)
        self.assertLessEqual(result.sign_test_p, plan.local_alpha)

    def test_holm_stops_after_first_failure(self):
        rejected = holm_rejections({"a": 0.01, "b": 0.03, "c": 0.031}, familywise_alpha=0.05)
        self.assertEqual(rejected, {"a"})


class ExternalBudgetBaselineTests(unittest.TestCase):
    def test_uniform_allocation_preserves_integer_batch_budget(self):
        allocation = uniform_task_allocation(["c", "a", "b"], 8, 50)
        self.assertEqual([item.task for item in allocation], ["a", "b", "c"])
        self.assertEqual([item.batches for item in allocation], [3, 3, 2])
        self.assertEqual(total_policy_episodes(allocation), 800)

    def test_focused_allocation_checks_family_size(self):
        with self.assertRaisesRegex(ValueError, "family size"):
            proposal_focused_allocation(
                ["a"],
                FamilywisePilotPlan(proposal_family_size=2),
            )

    def test_cost_matched_comparison_is_exact(self):
        plan = FamilywisePilotPlan(proposal_family_size=2, episodes_per_batch=25)
        report = fixed_budget_comparison(
            all_tasks=["a", "b", "c", "d", "e"],
            proposal_tasks=["b", "d"],
            plan=plan,
        )
        focused = sum(
            2 * row["batches"] * row["episodes_per_batch"]
            for row in report["proposal_focused"]
        )
        uniform = sum(
            2 * row["batches"] * row["episodes_per_batch"]
            for row in report["uniform_all_tasks"]
        )
        self.assertEqual(focused, uniform)
        random_cost = sum(
            2 * row["batches"] * row["episodes_per_batch"]
            for row in report["outcome_blind_random_tasks"]["allocation"]
        )
        self.assertEqual(focused, random_cost)
        self.assertEqual(focused, report["total_candidate_plus_fallback_episodes"])

    def test_random_allocation_is_outcome_blind_and_deterministic(self):
        plan = FamilywisePilotPlan(proposal_family_size=2)
        first = random_task_allocation(["a", "b", "c", "d"], 2, plan, seed=17)
        second = random_task_allocation(["d", "c", "b", "a"], 2, plan, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(item.batches == plan.required_unanimous_batches for item in first))


class ExternalExecutionGuardTests(unittest.TestCase):
    def test_locked_file_hash_is_line_ending_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.txt"
            crlf = root / "crlf.txt"
            lf.write_bytes(b"first\nsecond\n")
            crlf.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(sha256_file(lf), sha256_file(crlf))

    def test_unregistered_draft_refuses_confirmation_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_path = root / "design.json"
            protocol_path = root / "protocol.json"
            design = {"outcomes": "absent"}
            design_path.write_text(json.dumps(design), encoding="utf-8")
            protocol_path.write_text(
                json.dumps(
                    {
                        "status": "awaiting_external_registration",
                        "locked_design_path": str(design_path),
                        "locked_design_sha256": sha256_bytes(design_path),
                        "locked_design_canonical_sha256": canonical_sha256(design),
                        "registration": {
                            "provider": None,
                            "url": None,
                            "registered_at": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "externally registered"):
                validate_protocol(protocol_path, require_registration=True)

    def test_registration_finalization_records_exact_design_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_path = root / "design.json"
            draft_path = root / "draft.json"
            registered_path = root / "registered.json"
            design = {"purpose": "test", "values": [1, 2, 3]}
            design_path.write_bytes(
                (json.dumps(design, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            draft_path.write_text(
                json.dumps(
                    {
                        "status": "awaiting_external_registration",
                        "locked_design_path": str(design_path),
                        "locked_design_sha256": sha256_bytes(design_path),
                        "locked_design_canonical_sha256": canonical_sha256(design),
                        "registration": {},
                    }
                ),
                encoding="utf-8",
            )
            finalize_registration(
                draft_path=draft_path,
                output_path=registered_path,
                provider="test-registry",
                url="https://registry.example/immutable/1",
                registered_at="2026-07-15T12:00:00-04:00",
            )
            registered = json.loads(registered_path.read_text(encoding="utf-8"))
            self.assertEqual(
                registered["registration"]["registered_design_sha256"],
                sha256_bytes(design_path),
            )


class ExternalTaskDesignTests(unittest.TestCase):
    def test_splits_are_name_disjoint(self):
        names = {
            split: {task.name for task in all_tasks(split)}
            for split in SPLITS
        }
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                self.assertFalse(names[left] & names[right])

    def test_confirmation_is_complete_and_deterministic(self):
        expected_counts = {
            "gymnasium_frozenlake": 12,
            "or_gym_online_knapsack": 12,
            "safety_gymnasium_point_goal": 9,
        }
        for domain in DOMAINS:
            first = domain_tasks(domain, "confirmation")
            second = domain_tasks(domain, "confirmation")
            self.assertEqual(len(first), expected_counts[domain])
            self.assertEqual(first, second)
            self.assertEqual(task_manifest_sha256(first), task_manifest_sha256(second))

    def test_design_summary_contains_no_outcomes(self):
        summary = design_summary()
        self.assertIn("have not been reset", summary["scope"])
        self.assertEqual(set(summary["environment_locks"]), set(DOMAINS))
        self.assertNotIn("result", summary)

    def test_runtime_dependency_locks_cover_compatibility_stacks(self):
        self.assertEqual(
            dict(RUNTIME_DEPENDENCIES["or_gym_online_knapsack"]),
            {"gym": "0.26.2", "numpy": "1.26.4"},
        )
        safety_dependencies = dict(
            RUNTIME_DEPENDENCIES["safety_gymnasium_point_goal"]
        )
        self.assertEqual(safety_dependencies["mujoco"], "2.3.3")
        self.assertEqual(safety_dependencies["numpy"], "1.23.5")

    def test_frozenlake_horizon_is_explicit_for_both_map_sizes(self):
        tasks = domain_tasks("gymnasium_frozenlake", "confirmation")
        horizons = {
            int(task.parameter_dict()["map_size"]): int(task.parameter_dict()["max_steps"])
            for task in tasks
        }
        self.assertEqual(horizons, {4: 100, 8: 200})


class ExternalAdapterUnitTests(unittest.TestCase):
    def test_summary_uses_lower_tail_utility(self):
        rows = [
            ExternalEpisodeOutcome(
                domain="d",
                task="t",
                policy="p",
                episode=index,
                seed=index,
                utility=value,
                raw_return=value,
                cost=0.0,
                failure=False,
                steps=1,
                successes=1,
                resource_residual=0.0,
            )
            for index, value in enumerate((0.0, 10.0, 20.0, 30.0))
        ]
        summary = summarize_outcomes(rows)
        self.assertEqual(summary["mean_utility"], 15.0)
        self.assertEqual(summary["cvar_5_utility"], 0.0)
        self.assertEqual(summary["score"], 15.0)

    def test_knapsack_policy_never_accepts_an_item_that_does_not_fit(self):
        action = knapsack_action(
            "ratio_threshold_1_25",
            [95, 0, 10, 100],
            capacity=100,
            horizon=50,
            step=0,
        )
        self.assertEqual(action, 0)

    def test_safety_policy_turns_and_slows_for_target_obstacle(self):
        observation = {
            "goal_lidar": [1.0] + [0.0] * 15,
            "hazards_lidar": [1.0] + [0.0] * 15,
            "vases_lidar": [0.0] * 16,
        }
        action = safety_point_action("hazard_aware_moderate", observation)
        self.assertLessEqual(action[0], 0.25)
        self.assertGreaterEqual(action[1], -1.0)
        self.assertLessEqual(action[1], 1.0)


if __name__ == "__main__":
    unittest.main()
