"""Pooled-development PPO references for v2 inventory and MiniGrid domains."""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter

from experiments.frontier_v2_baseline_audit import (
    audit_baseline_manifest,
    file_sha256,
    select_checkpoint,
)
from experiments.frontier_v2_baseline_design import (
    BASELINE_SOURCE_LOCKS,
    COMPETITIVE_BASELINES,
    baseline_design_summary,
)
from experiments.frontier_v2_baseline_runner_hash import (
    runner_implementation_files,
    runner_implementation_sha256,
)
from experiments.frontier_v2_baseline_source_audit import (
    BASELINE_SOURCE_DIRECTORIES,
    audit_baseline_source,
)
from experiments.frontier_v2_double_dqn import (
    ACTION_COUNT,
    _activate_domain_source,
    _encoded_observation,
    _episode_metrics,
    _start_episode,
    _step_episode,
    _torch_modules,
    pooled_task_index,
)
from experiments.frontier_v2_external_adapters import bounded_score
from experiments.frontier_v2_external_design import (
    CODEBASE_LOCKS,
    all_tasks,
    canonical_episode_seed_base,
    domain_tasks,
    expected_episode_seeds,
    task_manifest_sha256,
)
from experiments.frontier_v2_source_audit import SOURCE_DIRECTORIES


SUPPORTED_BASELINES = {
    baseline.domain: baseline
    for baseline in COMPETITIVE_BASELINES
    if baseline.algorithm in {"clipped PPO", "recurrent PPO"}
}
ROLLOUT_STEPS = 250
NUM_ENVS = 10
ROLLOUT_TIME_STEPS = ROLLOUT_STEPS // NUM_ENVS
RECURRENCE_STEPS = 5
INVENTORY_INPUT_SIZE = 39
INVENTORY_ACTION_BINS = 11


@dataclass
class InventoryEpisode:
    domain: str
    task: object
    environment: object
    steps: int = 0
    raw_return: float = 0.0


def _start_inventory_episode(task, seed: int):
    np, _torch = _torch_modules()
    from or_gym.envs.supply_chain.inventory_management import (
        InvManagementBacklogEnv,
        InvManagementLostSalesEnv,
    )

    parameters = task.parameter_dict()
    environment_class = (
        InvManagementBacklogEnv
        if bool(parameters["backlog"])
        else InvManagementLostSalesEnv
    )
    lead_time_scale = float(parameters["lead_time_scale"])
    lead_times = [max(1, round(base * lead_time_scale)) for base in (3, 5, 10)]
    np.random.seed(seed % (2**32))
    environment = environment_class(
        periods=int(parameters["periods"]),
        dist=1,
        dist_param={"mu": float(parameters["demand_mean"])},
        seed_int=seed,
        L=lead_times,
    )
    observation = environment.reset()
    return (
        InventoryEpisode(
            domain="or_gym_inventory_management",
            task=task,
            environment=environment,
        ),
        observation,
    )


def _inventory_observation(observation):
    np, _torch = _torch_modules()
    values = np.asarray(observation, dtype=np.float32)
    if values.ndim != 1 or len(values) > INVENTORY_INPUT_SIZE:
        raise RuntimeError("inventory observation exceeds the frozen encoder")
    encoded = np.zeros(INVENTORY_INPUT_SIZE, dtype=np.float32)
    encoded[: len(values)] = np.tanh(values / 100.0)
    return encoded


def inventory_bin_to_action(action_bins, supply_capacity) -> tuple[int, ...]:
    if len(action_bins) != 3 or len(supply_capacity) != 3:
        raise ValueError("inventory action and capacity vectors must have three stages")
    return tuple(
        int(round(int(bin_index) * float(capacity) / (INVENTORY_ACTION_BINS - 1)))
        for bin_index, capacity in zip(action_bins, supply_capacity, strict=True)
    )


def _step_inventory(context: InventoryEpisode, action_bins):
    np, _torch = _torch_modules()
    action = inventory_bin_to_action(action_bins, context.environment.supply_capacity)
    observation, reward, done, _info = context.environment.step(
        np.asarray(action, dtype=np.int32)
    )
    context.steps += 1
    context.raw_return += float(reward)
    return observation, float(reward), bool(done)


def _inventory_metrics(context: InventoryEpisode) -> tuple[float, float]:
    parameters = context.task.parameter_dict()
    lower = float(parameters["profit_lower"])
    upper = float(parameters["profit_upper"])
    unmet = float(context.environment.B.sum() + context.environment.LS.sum())
    return bounded_score((context.raw_return - lower) / (upper - lower)), unmet


def _start(domain: str, task, seed: int):
    if domain == "or_gym_inventory_management":
        context, observation = _start_inventory_episode(task, seed)
        return context, _inventory_observation(observation)
    context, observation = _start_episode(domain, task, seed)
    return context, _encoded_observation(context, observation)


def _step(domain: str, context, action):
    if domain == "or_gym_inventory_management":
        observation, reward, done = _step_inventory(context, action)
        return _inventory_observation(observation), max(-10.0, min(10.0, reward / 4_000.0)), done
    observation, reward, done = _step_episode(context, int(action))
    return _encoded_observation(context, observation), reward, done


def _metrics(domain: str, context) -> tuple[float, float]:
    if domain == "or_gym_inventory_management":
        return _inventory_metrics(context)
    return _episode_metrics(context)


def _build_inventory_agent(torch):
    class InventoryAgent(torch.nn.Module):
        recurrent = False

        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(INVENTORY_INPUT_SIZE, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 128),
                torch.nn.Tanh(),
            )
            self.actor = torch.nn.Linear(128, 3 * INVENTORY_ACTION_BINS)
            self.critic = torch.nn.Linear(128, 1)

        def distribution_and_value(self, observation):
            hidden = self.trunk(observation)
            logits = self.actor(hidden).reshape(-1, 3, INVENTORY_ACTION_BINS)
            distributions = [
                torch.distributions.Categorical(logits=logits[:, index, :])
                for index in range(3)
            ]
            return distributions, self.critic(hidden).squeeze(-1)

        def action_value(self, observation, action=None):
            distributions, value = self.distribution_and_value(observation)
            if action is None:
                action = torch.stack(
                    [distribution.sample() for distribution in distributions], dim=1
                )
            log_probability = torch.stack(
                [
                    distribution.log_prob(action[:, index])
                    for index, distribution in enumerate(distributions)
                ],
                dim=1,
            ).sum(dim=1)
            entropy = torch.stack(
                [distribution.entropy() for distribution in distributions], dim=1
            ).sum(dim=1)
            return action, log_probability, entropy, value

    return InventoryAgent()


def _build_recurrent_agent(torch, input_size: int, action_count: int):
    class RecurrentAgent(torch.nn.Module):
        recurrent = True
        hidden_size = 128

        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_size, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, self.hidden_size),
                torch.nn.ReLU(),
            )
            self.memory = torch.nn.LSTMCell(self.hidden_size, self.hidden_size)
            self.actor = torch.nn.Linear(self.hidden_size, action_count)
            self.critic = torch.nn.Linear(self.hidden_size, 1)

        def step(self, observation, hidden, cell, episode_start):
            keep = (1.0 - episode_start.float()).unsqueeze(1)
            hidden = hidden * keep
            cell = cell * keep
            hidden, cell = self.memory(self.encoder(observation), (hidden, cell))
            distribution = torch.distributions.Categorical(logits=self.actor(hidden))
            value = self.critic(hidden).squeeze(-1)
            return distribution, value, hidden, cell

    return RecurrentAgent()


def _build_agent(domain: str, torch):
    if domain == "or_gym_inventory_management":
        return _build_inventory_agent(torch), INVENTORY_INPUT_SIZE
    size = {"minigrid_dynamic_obstacles": 16, "minigrid_lava_crossing": 11}[domain]
    input_size = size * size * 3
    return _build_recurrent_agent(torch, input_size, ACTION_COUNT[domain]), input_size


def _zero_memory(agent, torch, device):
    if not agent.recurrent:
        return None, None
    shape = (1, agent.hidden_size)
    return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


def _resolve_device(torch, device_name: str):
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("PPO device must be auto, cpu, or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    selected = (
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    return torch.device(selected)


def _evaluate_agent(domain: str, agent, *, episodes_per_task: int, device) -> dict:
    _np, torch = _torch_modules()
    task_scores = []
    task_costs = []
    agent.eval()
    for task in domain_tasks(domain, "calibration"):
        scores = []
        costs = []
        for seed in expected_episode_seeds(
            task,
            episodes=episodes_per_task,
            seed_base=canonical_episode_seed_base(task),
        ):
            context, observation = _start(domain, task, seed)
            hidden, cell = _zero_memory(agent, torch, device)
            episode_start = torch.ones(1, device=device)
            try:
                done = False
                while not done:
                    observation_tensor = torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        if agent.recurrent:
                            distribution, _value, hidden, cell = agent.step(
                                observation_tensor,
                                hidden,
                                cell,
                                episode_start,
                            )
                            action = int(distribution.probs.argmax(dim=1).item())
                        else:
                            distributions, _value = agent.distribution_and_value(
                                observation_tensor
                            )
                            action = [
                                int(distribution.probs.argmax(dim=1).item())
                                for distribution in distributions
                            ]
                    observation, _reward, done = _step(
                        domain, context, action
                    )
                    episode_start.zero_()
                score, cost = _metrics(domain, context)
                scores.append(score)
                costs.append(cost)
            finally:
                context.environment.close()
        task_scores.append(fmean(scores))
        task_costs.append(fmean(costs))
    agent.train()
    return {
        "calibration_equal_task_mean_score": fmean(task_scores),
        "calibration_equal_task_mean_cost": fmean(task_costs),
        "calibration_task_mean_scores": task_scores,
        "calibration_task_mean_costs": task_costs,
        "calibration_episodes_per_task": episodes_per_task,
    }


def _ppo_loss(torch, new_log_probability, entropy, new_value, batch) -> tuple:
    log_ratio = new_log_probability - batch["old_log_probability"]
    ratio = log_ratio.exp()
    advantages = batch["advantage"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = torch.maximum(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 0.8, 1.2),
    ).mean()
    unclipped_value_loss = (new_value - batch["return"]) ** 2
    clipped_value = batch["old_value"] + torch.clamp(
        new_value - batch["old_value"], -0.2, 0.2
    )
    clipped_value_loss = (clipped_value - batch["return"]) ** 2
    value_loss = 0.5 * torch.maximum(
        unclipped_value_loss, clipped_value_loss
    ).mean()
    entropy_loss = entropy.mean()
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss, policy_loss, value_loss


def _update_feedforward(agent, optimizer, rollout, *, rng, device) -> None:
    np, torch = _torch_modules()
    size = len(rollout["observations"])
    for _epoch in range(4):
        for indices in np.array_split(rng.permutation(size), 5):
            observation = torch.as_tensor(
                rollout["observations"][indices], dtype=torch.float32, device=device
            )
            action = torch.as_tensor(
                rollout["actions"][indices], dtype=torch.int64, device=device
            )
            _action, log_probability, entropy, value = agent.action_value(
                observation, action
            )
            batch = {
                key: torch.as_tensor(rollout[key][indices], device=device)
                for key in (
                    "old_log_probability",
                    "advantage",
                    "return",
                    "old_value",
                )
            }
            loss, _policy_loss, _value_loss = _ppo_loss(
                torch, log_probability, entropy, value, batch
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()


def _update_recurrent(agent, optimizer, rollout, *, rng, device) -> None:
    np, torch = _torch_modules()
    size = len(rollout["observations"])
    if size % RECURRENCE_STEPS:
        raise RuntimeError("recurrent rollout is not divisible by recurrence length")
    chunk_starts = np.arange(0, size, RECURRENCE_STEPS)
    for _epoch in range(4):
        for starts in np.array_split(rng.permutation(chunk_starts), 5):
            log_probabilities = []
            entropies = []
            values = []
            flat_indices = []
            starts = np.asarray(starts, dtype=np.int64)
            hidden = torch.as_tensor(
                rollout["hidden"][starts], dtype=torch.float32, device=device
            )
            cell = torch.as_tensor(
                rollout["cell"][starts], dtype=torch.float32, device=device
            )
            for offset in range(RECURRENCE_STEPS):
                indices = starts + offset
                observation = torch.as_tensor(
                    rollout["observations"][indices],
                    dtype=torch.float32,
                    device=device,
                )
                episode_start = torch.as_tensor(
                    rollout["episode_starts"][indices],
                    dtype=torch.float32,
                    device=device,
                )
                distribution, value, hidden, cell = agent.step(
                    observation, hidden, cell, episode_start
                )
                action = torch.as_tensor(
                    rollout["actions"][indices],
                    dtype=torch.int64,
                    device=device,
                )
                log_probabilities.append(distribution.log_prob(action))
                entropies.append(distribution.entropy())
                values.append(value)
                flat_indices.extend(indices.tolist())
            log_probability = torch.cat(log_probabilities)
            entropy = torch.cat(entropies)
            value = torch.cat(values)
            indices = np.asarray(flat_indices, dtype=np.int64)
            batch = {
                key: torch.as_tensor(rollout[key][indices], device=device)
                for key in (
                    "old_log_probability",
                    "advantage",
                    "return",
                    "old_value",
                )
            }
            loss, _policy_loss, _value_loss = _ppo_loss(
                torch, log_probability, entropy, value, batch
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()


def _vectorized_advantages(rewards, dones, values, *, next_values):
    np, _torch = _torch_modules()
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantages = np.zeros(rewards.shape[1], dtype=np.float32)
    for time_index in reversed(range(rewards.shape[0])):
        if time_index == rewards.shape[0] - 1:
            following_values = next_values
        else:
            following_values = values[time_index + 1]
        following_nonterminal = 1.0 - dones[time_index]
        delta = (
            rewards[time_index]
            + 0.99 * following_values * following_nonterminal
            - values[time_index]
        )
        last_advantages = (
            delta + 0.99 * 0.95 * following_nonterminal * last_advantages
        )
        advantages[time_index] = last_advantages
    return advantages, advantages + values


def _portable_state_dict(agent) -> dict:
    return {name: value.detach().cpu() for name, value in agent.state_dict().items()}


def train_ppo_seed(
    domain: str,
    *,
    training_seed: int,
    output_root: Path,
    total_steps: int,
    checkpoint_interval_steps: int,
    evaluation_episodes_per_task: int,
    device_name: str = "auto",
) -> dict:
    np, torch = _torch_modules()
    if total_steps <= 0 or total_steps % ROLLOUT_STEPS:
        raise ValueError("PPO training steps must be a positive multiple of 250")
    if checkpoint_interval_steps <= 0 or checkpoint_interval_steps % ROLLOUT_STEPS:
        raise ValueError("PPO checkpoint intervals must be a positive multiple of 250")

    torch.manual_seed(training_seed)
    np.random.seed(training_seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = _resolve_device(torch, device_name)
    agent, input_size = _build_agent(domain, torch)
    agent = agent.to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)
    rng = np.random.default_rng(training_seed)
    tasks = domain_tasks(domain, "development")
    episode_serials = np.arange(NUM_ENVS, dtype=np.int64)
    contexts = []
    current_observations = []
    for serial in episode_serials:
        task = tasks[pooled_task_index(training_seed, int(serial), len(tasks))]
        context, observation = _start(
            domain, task, training_seed + int(serial)
        )
        contexts.append(context)
        current_observations.append(observation)
    current_observations = np.asarray(current_observations, dtype=np.float32)
    if agent.recurrent:
        hidden = torch.zeros((NUM_ENVS, agent.hidden_size), device=device)
        cell = torch.zeros((NUM_ENVS, agent.hidden_size), device=device)
    else:
        hidden = cell = None
    episode_starts = np.ones(NUM_ENVS, dtype=np.float32)
    checkpoints = []
    total_episodes = 0
    started = perf_counter()
    try:
        for rollout_start in range(0, total_steps, ROLLOUT_STEPS):
            learning_rate = 2.5e-4 * (1.0 - rollout_start / total_steps)
            optimizer.param_groups[0]["lr"] = learning_rate
            observations = np.zeros(
                (ROLLOUT_TIME_STEPS, NUM_ENVS, input_size), dtype=np.float32
            )
            action_shape = (
                (ROLLOUT_TIME_STEPS, NUM_ENVS, 3)
                if not agent.recurrent
                else (ROLLOUT_TIME_STEPS, NUM_ENVS)
            )
            actions = np.zeros(action_shape, dtype=np.int64)
            log_probabilities = np.zeros(
                (ROLLOUT_TIME_STEPS, NUM_ENVS), dtype=np.float32
            )
            rewards = np.zeros((ROLLOUT_TIME_STEPS, NUM_ENVS), dtype=np.float32)
            dones = np.zeros((ROLLOUT_TIME_STEPS, NUM_ENVS), dtype=np.float32)
            values = np.zeros((ROLLOUT_TIME_STEPS, NUM_ENVS), dtype=np.float32)
            rollout_episode_starts = np.zeros(
                (ROLLOUT_TIME_STEPS, NUM_ENVS), dtype=np.float32
            )
            hidden_states = (
                np.zeros(
                    (ROLLOUT_TIME_STEPS, NUM_ENVS, agent.hidden_size),
                    dtype=np.float32,
                )
                if agent.recurrent
                else None
            )
            cell_states = (
                np.zeros(
                    (ROLLOUT_TIME_STEPS, NUM_ENVS, agent.hidden_size),
                    dtype=np.float32,
                )
                if agent.recurrent
                else None
            )

            for time_index in range(ROLLOUT_TIME_STEPS):
                observations[time_index] = current_observations
                rollout_episode_starts[time_index] = episode_starts
                observation_tensor = torch.as_tensor(
                    current_observations, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    if agent.recurrent:
                        hidden_states[time_index] = hidden.detach().cpu().numpy()
                        cell_states[time_index] = cell.detach().cpu().numpy()
                        distribution, value, hidden, cell = agent.step(
                            observation_tensor,
                            hidden,
                            cell,
                            torch.as_tensor(
                                episode_starts, dtype=torch.float32, device=device
                            ),
                        )
                        action_tensor = distribution.sample()
                        log_probability = distribution.log_prob(action_tensor)
                        environment_actions = action_tensor.cpu().numpy()
                        actions[time_index] = environment_actions
                    else:
                        (
                            action_tensor,
                            log_probability,
                            _entropy,
                            value,
                        ) = agent.action_value(observation_tensor)
                        environment_actions = action_tensor.cpu().numpy()
                        actions[time_index] = environment_actions
                log_probabilities[time_index] = log_probability.cpu().numpy()
                values[time_index] = value.cpu().numpy()
                next_observations = []
                next_episode_starts = np.zeros(NUM_ENVS, dtype=np.float32)
                for environment_index, context in enumerate(contexts):
                    environment_action = environment_actions[environment_index]
                    if not agent.recurrent:
                        environment_action = environment_action.tolist()
                    observation, reward, done = _step(
                        domain, context, environment_action
                    )
                    rewards[time_index, environment_index] = reward
                    dones[time_index, environment_index] = float(done)
                    next_episode_starts[environment_index] = float(done)
                    if done:
                        context.environment.close()
                        total_episodes += 1
                        episode_serials[environment_index] += NUM_ENVS
                        serial = int(episode_serials[environment_index])
                        task = tasks[
                            pooled_task_index(training_seed, serial, len(tasks))
                        ]
                        context, observation = _start(
                            domain, task, training_seed + serial
                        )
                        contexts[environment_index] = context
                    next_observations.append(observation)
                current_observations = np.asarray(
                    next_observations, dtype=np.float32
                )
                episode_starts = next_episode_starts

            with torch.no_grad():
                observation_tensor = torch.as_tensor(
                    current_observations, dtype=torch.float32, device=device
                )
                if agent.recurrent:
                    _distribution, next_value_tensor, _next_hidden, _next_cell = agent.step(
                        observation_tensor,
                        hidden,
                        cell,
                        torch.as_tensor(
                            episode_starts, dtype=torch.float32, device=device
                        ),
                    )
                else:
                    _distributions, next_value_tensor = agent.distribution_and_value(
                        observation_tensor
                    )
            advantages, returns = _vectorized_advantages(
                rewards,
                dones,
                values,
                next_values=next_value_tensor.cpu().numpy(),
            )
            flatten = lambda value: value.transpose(1, 0, *range(2, value.ndim)).reshape(  # noqa: E731
                (-1, *value.shape[2:]) if value.ndim > 2 else -1
            )
            rollout = {
                "observations": flatten(observations),
                "actions": flatten(actions),
                "old_log_probability": flatten(log_probabilities),
                "advantage": flatten(advantages),
                "return": flatten(returns),
                "old_value": flatten(values),
                "episode_starts": flatten(rollout_episode_starts),
            }
            if agent.recurrent:
                rollout["hidden"] = flatten(hidden_states)
                rollout["cell"] = flatten(cell_states)
                _update_recurrent(agent, optimizer, rollout, rng=rng, device=device)
            else:
                _update_feedforward(agent, optimizer, rollout, rng=rng, device=device)

            completed_steps = rollout_start + ROLLOUT_STEPS
            if completed_steps % checkpoint_interval_steps == 0:
                evaluation = _evaluate_agent(
                    domain,
                    agent,
                    episodes_per_task=evaluation_episodes_per_task,
                    device=device,
                )
                relative = Path(domain) / SUPPORTED_BASELINES[domain].name / (
                    f"seed-{training_seed}"
                ) / f"step-{completed_steps}.pt"
                checkpoint_path = output_root / relative
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "protocol_id": "riskshiftbench-frontier-v2-ppo-checkpoint-v1",
                        "domain": domain,
                        "training_seed": training_seed,
                        "step": completed_steps,
                        "input_size": input_size,
                        "recurrent": bool(agent.recurrent),
                        "agent_state_dict": _portable_state_dict(agent),
                    },
                    checkpoint_path,
                )
                checkpoints.append(
                    {
                        "step": completed_steps,
                        "checkpoint_path": relative.as_posix(),
                        "checkpoint_sha256": file_sha256(checkpoint_path),
                        **evaluation,
                    }
                )
    finally:
        for context in contexts:
            context.environment.close()

    selected = select_checkpoint(checkpoints, cost_limit=None)
    return {
        "training_seed": training_seed,
        "training_steps": total_steps,
        "training_episodes": total_episodes,
        "runtime_seconds": perf_counter() - started,
        "checkpoints": checkpoints,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
    }


def audit_selected_checkpoint_replay(
    payload: dict,
    *,
    checkpoint_root: Path,
    device_name: str = "auto",
) -> dict:
    _np, torch = _torch_modules()
    domain = str(payload["baseline_identifier"]).split(":", 1)[0]
    device = _resolve_device(torch, device_name)
    records = []
    for run in payload["runs"]:
        selected_sha256 = run["selected_checkpoint_sha256"]
        matches = [
            checkpoint
            for checkpoint in run["checkpoints"]
            if checkpoint["checkpoint_sha256"] == selected_sha256
        ]
        if len(matches) != 1:
            raise RuntimeError("selected PPO checkpoint is not unique")
        expected = matches[0]
        checkpoint = torch.load(
            checkpoint_root / expected["checkpoint_path"],
            map_location=device,
            weights_only=True,
        )
        agent, input_size = _build_agent(domain, torch)
        required = {
            "protocol_id": "riskshiftbench-frontier-v2-ppo-checkpoint-v1",
            "domain": domain,
            "training_seed": int(run["training_seed"]),
            "step": int(expected["step"]),
            "input_size": input_size,
            "recurrent": bool(agent.recurrent),
        }
        if any(checkpoint.get(key) != value for key, value in required.items()):
            raise RuntimeError("selected PPO checkpoint metadata changed")
        agent = agent.to(device)
        agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
        replay = _evaluate_agent(
            domain,
            agent,
            episodes_per_task=int(expected["calibration_episodes_per_task"]),
            device=device,
        )
        for field in (
            "calibration_equal_task_mean_score",
            "calibration_equal_task_mean_cost",
        ):
            if not math.isclose(
                float(replay[field]),
                float(expected[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"selected PPO checkpoint replay changed {field}")
        records.append(
            {
                "training_seed": int(run["training_seed"]),
                "selected_step": int(expected["step"]),
                "selected_checkpoint_sha256": selected_sha256,
                "calibration_equal_task_mean_score": replay[
                    "calibration_equal_task_mean_score"
                ],
                "calibration_equal_task_mean_cost": replay[
                    "calibration_equal_task_mean_cost"
                ],
            }
        )
    return {
        "design": "riskshiftbench-frontier-v2-ppo-selected-replay-audit-v1",
        "checkpoint_count": len(records),
        "calibration_replay_exact": True,
        "records": records,
    }


def train_ppo_baseline(
    domain: str,
    *,
    output_root: Path,
    source_root: Path,
    baseline_source_root: Path,
    device_name: str = "auto",
) -> dict:
    try:
        baseline = SUPPORTED_BASELINES[domain]
    except KeyError as error:
        raise KeyError(f"no PPO baseline for {domain}") from error
    codebase, environment_audit = _activate_domain_source(domain, source_root)
    baseline_source = (
        baseline_source_root / BASELINE_SOURCE_DIRECTORIES[baseline.implementation_source]
    )
    implementation_audit = audit_baseline_source(
        baseline_source, baseline.implementation_source
    )
    implementation_audit_payload = asdict(implementation_audit)
    implementation_audit_payload["source"] = BASELINE_SOURCE_DIRECTORIES[
        baseline.implementation_source
    ]
    environment_audit_payload = asdict(environment_audit)
    environment_audit_payload["source"] = SOURCE_DIRECTORIES[codebase]
    design = baseline_design_summary()
    runs = [
        train_ppo_seed(
            domain,
            training_seed=seed,
            output_root=output_root,
            total_steps=baseline.training_steps_per_seed,
            checkpoint_interval_steps=baseline.checkpoint_interval_steps,
            evaluation_episodes_per_task=100,
            device_name=device_name,
        )
        for seed in baseline.training_seeds
    ]
    _np, torch = _torch_modules()
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-baseline-checkpoints-v1",
        "baseline_design_sha256": design["design_sha256"],
        "baseline_implementation_sha256": design["internal_implementation_sha256"],
        "runner_implementation_files": list(
            runner_implementation_files(baseline.algorithm)
        ),
        "runner_implementation_sha256": runner_implementation_sha256(
            baseline.algorithm
        ),
        "baseline_identifier": baseline.identifier,
        "baseline_spec": asdict(baseline),
        "development_manifest_sha256": task_manifest_sha256(
            all_tasks("development")
        ),
        "calibration_manifest_sha256": task_manifest_sha256(
            all_tasks("calibration")
        ),
        "source_lock": asdict(BASELINE_SOURCE_LOCKS[baseline.implementation_source]),
        "baseline_source_audit": implementation_audit_payload,
        "environment_codebase_lock": asdict(CODEBASE_LOCKS[codebase]),
        "environment_source_audit": environment_audit_payload,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(_resolve_device(torch, device_name)),
            "device_name": torch.cuda.get_device_name(0)
            if _resolve_device(torch, device_name).type == "cuda"
            else "cpu",
        },
        "algorithm_hyperparameters": {
            "architecture": "MLP(128,128)" if domain == "or_gym_inventory_management" else "MLP(256,128)+LSTM(128)",
            "optimizer": "Adam",
            "learning_rate": 2.5e-4,
            "learning_rate_schedule": "linear to zero",
            "rollout_steps": ROLLOUT_STEPS,
            "update_epochs": 4,
            "minibatches": 5,
            "recurrence_steps": RECURRENCE_STEPS if domain.startswith("minigrid_") else 1,
            "discount": 0.99,
            "gae_lambda": 0.95,
            "clip_coefficient": 0.2,
            "entropy_coefficient": 0.01,
            "value_coefficient": 0.5,
            "maximum_gradient_norm": 0.5,
            "inventory_action_bins_per_stage": INVENTORY_ACTION_BINS if domain == "or_gym_inventory_management" else None,
            "task_sampling": "episode-stratified pooled development round robin",
        },
        "runs": runs,
    }
    payload["selected_checkpoint_replay_audit"] = audit_selected_checkpoint_replay(
        payload, checkpoint_root=output_root, device_name=device_name
    )
    manifest_path = output_root / domain / baseline.name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    audit_baseline_manifest(payload, checkpoint_root=output_root)
    return payload


def smoke_ppo(
    domain: str,
    *,
    steps: int,
    output_root: Path,
    source_root: Path,
    device_name: str = "auto",
) -> dict:
    if domain not in SUPPORTED_BASELINES:
        raise KeyError(domain)
    _activate_domain_source(domain, source_root)
    seed = SUPPORTED_BASELINES[domain].training_seeds[0]
    run = train_ppo_seed(
        domain,
        training_seed=seed,
        output_root=output_root,
        total_steps=steps,
        checkpoint_interval_steps=steps,
        evaluation_episodes_per_task=2,
        device_name=device_name,
    )
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-ppo-smoke-v1",
        "nonconfirmatory": True,
        "domain": domain,
        "baseline_identifier": SUPPORTED_BASELINES[domain].identifier,
        "steps": steps,
        "run": run,
        "runs": [run],
    }
    payload["selected_checkpoint_replay_audit"] = audit_selected_checkpoint_replay(
        payload, checkpoint_root=output_root, device_name=device_name
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=tuple(SUPPORTED_BASELINES), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baselines"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument(
        "--baseline-source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baseline_sources"),
    )
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_steps is not None:
        payload = smoke_ppo(
            args.domain,
            steps=args.smoke_steps,
            output_root=args.output_root,
            source_root=args.source_root,
            device_name=args.device,
        )
    else:
        payload = train_ppo_baseline(
            args.domain,
            output_root=args.output_root,
            source_root=args.source_root,
            baseline_source_root=args.baseline_source_root,
            device_name=args.device,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
