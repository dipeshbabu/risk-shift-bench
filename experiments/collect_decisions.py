"""Minimal CLI for collecting human hit/stand decisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from risk_shift_bench.active_query import candidate_states
from risk_shift_bench.dataset import DecisionRecord, save_jsonl


def parse_action(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"h", "hit"}:
        return "hit"
    if value in {"s", "stand"}:
        return "stand"
    raise ValueError("Enter h/hit or s/stand.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--decisions", type=int, default=30)
    parser.add_argument("--out", default="artifacts/human/decisions.jsonl")
    args = parser.parse_args()

    states = candidate_states()
    records: list[DecisionRecord] = []
    print("Enter h/hit or s/stand for each state. Press Ctrl+C to stop early.")
    try:
        for step, state in enumerate(states[: args.decisions]):
            print()
            print(f"Decision {step + 1}/{args.decisions}")
            print(f"Player cards: {list(state.player_cards)}  total={state.player_total}")
            print(f"Dealer card:  {state.dealer_card}")
            print(f"Bankroll:     {state.current_bankroll:.0f}  bet={state.bet:.0f}")
            while True:
                try:
                    action = parse_action(input("Action [h/s]: "))
                    break
                except ValueError as exc:
                    print(exc)
            records.append(
                DecisionRecord(
                    subject_id=args.subject_id,
                    episode_id=f"human_{args.subject_id}",
                    step_id=step,
                    player_cards=state.player_cards,
                    dealer_card=state.dealer_card,
                    current_bankroll=state.current_bankroll,
                    initial_bankroll=state.initial_bankroll,
                    bet=state.bet,
                    recent_outcomes=state.recent_outcomes,
                    action_taken=action,
                    target_bankroll=state.target_bankroll,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
    finally:
        if records:
            save_jsonl(records, args.out)
            print(f"Saved {len(records)} decisions to {args.out}")


if __name__ == "__main__":
    main()

