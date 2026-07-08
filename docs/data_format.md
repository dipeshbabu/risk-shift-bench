# Data Format

Benchmark episode records are JSONL rows with:

```text
task
policy
seed
final_bankroll
min_bankroll
max_drawdown
ruined
target_hit
rounds_played
```

Decision-inference records are JSONL rows with:

```text
subject_id
episode_id
step_id
player_cards
dealer_card
current_bankroll
initial_bankroll
bet
recent_outcomes
action_taken
target_bankroll
timestamp
```

