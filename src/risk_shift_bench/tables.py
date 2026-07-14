"""Paper-style tables from benchmark summaries."""

from __future__ import annotations

from collections import defaultdict


def best_policy_by_task(summaries: list[dict], metric: str = "mean_final_bankroll") -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        by_task[row["task"]].append(row)
    output = []
    for task, rows in sorted(by_task.items()):
        best = max(rows, key=lambda row: row[metric])
        output.append({"task": task, "metric": metric, "policy": best["policy"], "value": best[metric]})
    return output


def normalized_policy_ranks(summaries: list[dict], metrics: tuple[str, ...] | None = None) -> list[dict]:
    selected = metrics or (
        "mean_final_bankroll",
        "cvar_5_final_bankroll",
        "target_probability",
        "ruin_probability",
        "mean_max_drawdown",
    )
    lower_is_better = {"ruin_probability", "mean_max_drawdown"}
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        by_task[row["task"]].append(row)

    ranks: dict[tuple[str, str], list[float]] = defaultdict(list)
    for task, rows in by_task.items():
        for metric in selected:
            ordered = sorted(rows, key=lambda row: row[metric], reverse=metric not in lower_is_better)
            for rank, row in enumerate(ordered, start=1):
                ranks[(task, row["policy"])].append(float(rank))

    return [
        {
            "task": task,
            "policy": policy,
            "mean_rank": sum(values) / len(values),
        }
        for (task, policy), values in sorted(ranks.items())
    ]


def to_latex_table(rows: list[dict], columns: tuple[str, ...] | None = None, caption: str = "", label: str = "") -> str:
    if not rows:
        return ""
    selected = columns or tuple(rows[0].keys())
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{" + "l" * len(selected) + "}",
        "\\toprule",
        " & ".join(selected).replace("_", "\\_") + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = []
        for column in selected:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"

