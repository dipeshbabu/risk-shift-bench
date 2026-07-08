"""Dependency-free SVG figures for benchmark summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def bar_chart_svg(
    rows: list[dict],
    metric: str,
    out_path: str | Path,
    title: str,
    width: int = 1200,
    row_height: int = 26,
) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    bars = []
    max_value = max(abs(float(row[metric])) for row in rows) if rows else 1.0
    y = 40
    for task, task_rows in sorted(grouped.items()):
        bars.append(f'<text x="20" y="{y}" font-size="14" font-weight="bold">{task}</text>')
        y += row_height
        for row in sorted(task_rows, key=lambda item: item["policy"]):
            value = float(row[metric])
            bar_width = 700 * (abs(value) / max(max_value, 1e-9))
            bars.append(f'<text x="40" y="{y}" font-size="12">{row["policy"]}</text>')
            bars.append(f'<rect x="310" y="{y - 13}" width="{bar_width:.1f}" height="16" fill="#3b82f6" />')
            bars.append(f'<text x="{320 + bar_width:.1f}" y="{y}" font-size="12">{value:.2f}</text>')
            y += row_height
        y += 10
    height = max(y + 20, 120)
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white" />',
            f'<text x="20" y="24" font-size="18" font-weight="bold">{title}</text>',
            *bars,
            "</svg>",
        ]
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")

