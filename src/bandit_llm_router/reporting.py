"""Markdown tables for the demo and the bench scripts.

The README's Results section is pasted from bench output, so the table formatting lives in one place
and both the demo and the benches print the same shapes. Numbers are formatted here, never rounded
differently in two places.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["format_number", "markdown_table"]


def format_number(value: object) -> str:
    """Render a cell: floats with four significant decimals, everything else as ``str``."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # noqa: PLR0124 - NaN check
            return str(value)
        return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.1f}"
    return str(value)


def markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    """A Markdown table from uniform rows; the first row's keys are the columns."""
    if not rows:
        return "(no rows)"
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(format_number(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])
