from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from src.ui.pages import dataframe_for_streamlit


def test_plotly_charts_have_explicit_streamlit_keys() -> None:
    source_path = Path("src/ui/pages.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    missing_keys: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "plotly_chart":
            if not any(keyword.arg == "key" for keyword in node.keywords):
                missing_keys.append(node.lineno)

    assert missing_keys == []


def test_dataframe_for_streamlit_stringifies_nested_object_cells() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "warnings": [["a", "b"], "plain"],
            "metadata": [{"source": "test"}, None],
        }
    )

    display = dataframe_for_streamlit(frame)

    assert display.loc[0, "warnings"] == '["a", "b"]'
    assert display.loc[1, "warnings"] == "plain"
    assert display.loc[0, "metadata"] == '{"source": "test"}'
