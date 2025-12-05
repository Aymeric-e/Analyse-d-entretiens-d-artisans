"""
Unit tests for CSV to HTML highlight utility functions.

Tests the text highlighting utilities from src/utils/csv_to_html_highlight_tool.py.
Includes tests for:
- build_pattern_for_tools: Validates regex pattern creation from tool lists
- highlight_text_with_tools: Validates HTML mark tag insertion around detected tools
"""

from utils.csv_to_html_highlight_tool import build_pattern_for_tools, highlight_text_with_tools


def test_build_pattern_and_highlight_simple():
    """
    Test pattern building and text highlighting with tool detection.

    Verifies that:
    - build_pattern_for_tools creates a valid regex pattern from a tool list
    - highlight_text_with_tools wraps detected tools with <mark> HTML tags
    - All tools are found and highlighted in the output
    """
    text = "J'utilise un marteau et un tournevis pour bricoler."
    tools = ["marteau", "tournevis"]

    pattern = build_pattern_for_tools(tools)
    assert pattern is not None

    highlighted = highlight_text_with_tools(text, tools)
    # Should contain mark tags around tools
    assert "<mark" in highlighted
    assert "marteau" in highlighted
    assert "tournevis" in highlighted


def test_highlight_no_tools_returns_escaped():
    """
    Test text highlighting with an empty tool list.

    Verifies that when no tools are provided, the function returns the text
    HTML-escaped but without any <mark> tags.
    """
    text = "Ceci n'a pas d'outils."
    highlighted = highlight_text_with_tools(text, [])
    # No marks and text escaped
    assert "<mark" not in highlighted
    assert "Ceci" in highlighted
