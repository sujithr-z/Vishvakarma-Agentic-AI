"""Unit tests for strict 2-type decision parser."""
import pytest
from app.agent.parser import parse_decision, AgentDecision


def test_parse_tool_decision():
    raw = '{"type": "tool", "tool_name": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Need climate data."}'
    decision = parse_decision(raw)
    assert decision.type == "tool"
    assert decision.tool_name == "get_climate"
    assert decision.arguments == {"location": "Kerala"}
    assert decision.reason == "Need climate data."


def test_parse_final_decision():
    raw = '{"type": "final", "content": "The calculation result is 2."}'
    decision = parse_decision(raw)
    assert decision.type == "final"
    assert decision.content == "The calculation result is 2."
    assert decision.tool_name is None


def test_parse_markdown_wrapped_json():
    raw = """Here is my decision:
```json
{
  "type": "tool",
  "tool_name": "search_knowledge",
  "arguments": {
    "query": "hot humid ventilation"
  },
  "reason": "Search passive strategies."
}
```
Hope this helps!"""
    decision = parse_decision(raw)
    assert decision.type == "tool"
    assert decision.tool_name == "search_knowledge"
    assert decision.arguments["query"] == "hot humid ventilation"


def test_parse_invalid_text_raises_error():
    raw = "The thermal performance score of the bamboo shelter is 0.78, which is above the 0.70 threshold."
    # Non-JSON output must raise ValueError without silent repair
    with pytest.raises(ValueError) as excinfo:
        parse_decision(raw)
    assert "JSON" in str(excinfo.value) or "valid" in str(excinfo.value)
