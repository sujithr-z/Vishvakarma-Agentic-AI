"""Unit tests for decision parser and json sanitization."""
import pytest
from app.agent.parser import parse_decision, AgentDecision


def test_parse_clean_json():
    raw = '{"action": "get_climate", "arguments": {"location": "Kerala"}, "reason": "Need climate data."}'
    decision = parse_decision(raw)
    assert decision.action == "get_climate"
    assert decision.arguments == {"location": "Kerala"}
    assert decision.reason == "Need climate data."


def test_parse_markdown_wrapped_json():
    raw = """Here is my decision:
```json
{
  "action": "search_knowledge",
  "arguments": {
    "query": "hot humid ventilation"
  },
  "reason": "Search passive strategies."
}
```
Hope this helps!"""
    decision = parse_decision(raw)
    assert decision.action == "search_knowledge"
    assert decision.arguments["query"] == "hot humid ventilation"


def test_parse_json_with_trailing_commas():
    raw = '{"action": "generate_design", "arguments": {"capacity": 5, "budget": 80000,}, "reason": "Baseline design",}'
    decision = parse_decision(raw)
    assert decision.action == "generate_design"
    assert decision.arguments["capacity"] == 5


def test_parse_conversational_text_fallback():
    raw = "The thermal performance score of the bamboo shelter is 0.78, which is above the 0.70 threshold."
    decision = parse_decision(raw)
    assert decision.action == "answer"
    assert "0.78" in decision.arguments.get("message", "")
