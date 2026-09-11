"""Strict 2-Type Decision Parser (TOOL vs FINAL) for Qwen 1.8B."""
import re
import json
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator, ValidationError

AVAILABLE_REGISTERED_TOOLS = [
    "get_climate",
    "search_knowledge",
    "generate_design",
    "evaluate_thermal",
    "evaluate_cost",
    "evaluate_structure",
    "modify_design",
    "analyze_thermal_constraints",
    "calculate_improvement_cost",
    "save_experience",
    "retrieve_experience"
]


class AgentDecision(BaseModel):
    """
    Strict 2-Type Decision Model:
    1. type="tool"  -> requires tool_name (from AVAILABLE_REGISTERED_TOOLS), arguments dict, reason
    2. type="final" -> requires content string, reason (optional)
    """
    type: Literal["tool", "final"] = Field(..., description="Decision type: 'tool' or 'final'")
    tool_name: Optional[str] = Field(default=None, description="Exact tool name if type='tool'")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments if type='tool'")
    reason: str = Field(default="", description="One-sentence rationale")
    content: Optional[str] = Field(default=None, description="Final response content if type='final'")

    @model_validator(mode="after")
    def validate_type_fields(self):
        if self.type == "tool":
            if not self.tool_name or not isinstance(self.tool_name, str) or not self.tool_name.strip():
                raise ValueError("Decision with type='tool' MUST provide a valid 'tool_name'.")
            self.tool_name = self.tool_name.strip()
            # Clean any quotes or formatting in tool_name
            self.tool_name = re.sub(r'[\s<>"\'`]', '', self.tool_name)
            self.content = None
        elif self.type == "final":
            if not self.content or not isinstance(self.content, str) or not self.content.strip():
                raise ValueError("Decision with type='final' MUST provide non-empty 'content'.")
            self.content = self.content.strip()
            self.tool_name = None
        return self

    @property
    def action(self) -> str:
        """Backward compatibility helper mapping to tool_name or finish."""
        if self.type == "final":
            return "finish"
        return self.tool_name or ""


def extract_json_block(text: str) -> Optional[str]:
    """Extract first valid JSON block from text."""
    if not text:
        return None

    text = text.strip()

    # Case 1: Markdown code fences
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    # Case 2: Balanced braces
    start_idx = text.find("{")
    if start_idx != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if c == '"' and not escape:
                in_string = not in_string
            elif c == '\\' and in_string:
                escape = not escape
                continue
            elif not in_string:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start_idx:i+1]
            escape = False

    return None


def parse_decision(raw_output: str) -> AgentDecision:
    """
    Strictly parse raw model output into an AgentDecision.
    Raises ValueError on invalid output without silent fallbacks.
    """
    if not raw_output or not raw_output.strip():
        raise ValueError("Empty output received from model.")

    json_str = extract_json_block(raw_output)
    if not json_str:
        raise ValueError(f"Output does not contain a valid JSON object. Received: '{raw_output[:100]}...'")

    try:
        data = json.loads(json_str)
    except Exception as err:
        raise ValueError(f"JSON decode failed: {err}. Raw text: '{json_str[:100]}...'") from err

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object (dict), got {type(data).__name__}.")

    # Normalize keys for model friendliness
    # 1. Type normalization
    dec_type = data.get("type", "").strip().lower()
    
    # Handle legacy 'action' key if type wasn't explicitly provided
    if not dec_type:
        legacy_action = str(data.get("action", "")).strip().lower()
        if legacy_action in ["final", "finish", "answer", "done", "complete"]:
            dec_type = "final"
        elif legacy_action in AVAILABLE_REGISTERED_TOOLS or "tool_name" in data:
            dec_type = "tool"
            if "tool_name" not in data and legacy_action:
                data["tool_name"] = legacy_action

    if dec_type in ["tool", "call_tool", "execute_tool"]:
        data["type"] = "tool"
        if "tool_name" not in data:
            if "action" in data:
                data["tool_name"] = str(data["action"]).strip()
            elif "tool" in data:
                data["tool_name"] = str(data["tool"]).strip()
    elif dec_type in ["final", "answer", "finish", "response", "message"]:
        data["type"] = "final"
        if "content" not in data:
            for alt in ["message", "summary", "text", "response", "answer"]:
                if alt in data and data[alt]:
                    data["content"] = str(data[alt]).strip()
                    break
            if "arguments" in data and isinstance(data["arguments"], dict):
                msg_in_args = data["arguments"].get("message") or data["arguments"].get("summary") or data["arguments"].get("content")
                if msg_in_args:
                    data["content"] = str(msg_in_args).strip()

    # Ensure arguments is a dict
    if "arguments" not in data or not isinstance(data["arguments"], dict):
        if "args" in data and isinstance(data["args"], dict):
            data["arguments"] = data["args"]
        elif "parameters" in data and isinstance(data["parameters"], dict):
            data["arguments"] = data["parameters"]
        else:
            data["arguments"] = {}

    try:
        return AgentDecision(**data)
    except ValidationError as val_err:
        raise ValueError(f"Decision schema validation failed: {val_err}") from val_err
