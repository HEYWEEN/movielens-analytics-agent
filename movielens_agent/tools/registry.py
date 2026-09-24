"""Small explicit registry; future iteration tools register here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    output_type: str
    required_inputs: tuple[str, ...]


TOOLS = {
    "governance.clean_and_assess": ToolSpec(
        "governance.clean_and_assess", "cleaned_dataset", ("raw_dataset",)),
}


def get_tool(tool_id):
    return TOOLS[tool_id]
