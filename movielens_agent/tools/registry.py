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
    "classification.logistic_regression": ToolSpec(
        "classification.logistic_regression", "high_rating_prediction", ("user_id", "movie_id", "analysis_version")),
    "classification.decision_tree": ToolSpec(
        "classification.decision_tree", "high_rating_prediction", ("user_id", "movie_id", "analysis_version")),
    "clustering.kmeans": ToolSpec(
        "clustering.kmeans", "user_segment", ("user_id", "analysis_version")),
    "clustering.agglomerative": ToolSpec(
        "clustering.agglomerative", "user_segment", ("user_id", "analysis_version")),
    "reduction.pca": ToolSpec(
        "reduction.pca", "user_coordinates", ("analysis_version",)),
    "reduction.nmf": ToolSpec(
        "reduction.nmf", "user_coordinates", ("analysis_version",)),
}


def get_tool(tool_id):
    return TOOLS[tool_id]
