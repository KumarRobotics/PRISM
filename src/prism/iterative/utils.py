import json
from typing import Optional


def spine_data_to_prompt(
    input_data: str, include_graph_semantics: Optional[bool] = False
) -> str:
    """Get prompt for data generator from a data example.

    Assumes the SPINE data format.
    """
    task, graph = input_data.split("Scene graph:")
    task = task.split("task: ")[1]
    graph = json.loads(graph)
    graph_semantics = [n["name"] for n in graph["objects"] + graph["regions"]]

    prompt = "example task: "
    prompt += task

    if include_graph_semantics:
        prompt += graph_semantics

    return prompt
