import json
from typing import List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion
from spine.mapping.graph_util import GraphHandler

from prism.data.utils import GPTQueryClient

QUERY = """
You are generating data for training an llm-based planner, like the SPINE paper from ravichandran et al.

Generate a scene graph for training in the following format

{
        "objects": [{"name": "object_1_name", "coords": [west_east_coordinate, south_north_coordinate]}, ...],
        "regions": [{"name": "region_1_name", "coords": [west_east_coordinate, south_north_coordinate]}, ...],
        "object_connections: [["object_name", "region_name"], ...],
        "region_connections": [["some_region_name", "other_region_name"], ...]
        "robot_location": "region_of_robot_location
}

For example,

{
"objects":
[
    {"name": "shed_1", "coords": [78, 9]},
    {"name": "gate_1", "coords": [52, -56]}
],
"regions": [
    {"name": "ground_1", "coords": [0, 0]},
    {"name": "road_1", "coords": [5.7, -8.3]},
    {"name": "road_2", "coords": [19.3, -6.5]},
    {"name": "road_3", "coords": [35.7, -12.1]},
    {"name": "road_4", "coords": [52.7, -20]},
    {"name": "road_5", "coords": [57.2, -31.6]},
    {"name": "bridge_1", "coords": [54.3, -46.7]},
    {"name": "road_6", "coords": [52.4, -56.5]},
    {"name": "driveway_1", "coords": [78.4, 9.1]}
],
"object_connections": [
    ["shed_1", "driveway_1"],
    ["gate_1", "road_6"]
],
"region_connections":[
    ["ground_1", "road_1"],
    ["road_1", "road_2"],
    ["road_2", "road_3"],
    ["road_3", "road_4"],
    ["road_4", "road_5"],
    ["road_5", "bridge_1"],
    ["bridge_1", "road_6"],
    ["road_6", "driveway_1"]
],
"robot_location": "ground_1"
}

Make sure all nodes referenced in the conntections are listed in the objects and regions list.
Provide your answer in the following JSON format:

{
reasoning: describe the type of scene you are creating,
graph: <JSON GRAPH>,
tasks: list of tasks that correspond to the graph.
}


Add a "description" attribute to each node that provides information.
These will be hidden from the robot

Task generation instructions
- DO NOT reference specific objects or nodes. Make the planner infer theese.
- Tasks should request specific information, not general exploration. Make the planner map or inspect certain entities. For example, start tasks with phrases such as "what", "I heard", "find out", "map", "inspect", "Can I", "is there", and likewise

"""

SCENE_PRIOR = """We are improving the SPINE planner proposed by ravichandran et al.
You need to generate data for training. Describe scenes you would train in, such as regions, objects, and general scene description

Describe ONE example environment, including scene, regions, and objects.
Such as `semi-urban office park with fields, roads, parking lots, buildings, people...` and more.

You will be randomly sampled, so be creative but realistic.

Your response should be a JSON with a "description" key, the value be the description.
"""


class TaskGraphGen:
    def __init__(self):
        self.client = GPTQueryClient()  # OpenAI()

    def _build_prompt(self, n_regions=10, n_objects=10, n_tasks=2, prior=""):
        query = (
            QUERY
            + f"\nYour graph should have {n_regions} regions, {n_objects} objects, and you should generate {n_tasks} tasks"
        )

        if prior != "":
            query += f"\nYour tasks and scene should be like the following: {prior}"

        return query

    def get_tasks(
        self, n_regions=10, n_objects=10, n_tasks=2, description=""
    ) -> List[str]:
        """Get GPT generated tasks for putting planner data

        Parameters
        ----------
        n_regions : int, optional
            Number of regions in the graph, by default 10
        n_objects : int, optional
            Number of objects in the graph, by default 10
        n_tasks : int, optional
            Number of tasks to generate, by default 2
        description : str, optional
            an example/prior scene description to give the LLM to base the tasks on.

        Returns
        -------
        List[str]
            list of tasks
        """

        if description == "":
            description = self.client.query_gpt(
                query=SCENE_PRIOR, temperature=0.95, max_tokens=256
            )
            description = json.loads(description.choices[0].message.content)[
                "description"
            ]

        response = self.client.query_gpt(
            query=self._build_prompt(
                n_regions=n_regions,
                n_objects=n_objects,
                n_tasks=n_tasks,
                prior=description,
            )
        )

        # try to load the graph for error handling
        json_content = json.loads(response.choices[0].message.content)
        json_content["description"] = description
        graph_handle = GraphHandler(graph_path="")
        graph_handle.reset(
            json_content["graph"],
            current_location=json_content["graph"]["robot_location"],
        )

        # make sure GPT isn't hallucinating edges
        for [source, end] in graph_handle.graph.edges:
            assert source in graph_handle.graph.nodes, f"{source} not in graph"
            assert end in graph_handle.graph.nodes, f"{end} not in graph"

        return json_content


if __name__ == "__main__":
    gen = TaskGraphGen()

    rnd_data = gen.get_tasks()

    graph_handler = GraphHandler(graph_path="")

    whatdoihave = 0
