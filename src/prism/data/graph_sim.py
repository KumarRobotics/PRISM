from copy import deepcopy
from typing import Optional

import numpy as np
from spine.mapping.graph_util import GraphHandler
from spine.spine_util import UpdatePromptFormer


class GraphSim:
    def __init__(self, graph: GraphHandler):
        self.graph = graph

        for node in self.graph.graph.nodes:
            if "description" not in self.graph.graph.nodes[node]:
                self.graph.graph.nodes[node]["description"] = "no description"

        self.partial_graph = deepcopy(graph)
        self.init_partial_graph()

    def init_partial_graph(self):
        for node in self.partial_graph.graph.nodes:
            self.partial_graph.graph.nodes[node].pop("description", 0)

        self.have_updates = False
        self.removed_nodes = []
        self.action_history = []
        self.updator = UpdatePromptFormer()
        self.have_updates = False
        self.partial_graph.as_json_str = self.partial_graph.to_json_str()

    def reset(self, **args):
        self.graph.reset(**args)
        self.partial_graph = deepcopy(self.graph)
        self.init_partial_graph()

    def randomly_remove_nodes(
        self, *, pct: float = 0, n_nodes: float = 0, to_remove=[]
    ):
        all_nodes = list(self.partial_graph.graph.nodes)

        # first check if we should randomly remove
        if pct > 0:
            assert n_nodes == 0 and len(to_remove) == 0
            n_to_remove = (len(all_nodes) * pct) // 100
            n_nodes = n_to_remove
        elif n_nodes > 0:
            assert len(to_remove) == 0

        # then check to remove specific nodes
        if len(to_remove) > 0:
            assert n_nodes == 0 and pct == 0
        else:
            to_remove = list(np.random.choice(all_nodes, n_nodes, replace=False))

        # make sure we don't remove current location
        if self.partial_graph.current_location in to_remove:
            to_remove.remove(self.partial_graph.current_location)

        for node in to_remove:
            self.partial_graph.remove_node(node)

        self.removed_nodes.extend(list(to_remove))

        self.partial_graph.as_json_str = self.partial_graph.to_json_str()

    def get_updator(self) -> UpdatePromptFormer:
        return self.updator

    def add_new_node(self, source_node, target, debug: Optional[bool] = True):
        if debug:
            print(f"discovered missing node: {target}")
        node_info = self.graph.graph.nodes[target]

        node_info["name"] = target
        self.updator.update(new_nodes=[{target: node_info}])
        edge_info = self.graph.graph.get_edge_data(target, source_node)

        # only add type and coorindates
        self.partial_graph.graph.add_node(
            target, type=node_info["type"], coords=node_info["coords"]
        )
        self.partial_graph.graph.add_edge(target, source_node, **edge_info)

        self.removed_nodes.remove(target)
        self.have_updates = True

    def add_edges(self, source: str, target: str):
        self.updator.update(new_connections=[[source, target]])
        edge_info = self.graph.graph.get_edge_data(source, target)
        self.partial_graph.graph.add_edge(source, target, **edge_info)
        self.have_updates = True

    def take_action(self, action, argument) -> bool:
        if action == "map_region" or action == "explore_region":
            if action == "explore_region":
                current_location = argument[0]
            else:
                current_location = argument

            neighbors = self.graph.get_neighbors(current_location)

            for n in neighbors:
                if n not in self.partial_graph.graph.nodes:
                    self.add_new_node(current_location, n)

                gt_edges = [
                    sorted(e)
                    for e in list(self.partial_graph.get_edges(current_location).keys())
                ]

                query_edge = sorted((current_location, n))
                if query_edge not in gt_edges:
                    self.add_edges(query_edge[0], query_edge[1])

            assert (
                "description" in self.graph.graph.nodes[current_location]
            ), f"{current_location} has no description"
            description = self.graph.graph.nodes[current_location]["description"]
            self.partial_graph.graph.nodes[current_location][
                "description"
            ] = description
            self.updator.update(
                attribute_updates=[
                    {"name": current_location, "description": description}
                ]
            )

        # TODO incomplete
        elif action == "extend_map":
            self.updator.update(
                freeform_updates=["Do not call extend_map. Try explore_region instead"]
            )

            # current_location = self.partial_graph.current_location
            # line =  np.array(argument) - self.partial_graph.graph.nodes[current_location]["coords"]
            # debug = 0

            # # see what nodes are close
            # # TODO try and implement exploration
            # for n in self.removed_nodes:
            #     distance_thresh = 10
            #     print(np.linalg.norm(self.graph.get_node_coord(current_location) - self.graph.get_node_coord(n)))
            #     if np.linalg.norm(self.graph.get_node_coord(current_location) - self.graph.get_node_coord(n)) < 25:
            #         self.add_new_node(source_node=current_location, target=n)

        elif action == "inspect":
            target = argument[0]
            description = self.graph.graph.nodes[target]["description"]
            self.partial_graph.graph.nodes[target]["description"] = description
            self.updator.update(
                attribute_updates=[{"name": target, "description": description}]
            )
            self.have_updates = True

        elif action == "goto":
            location = argument
            self.updator.update(location_updates=[location])
            self.partial_graph.update_location(location)

        if len(self.action_history) > 0 and action in self.action_history[-3:]:
            self.updator.update(
                freeform_updates=[
                    f"You are calling {action} multiple times in a row, which will not help you solve the task. Try calling something else. If you have tried all options, answer the user with your results."
                ]
            )
        self.action_history.append(action)

        return self.have_updates

    def __str__(self):
        out = ""

        out += f"full graph\n---\nn_nodes: {len(self.graph.graph.nodes)}\nn_edges: {len(self.graph.graph.edges)}"
        out += f"\n===\npartial graph\n---\nn_nodes: {len(self.partial_graph.graph.nodes)}\nn_edges: {len(self.partial_graph.graph.edges)}"

        return out
