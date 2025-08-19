import json
import os
from pathlib import Path
from typing import List, Union

import numpy as np
from spine.mapping.graph_util import GraphHandler
from spine.spine import SPINE

from prism.data.graph_gen import TaskGraphGen
from prism.data.graph_sim import GraphSim
from prism.data.planning_sim import PlanningSim
from prism.data.utils import aggregate


class DataGenerator:
    def __init__(
        self,
        graph_unknown: Union[int, List[int]],
        n_region_list: Union[int, List[int]],
        n_objects_list: Union[int, List[int]],
    ):
        self.unknown_pcts = graph_unknown
        self.n_regions_list = n_region_list
        self.n_objects_list = n_objects_list
        self.context_gen = TaskGraphGen()
        self.planning_sim = PlanningSim()

    def generate(
        self, log_dir: str, n_samples: int, n_tasks: int, description: str = ""
    ) -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        data_counter = 0

        for idx in range(n_samples):
            n_regions = self.n_regions_list[idx % len(self.n_objects_list)]
            n_objects = self.n_objects_list[idx % len(self.n_objects_list)]

            for i in range(10):
                try:
                    # error handling in case data generation fails
                    print(
                        f"generation attempt {i+1}/10 with {n_regions} regions, and {n_objects} objects"
                    )
                    rnd_data = self.context_gen.get_tasks(
                        n_regions=n_regions,
                        n_objects=n_objects,
                        n_tasks=n_tasks,
                        description=description,
                    )

                    break

                except Exception as ex:
                    print(f"graph generator invalid: {ex}")
            tasks = rnd_data["tasks"]
            print(f"tasks: {tasks}")

            with open(f"{log_dir}/data_gen_{idx:03d}.json", "w") as f:
                json_str = json.dumps(rnd_data, indent=2)
                f.write(json_str)

            # save graphs separately for Graph handler
            graph_path = f"{log_dir}/graph_gen_{idx:03d}.json"
            with open(graph_path, "w") as f:
                json_str = json.dumps(rnd_data["graph"], indent=2)
                f.write(json_str)

            for task_idx, task in enumerate(tasks):
                graph_handle = GraphHandler(
                    graph_path=graph_path, init_node=rnd_data["graph"]["robot_location"]
                )
                # graph_handle.reset(
                #    rnd_data["graph"],
                #    current_location=,
                # )
                graph_data_gen = GraphSim(graph_handle)
                # breakpoint()
                # TODO: fix node removal
                unknown_pct = self.unknown_pcts[task_idx % len(self.unknown_pcts)]
                graph_data_gen.randomly_remove_nodes(pct=unknown_pct)

                # log_name = f"{log_dir}/sample_{idx:03d}_{task_idx:03d}_unknown_pct_{unknown_pct}_n_regions_{n_regions}_n_objects_{n_objects}.json"
                log_name = f"{log_dir}/sample_{idx:03d}_{task_idx:03d}_n_regions_{n_regions}_n_objects_{n_objects}.json"
                planner = SPINE(graph=graph_data_gen.partial_graph, log_name=log_name)
                # breakpoint()
                out = self.planning_sim.run_planning(
                    llm_planner=planner, task=task, graph_data_gen=graph_data_gen
                )

                # some simple verification. Mark plans that don't come up with an answer
                try:
                    if not out["plan"][0][0].startswith("answer"):
                        os.rename(
                            log_name, log_name.replace(".json", "_failed") + ".json"
                        )
                except:
                    os.rename(log_name, log_name.replace(".json", "failed") + ".json")

                data_counter += 1

        aggregate(
            root_dir=log_dir,
            glob_str="sample*json",
            out_file=f"{log_dir}/formatted.json",
        )
