import json
import random
from pathlib import Path
from typing import Dict, List

from llm_planner_alfred.hlp_planner import LLM_Planner, clean_llm_output
from tqdm import tqdm

from prism.data.utils import GPTQueryClient, aggregate

QUERY = """
You are a data generator for synthesizing tasks for the ALFRED simulator.

You should use objects from the following list: 'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bowl', 'Box', 'Bread', 'ButterKnife', 'Cabinet', 'Candle', 'Cart', 'CD', 'CellPhone', 'Chair', 'Cloth', 'CoffeeMachine', 'CounterTop', 'CreditCard', 'Cup', 'Curtains', 'Desk', 'DeskLamp', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'FloorLamp', 'Footstool', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LightSwitch', 'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Poster', 'Pot', 'Potato', 'RemoteControl', 'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'ShowerGlass', 'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveBurner', 'StoveKnob', 'DiningTable', 'CoffeeTable', 'SideTable', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'Towel', 'TowelHolder', 'TVStand', 'Vase', 'Watch', 'WateringCan', 'Window', 'WineBottle'

The agent has the following capabilities
OpenObject
CloseObject
PickupObject
PutObject
ToggleObjectOn
ToggleObjectOff
SliceObject
Navigation

Your response should be json with the following keys
{
tasks: ["task_1", "task_2", ...],
visible objects: [list of objects in the scene],
reasoning: your reasoning
}"""

task_schema = {
    "task_instr": [],  # the task
    "step_instr": [],  # list of possible instructions
    "vis_objects": [],
    "completed_plans": [],
}


class AlfredDataGen:
    def __init__(self, log_dir: str):
        self.client = GPTQueryClient()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def gen_input_data(
        self, n_objects: int, n_tasks: int, description: str = "", num_tries: int = 10
    ) -> Dict[str, List[str]]:
        if isinstance(n_objects, list):
            n_objects = random.choice(n_objects)

        if isinstance(n_tasks, list):
            n_tasks = random.choice(n_tasks)

        query = (
            QUERY
            + f"\nPick {n_objects} visible objects, the generate {n_tasks} tasks associated with them."
        )
        if description != "":
            query += (
                f"\nThe tasks and scene should be like the following: {description}"
            )

        for i in range(num_tries):
            response = self.client.query_gpt(query=query, temperature=1, max_tokens=300)

            data = json.loads(response.choices[0].message.content)
            # check if the data is valid
            if "tasks" in data and "visible objects" in data:
                break
            elif i == num_tries - 1:
                raise ValueError(f"GPT returned invalid data for {query}")

        gen_tasks = []
        for task in data["tasks"]:
            gen_tasks.append(
                {
                    "task_instr": [task],
                    "step_instr": [],
                    "vis_objs": data["visible objects"],
                    "completed_plans": [],
                }
            )

        return {"tasks": gen_tasks, "n_objects": n_objects, "n_tasks": n_tasks}

    def generate(self, n_samples: int, n_tasks: int, n_objects: int, k: int = 5):

        for sample in tqdm(range(n_samples), desc="Generating data"):
            print(f"Generating tasks for sample {sample}")
            generate_result = self.gen_input_data(n_objects=n_objects, n_tasks=n_tasks)
            input_datum = generate_result["tasks"]

            for idx, input_data in enumerate(input_datum):
                print(f"Generating plans for sample {sample}, task {idx}")
                n_obj, n_task = generate_result["n_objects"], generate_result["n_tasks"]
                log_path = (
                    self.log_dir
                    / f"sample_{sample:03d}_task_{idx:03d}_objects_{n_obj:03d}_tasks_{n_task:03d}.json"
                )

                planner = LLM_Planner(llm="gpt-4o", log_name=log_path)
                prompt = planner.generate_prompt(input_data, k=k)
                out = planner.call_llm(prompt=prompt)
        aggregate(
            self.log_dir,
            glob_str="sample*json",
            out_file=str(self.log_dir / "formatted.json"),
            cutbefore=0,
        )


if __name__ == "__main__":
    test_log_path = "~/llm-distillation/data/alfred"
    alfred_data_gen = AlfredDataGen(log_dir=test_log_path)
    alfred_data_gen.generate(
        n_samples=10, n_tasks=3, n_objects=[i for i in range(1, 10)]
    )
