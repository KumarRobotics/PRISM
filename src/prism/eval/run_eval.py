import json
import re
from collections import namedtuple
from typing import Dict, List, Tuple

from spine.mapping.graph_util import GraphHandler
from spine.spine import SPINE

from prism.data.graph_sim import GraphSim
from prism.data.planning_sim import PlanningSim
from prism.models.unsloth import from_pretrained

# Modified to accept either a file path or a graph data dictionary
EvalSample = namedtuple("EvalSample", ["task", "answer", "graph", "init_node"])


class EvalResult:
    def __init__(self, formatted: bool, plan_keyword: bool):
        self.formatted = formatted
        self.plan_keyword = plan_keyword

    def is_correct(self):
        return self.formatted and self.plan_keyword


def correct_keys(answer: Dict[str, str]) -> bool:
    return (
        "primary_goal" in answer
        and "relevant_graph" in answer
        and "reasoning" in answer
        and "plan" in answer
    )


def eval_answer(parsed_answer: Dict[str, str], answer_key: str):
    formatted = False
    keyphrase = False
    try:
        formatted = correct_keys(parsed_answer)

        keyphrase = bool(
            re.search(answer_key, str(parsed_answer["plan"]), re.IGNORECASE)
        )

        return EvalResult(formatted=formatted, plan_keyword=keyphrase), parsed_answer
    except:
        return EvalResult(False, False), parsed_answer


def to_json(output: str) -> Tuple[str, bool]:
    try:
        s = json.loads(output)
        return s, True
    except:
        output, False


class Unsloth:
    def __init__(self, model_path: str, is_four_bit: bool):
        self.model, self.tokenizer = from_pretrained(
            path=model_path, inference=True, load_in_4bit=is_four_bit
        )

    def run(self, task: str, graph_handler: GraphHandler):
        messages = [
            {
                "role": "user",
                "content": f"task: {task}. scene graph {graph_handler.to_json_str()}",
            }
        ]

        print(f"\n====\n\ntask: {task}\n----\n")

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=4048,
            use_cache=True,
            temperature=0.01,
            min_p=0.1,
        )
        out = self.tokenizer.batch_decode(outputs)

        planner_response = out[0].split("end_header_id|>")[-1].split("<|eot_id|>")[0]
        return planner_response


def eval_model(
    *, model_path: str, is_four_bit: bool, eval_samples: List[EvalSample]
) -> float:
    total_correct = 0

    multi_turn = True

    if multi_turn:
        graph_handler = GraphHandler("")
        graph_sim = GraphSim(graph_handler)
        llm_planner = SPINE(
            graph=graph_sim.partial_graph,
            llm="unsloth",
            model_path=model_path,
        )

        model = PlanningSim(debug=False)
    else:
        model = Unsloth(model_path=model_path, is_four_bit=is_four_bit)

    for eval_sample in eval_samples:
        graph_path = eval_sample.graph
        init_node = eval_sample.init_node
        task = eval_sample.task
        answer = eval_sample.answer

        if multi_turn:
            graph_sim.reset(graph_as_dict=graph_path, current_location=init_node)
            llm_planner.graph = graph_sim.partial_graph

            planner_response = model.run_planning(
                llm_planner=llm_planner,
                task=task,
                graph_data_gen=graph_sim,
                max_iterations=10,
            )

        else:
            planner_response = model(task=task, graph_handler=graph_handler)

            try:
                planner_response = json.loads(planner_response)
            except:
                planner_response = {"wrong": planner_response}

        result, formatted_answer = eval_answer(planner_response, answer)

        if result.formatted:
            print(formatted_answer)
        else:
            print(f"incorrect formatting\n{formatted_answer}")

        print(f"correct answer: {result.plan_keyword}")

        print(f"\n=====\n")

        total_correct += result.plan_keyword

    return total_correct / len(eval_samples)
