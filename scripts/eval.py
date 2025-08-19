import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# from llm_distillation.eval.run_eval import EvalSample, eval_model
from llm_distillation.eval.run_eval_new import (EvalResult, EvalSample,
                                                eval_model)

ROOT = Path(
    importlib.util.find_spec("llm_distillation").submodule_search_locations[0]
).parents[:2][1]

eval_config_path = ROOT / "data/eval/eval_1_multi_step_perch.json"

# (path, is four bit)
# MODEL_PATH = (str(ROOT / "models/full_gen_v1_llama-3_1-8b_r16"), False)
# MODEL_PATH = (str(ROOT / "models/gen_fully_known_v1_llama-3_2-3b_r32"), False)

# BASE_PATH = [
#     "/home/zacravi/projects/llm-distillation/models/gen_fully_known_v1_llama-3_2-3b_r32"
# ]
# MODEL_PATH = BASE_PATH

MODEL_PATH = sorted(
    Path(
        "/home/zacravi/projects/llm-distillation/results/base_trainer/ablations/with-fail"
    ).rglob("*checkpoint-*")
)

# MODEL_PATH = [
#     "/home/zacravi/projects/llm-distillation/results/base_trainer/train_explore_combo/v1/outputs/checkpoint-196"
# ]

# MODEL_PATH = sorted(
#     Path(
#         "/home/zacravi/projects/llm-distillation/results/base_trainer/high_temp_data"
#     ).rglob("*checkpoint-*")
# )

# MODEL_PATH = [
#     "/home/zacravi/projects/llm-distillation/results/base_trainer/semantic_curriculum/v4/outputs/checkpoint-28"
# ]

# MODEL_PATH = BASE_PATH + MODEL_PATH

# MODEL_PATH.append(BASE_PATH)

# MODEL_PATH = [
#     "/home/zacravi/projects/llm-distillation/results/noniterative/v1/outputs/checkpoint-171"
# ]


if __name__ == "__main__":
    eval_config_path = str(eval_config_path)

    with open(eval_config_path) as f:
        eval_config = json.load(f)

    for model in MODEL_PATH:
        model = str(model)
        eval_samples = []
        graph = eval_config["graph"]
        TASKS = eval_config["tasks"]

        print(f"evaluating: {model}")

        for entry in TASKS:
            eval_samples.append(
                EvalSample(
                    task=entry["task"],
                    answer=entry["answer"],
                    graph=graph,
                    init_node=entry["init_node"],
                )
            )

        try:
            result_correct = eval_model(
                model_path=model,
                is_four_bit=False,  # MODEL_PATH[1],
                eval_samples=eval_samples,
            )
        except Exception as ex:
            print(f"failed: {ex}")

            result_correct = 0

        print(result_correct)

        with open(f"with-fail.txt", "a") as f:
            f.write(f"{result_correct}, {model}\n")
