import os

os.environ["WANDB_PROJECT"] = "SLM-distill"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from transformers import HfArgumentParser

from prism.training.train import TrainConfig, train_model

if __name__ == "__main__":
    parser = HfArgumentParser(TrainConfig)
    (config,) = parser.parse_args_into_dataclasses()

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{config.checkpoint_dir}/training_config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    print(config)
    train_model(config)
