import json
import os

os.environ["WANDB_PROJECT"] = "SLM-distill"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from dataclasses import dataclass, field
from typing import List

import wandb
from datasets import load_dataset
from transformers import HfArgumentParser
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

from prism.eval.callbacks import EvalCallback
from prism.eval.run_eval import EvalSample
from prism.training.utils import (TurnAwareCollator,
                                  get_formatting_prompts_func,
                                  train_on_responses_only)


@dataclass
class TrainConfig:
    name: str
    checkpoint_dir: str
    data: str
    bit4: bool = False
    eval_data: str = "../data/eval/eval_1_multi_step.json"
    r: int = 16
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    wandb_project: str = "SLM-distill"
    epochs: int = 2
    val_frac: float = 0.1  # Fraction of training data to use for validation
    # New parameters for training arguments and model configuration
    lora_alpha: int = 16
    lora_dropout: float = 0.2
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_pror",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8  # Eval batch size can be different
    gradient_accumulation_steps: int = 2
    report_to: str = "wandb"
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    weight_decay: float = 0.05
    debug: bool = False


def train_model(config: TrainConfig) -> None:
    """Train model via unsloth

    Notes
    -----
    Uses Lora SFT.
    """
    SAVE_NAME = config.name
    max_seq_length = 2048
    load_in_4bit = False
    dtype = None

    SAVE_NAME += f"_r{config.r}"
    if config.bit4:
        SAVE_NAME += "_4bit"
        load_in_4bit = True

    base_model = config.base_model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    # Load the full dataset
    full_dataset = load_dataset("json", data_files=[config.data], split="train")

    # Subsample if in debug mode
    if config.debug:
        full_dataset = full_dataset.select(range(min(100, len(full_dataset))))

    # Apply preprocessing
    full_dataset = standardize_sharegpt(full_dataset)
    formatting_prompts_func = get_formatting_prompts_func(tokenizer)
    full_dataset = full_dataset.map(formatting_prompts_func, batched=True)

    # Create train/validation split
    if config.val_frac > 0:
        # Calculate split sizes
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * config.val_frac)
        train_size = dataset_size - val_size

        # Create the splits
        train_val_split = full_dataset.train_test_split(
            test_size=val_size,
            train_size=train_size,
            seed=3407,  # Use the same seed for reproducibility
        )

        # Get the splits
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]

        print(
            f"Dataset split: {train_size} training samples, {val_size} validation samples"
        )
    else:
        # Use all data for training if val_frac is 0
        train_dataset = full_dataset
        val_dataset = None
        print(f"Using all {len(full_dataset)} samples for training (no validation)")

    # Use the training dataset for SFTTrainer
    dataset = train_dataset

    sft_config = SFTConfig(
        # max_seq_length=max_seq_length,  # deprecated in updated version
        dataset_text_field="text",
        packing=False,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to=config.report_to,
    )
    # Setup SFT trainer with train and optionally validation dataset
    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": dataset,
        "data_collator": TurnAwareCollator(
            tokenizer=tokenizer, padding="longest", include_turns=False
        ),
        "args": sft_config,
    }

    # Add validation dataset if available
    if val_dataset is not None:
        trainer_kwargs["eval_dataset"] = val_dataset

    trainer = SFTTrainer(**trainer_kwargs)

    # Apply train_on_responses_only which adds the turn field to the dataset
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Load eval samples before training
    with open(config.eval_data) as f:
        data = json.load(f)
        tasks = data["tasks"]
        graph_data = data["graph"]  # Get the graph data

    eval_samples = []
    for entry in tasks:
        eval_samples.append(
            EvalSample(
                task=entry["task"],
                answer=entry["answer"],
                graph=graph_data,  # Pass the graph data dictionary directly
                init_node=entry["init_node"],
            )
        )

    # Add both callbacks
    trainer.add_callback(EvalCallback(eval_samples))

    trainer.train()

    return trainer


if __name__ == "__main__":
    parser = HfArgumentParser(TrainConfig)
    (config,) = parser.parse_args_into_dataclasses()
    print(config)
    train_model(config)
