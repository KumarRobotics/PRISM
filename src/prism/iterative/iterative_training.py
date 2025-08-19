import copy
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from unsloth.chat_templates import standardize_sharegpt

from prism.data.data_gen import DataGenerator
from prism.iterative.utils import spine_data_to_prompt
from prism.training.train import TrainConfig, train_model
from prism.training.utils import TurnAwareCollator, get_formatting_prompts_func


@dataclass
class IterativePipelineConfig:
    """Configuration for the iterative training pipeline."""

    initial_data_path: str = "../data/gpt_gen_formatted.json"
    output_dir: str = "output/iterative_training"
    n_episodes: int = 5
    n_batch: int = 10
    held_out_data_path: Optional[str] = None
    seed: int = 42
    graph_unknown: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.3]
    )
    min_regions: int = 8
    max_regions: int = 15
    min_objects: int = 5
    max_objects: int = 15
    tasks_per_sample: int = 2
    graphs_per_sample: int = 1
    eval_task_file: Optional[str] = None
    skip_eval_tasks: bool = False
    initial_dataset_size: Optional[int] = None
    objective: str = "eval/mean_accuracy"
    objective_type: str = "max"
    eval_metric: str = "eval/mean_accuracy"
    sampling_strategy: str = "accuracy_weighted"
    data_format: str = "spine"


class IterativeTrainingPipeline:
    """Iterative training pipeline that follows train-evaluate-generate-retrain cycle."""

    def __init__(
        self,
        pipeline_config: IterativePipelineConfig,
        train_config: TrainConfig,
    ):
        """
        Initialize the iterative training pipeline.

        Args:
            pipeline_config: Configuration for the iterative training pipeline.
            train_config: Training configuration.
        """
        # Set seeds for reproducibility
        random.seed(pipeline_config.seed)
        np.random.seed(pipeline_config.seed)
        torch.manual_seed(pipeline_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(pipeline_config.seed)

        # Setup output directory
        self.output_dir = Path(pipeline_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store config objects
        self.pipeline_config = pipeline_config
        self.train_config = train_config

    def _get_data_gen_prompt_from_example(self, example: str) -> str:
        if self.pipeline_config.data_format == "spine":
            return spine_data_to_prompt(example)
        else:
            raise ValueError(f"{self.pipeline_config.data_format} is not supported")

    def _save_dataset(self, dataset: Dataset, path: str):
        """Save a dataset to a standard JSON file list format."""
        # Convert dataset to a list of dictionaries
        samples = dataset.to_list()

        # Save as a standard JSON list
        with open(path, "w") as f:
            json.dump(samples, f, indent=2)

    def _load_dataset(self, data_path: str) -> Dataset:
        """Load a dataset from a JSON file.

        Args:
            data_path: Path to the JSON file containing the dataset.

        Returns:
            Dataset object.
        """
        with open(data_path, "r") as f:
            data = json.load(f)
        return Dataset.from_list(data)

    def _train_model(self, episode: int) -> str:
        """
        Train a model on the current dataset.

        Args:
            episode: Current episode number.

        Returns:
            Path to the saved model.
        """
        # Update train config for this episode
        episode_config = self.train_config
        episode_config.name = f"{self.train_config.name}_episode_{episode}"

        # Set data path based on episode number
        if episode == 1:
            # Use the original initial data path for the first episode
            print(
                f"Using initial data path for episode 1: {self.pipeline_config.initial_data_path}"
            )
            episode_config.data = self.pipeline_config.initial_data_path
        else:
            # For subsequent episodes, save the current dataset and use its path
            dataset_path = self.output_dir / f"dataset_episode_{episode}.json"
            print(f"Saving dataset for episode {episode} to {dataset_path}")
            self._save_dataset(self.current_dataset, dataset_path)
            episode_config.data = str(dataset_path)

        # Train the model
        trainer = train_model(episode_config)

        # Return path to the trained model
        return trainer

    def _sample_existing_data(self) -> str:
        """Sample existing data from the current dataset."""
        if self.pipeline_config.sampling_strategy == "uniform":
            random_idx = random.randint(0, len(self.current_dataset) - 1)
        elif self.pipeline_config.sampling_strategy == "accuracy_weighted":
            # Use numpy for weighted sampling
            weights = 1 / np.array(self.current_metrics["train/mean_accuracy"])
            random_idx = np.random.choice(
                len(self.current_dataset), p=weights / np.sum(weights)
            )
        elif self.pipeline_config.sampling_strategy == "loss_weighted":
            # Use numpy for weighted sampling
            weights = np.array(self.metrics["train/mean_loss"])
            random_idx = np.random.choice(
                len(self.current_dataset), p=weights / np.sum(weights)
            )
        return self.current_dataset[random_idx]["conversations"][0]["content"]

    def _get_data_generator(self) -> DataGenerator:
        # Initialize the data generator
        # Vary region and object counts for diversity
        n_regions_list = [
            random.randint(
                self.pipeline_config.min_regions, self.pipeline_config.max_regions
            )
            for _ in range(self.pipeline_config.n_batch)
        ]
        n_objects_list = [
            random.randint(
                self.pipeline_config.min_objects, self.pipeline_config.max_objects
            )
            for _ in range(self.pipeline_config.n_batch)
        ]
        # Vary graph unknowns (percentage of nodes removed) for difficulty
        graph_unknown = self.pipeline_config.graph_unknown

        data_generator = DataGenerator(
            graph_unknown=graph_unknown,
            n_region_list=n_regions_list,
            n_objects_list=n_objects_list,
        )
        return data_generator

    def _generate_new_data(self, episode: int) -> Dataset:
        """
        Generate new data based on the current dataset.

        Args:
            episode: Current episode number.

        Returns:
            Dataset with new samples.
        """
        print(f"Generating {self.pipeline_config.n_batch} new samples...")
        # Create data generation directory for this episode
        gen_dir = self.output_dir / f"data_gen_episode_{episode}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(self.pipeline_config.n_batch):
            in_context_example = self._sample_existing_data()
            data_gen_prompt = self._get_data_gen_prompt_from_example(in_context_example)
            # construct data generator here in order to add graph size
            # diversity. Then generate samples.
            data_generator = self._get_data_generator()
            data_generator.generate(
                log_dir=str(gen_dir),
                n_samples=self.pipeline_config.graphs_per_sample,
                n_tasks=self.pipeline_config.tasks_per_sample,
                description=data_gen_prompt,
            )
        # Process the generated data into a dataset format
        new_samples = []
        generated_files = list(gen_dir.glob("sample_*.json"))

        for file_path in generated_files:
            if "_failed" in file_path.name:
                continue  # Skip failed generations

            try:
                with open(file_path, "r") as f:
                    log_data = json.load(f)

                # Extract conversation from planning log
                if "prompt" in log_data and "response" in log_data:
                    # Each sample should have its own conversations array
                    conversations = []

                    # Add initial description
                    description = "Exploring a new environment"
                    desc_file = (
                        gen_dir
                        / f"data_gen_{file_path.name.split('_')[1].zfill(3)}.json"
                    )
                    if desc_file.exists():
                        try:
                            with open(desc_file, "r") as f:
                                gen_data = json.load(f)
                                description = gen_data.get(
                                    "description", "Exploring a new environment"
                                )
                        except:
                            pass  # Use default description if file can't be read

                    conversations.append(
                        {
                            "role": "user",
                            "content": f"Let's explore a new environment: {description}",
                        }
                    )

                    # Add the planning task
                    conversations.append({"role": "user", "content": log_data["task"]})

                    # Add the model response/planning steps
                    planning_steps = []
                    if "planning_steps" in log_data:
                        planning_steps = log_data["planning_steps"]
                    elif isinstance(log_data.get("response"), list):
                        planning_steps = log_data["response"]

                    if planning_steps:
                        # Join all planning steps into a single response
                        planning_text = "\n".join(
                            [step for step in planning_steps if step]
                        )
                        conversations.append(
                            {"role": "assistant", "content": planning_text}
                        )

                        # Add sample with proper structure - each item has its own conversations array
                        new_samples.append({"conversations": conversations})
                        print(f"Processed sample from {file_path.name}")

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

        # If we couldn't generate proper samples, create some placeholder ones
        # Create a dataset from the new samples - ensures each sample is a separate dictionary
        # with a 'conversations' key containing its conversation turns
        new_dataset = Dataset.from_list(new_samples)

        # Save the generated samples for inspection
        with open(self.output_dir / f"new_samples_episode_{episode}.json", "w") as f:
            json.dump(new_samples, f, indent=2)

        print(f"Generated {len(new_samples)} new samples")
        return new_dataset

    def _combine_datasets(self, dataset1: Dataset, dataset2: Dataset) -> Dataset:
        """Combine two datasets.

        Each dataset should contain samples with a 'conversations' field.
        Returns a new dataset with samples from both datasets.
        """
        # Convert datasets to lists of samples
        samples1 = dataset1.to_list()
        samples2 = dataset2.to_list()

        # Combine the lists
        combined_samples = samples1 + samples2

        # Create a new dataset from the combined list
        return Dataset.from_list(combined_samples)

    def _evaluate_dataset(self, trainer, split="train"):
        """Evaluate either the train or eval dataset, depending on the split argument."""
        # Make sure the model is in evaluation mode
        trainer.model.eval()

        # Get the correct dataloader
        if split == "train":
            collator = TurnAwareCollator(
                tokenizer=trainer.tokenizer, padding="longest", include_turns=False
            )
            dataloader = DataLoader(
                trainer.get_train_dataloader().dataset,
                batch_size=trainer.args.per_device_train_batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            # preprocess the dataset
        else:
            dataloader = trainer.get_eval_dataloader()
            if dataloader is None:
                print("No eval dataset is available. Returning empty metrics.")
                return {}

        all_losses = []
        all_accuracies = []
        for step, batch in enumerate(dataloader):
            # Move batch to the correct device
            batch = {
                k: v.to(trainer.model.device, non_blocking=True)
                for k, v in batch.items()
            }
            with torch.no_grad():
                outputs = trainer.model(**batch)
                logits = outputs.logits  # shape [batch_size, seq_len, vocab_size]
                labels = batch["labels"]  # shape [batch_size, seq_len]

                # Per-token cross-entropy (no reduction yet)
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                losses = loss_fct(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )
                losses = losses.view(labels.shape)  # back to [batch_size, seq_len]

                # Ignore masked tokens labeled as -100
                mask = labels != -100

                # Compute average loss per sample (sum over tokens / number of valid tokens)
                sample_loss = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

                # Compute token-level accuracy
                preds = logits.argmax(dim=-1)
                correct = (preds == labels) & mask
                sample_accuracy = correct.sum(dim=1).float() / mask.sum(
                    dim=1
                ).clamp_min(1)

            all_losses.append(sample_loss.cpu())
            all_accuracies.append(sample_accuracy.cpu())

        # Gather all per-sample metrics
        all_losses = torch.cat(all_losses, dim=0)
        all_accuracies = torch.cat(all_accuracies, dim=0)

        metrics = {
            f"{split}/mean_loss": all_losses.mean(),
            f"{split}/mean_accuracy": all_accuracies.mean(),
            f"{split}/per_sample_loss": all_losses,  # 1D tensor with per-sample cross-entropy
            f"{split}/per_sample_accuracy": all_accuracies,  # 1D tensor with per-sample accuracy
        }
        return metrics

    def _post_training_evaluation(self, trainer, data_path=None) -> dict:
        """
        Evaluate the model for all samples in provided data.

        Args:
            trainer: The SFTTrainer instance after training.
            data_path: Optional path to evaluation data. If None, uses trainer's datasets.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Use trainer's model and tokenizer
        model = trainer.model
        tokenizer = trainer.tokenizer

        # Function to format examples as in train.py
        formatting_prompts_func = get_formatting_prompts_func(tokenizer)

        # Load dataset if provided, otherwise use trainer's datasets
        if data_path:
            print(f"Loading evaluation data from {data_path}")
            eval_dataset = load_dataset("json", data_files=[data_path], split="train")
            eval_dataset = standardize_sharegpt(eval_dataset)
            eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
        else:
            # Use trainer's datasets
            train_dataset = trainer.train_dataset
            eval_dataset = (
                trainer.eval_dataset
                if hasattr(trainer, "eval_dataset") and trainer.eval_dataset
                else None
            )

        metrics = {}

        # Evaluation function

        # Evaluate train dataset if available
        if train_dataset:
            train_metrics = self._evaluate_dataset(trainer, split="train")
            metrics.update(train_metrics)
            print(
                f"Train metrics: Loss = {train_metrics['train/mean_loss']:.4f}, Token Accuracy = {train_metrics['train/mean_accuracy']:.4f}"
            )

        # Evaluate validation dataset if available
        if eval_dataset:
            eval_metrics = self._evaluate_dataset(trainer, split="eval")
            metrics.update(eval_metrics)
            print(
                f"Eval metrics: Loss = {eval_metrics['eval/mean_loss']:.4f}, Token Accuracy = {eval_metrics['eval/mean_accuracy']:.4f}"
            )

        return metrics

    def run_pipeline(self):
        """Run the iterative training pipeline."""
        # Initialize wandb
        self.run = wandb.init(
            project=self.train_config.wandb_project,
            name=f"iterative-training-{self.train_config.name}",
            config={**asdict(self.pipeline_config), **asdict(self.train_config)},
        )

        # Load initial dataset
        print(f"Loading initial dataset from {self.pipeline_config.initial_data_path}")
        self.current_dataset = self._load_dataset(
            self.pipeline_config.initial_data_path
        )

        # Subsample the initial dataset if initial_dataset_size is provided
        if self.pipeline_config.initial_dataset_size is not None:
            print(
                f"Subsampling initial dataset to {self.pipeline_config.initial_dataset_size} samples"
            )
            self.current_dataset = self.current_dataset.select(
                range(self.pipeline_config.initial_dataset_size)
            )
            # save the subsampled dataset in a new path as episode 0
            dataset_path = self.output_dir / f"dataset_episode_0.json"
            self._save_dataset(self.current_dataset, dataset_path)
            # update the initial_data_path to the new path
            self.pipeline_config.initial_data_path = str(dataset_path)

        print(f"Initial dataset size: {len(self.current_dataset)}")

        # Load held out dataset if provided
        if self.pipeline_config.held_out_data_path:
            print(
                f"Loading held out dataset from {self.pipeline_config.held_out_data_path}"
            )
            self.held_out_dataset = self._load_dataset(
                self.pipeline_config.held_out_data_path
            )
            print(f"Held out dataset size: {len(self.held_out_dataset)}")
        else:
            self.held_out_dataset = None

        # Track best model and performance
        best_model_path = None
        best_performance = 0.0
        # copy train config to episode config
        self.episode_config = copy.deepcopy(self.train_config)
        for episode in range(1, self.pipeline_config.n_episodes + 1):
            print(
                f"\n=== Starting Episode {episode}/{self.pipeline_config.n_episodes} ==="
            )
            # Train model
            trainer = self._train_model(episode)
            # Evaluate model
            self.current_metrics = self._post_training_evaluation(trainer)
            # eval_metrics = self._evaluate_on_tasks(model_path, split="val")

            objective = self.current_metrics[self.pipeline_config.objective]
            if self.pipeline_config.objective_type == "min":
                objective = -objective
            # Update best model if performance improved
            if objective > best_performance:
                best_performance = objective
                best_model_path = self.episode_config.save_path
                print(
                    f"New best model with {self.pipeline_config.objective}: {self.current_metrics[self.pipeline_config.objective]:.4f}"
                )

            # Generate new data
            new_dataset = self._generate_new_data(episode)

            # Combine with existing dataset
            self.current_dataset = self._combine_datasets(
                self.current_dataset, new_dataset
            )

            # Log dataset size
            if self.run:
                self.run.log(
                    {
                        "dataset/size": len(self.current_dataset),
                        "episode": episode,
                        "best_task_accuracy": best_performance,
                    }
                )

            print(
                f"=== Episode {episode} completed. Dataset size: {len(self.current_dataset)} ===\n"
            )

            # Save final best model symlink
            if best_model_path:
                best_model_symlink = self.output_dir / "best_model"
                if os.path.islink(best_model_symlink):
                    os.unlink(best_model_symlink)
                os.symlink(os.path.abspath(best_model_path), best_model_symlink)
                print(
                    f"Best model linked at {best_model_symlink} with accuracy {best_performance:.4f}"
                )

        if self.run:
            self.run.finish()


def run_iterative_training(
    pipeline_config: IterativePipelineConfig, train_config: TrainConfig
):
    pipeline = IterativeTrainingPipeline(
        pipeline_config=pipeline_config,
        train_config=train_config,
    )
    pipeline.run_pipeline()


def main():
    # Use HuggingFace's ArgumentParser to parse both configs
    parser = HfArgumentParser((IterativePipelineConfig, TrainConfig))
    pipeline_args, train_args = parser.parse_args_into_dataclasses()

    # Initialize pipeline
    pipeline = IterativeTrainingPipeline(
        pipeline_config=pipeline_args,
        train_config=train_args,
    )
    # Run pipeline
    pipeline.run_pipeline()
    # train_model(train_args)


if __name__ == "__main__":
    main()
