# Iterative Training Pipeline

This module implements an iterative training pipeline that follows a train-evaluate-generate-retrain cycle:

1. Trains a model on an initial dataset
2. Evaluates the model on the training dataset, a held-out dataset (optional), and a task evaluation set (optional)
3. Generates new data samples using the TaskGraphGen and DataGenerator
4. Uses existing samples as context/prior for generating new samples
5. Combines the new data with the existing dataset
6. Repeats the process for a specified number of episodes

## Usage

### Command-line

```bash
python ./src/prism/iterative/run_iterative_training.py \
    --initial_data /path/to/initial_data.json \
    --output_dir /path/to/output \
    --n_episodes 2 \
    --n_batch 10 \
    --eval_task_file eval_tasks_1.json \
    --graph_unknown 0.2,0.3,0.4,0.5
```

For debugging, you can subsample the initial dataset:

```bash
python ./src/prism/iterative/run_iterative_training.py \
    --initial_data /path/to/initial_data.json \
    --output_dir /path/to/output \
    --n_episodes 1 \
    --n_batch 5 \
    --initial_dataset_size 50 \
    --skip_eval_tasks
```

### Python API

```python
from prism.training.train import TrainConfig
from prism.iterative import IterativeTrainingPipeline

# Create training config
train_config = TrainConfig(
    name="iterative_model",
    data="/path/to/initial_data.json",
    r=32,
    base_model="llama-3.2-3b",
    epochs=3,
)

# Initialize pipeline
pipeline = IterativeTrainingPipeline(
    initial_data_path="/path/to/initial_data.json",
    output_dir="/path/to/output",
    n_episodes=5,
    n_batch=10,
    train_config=train_config,
    held_out_data_path="/path/to/held_out_data.json",
    seed=42,
    eval_task_file="eval_tasks_1.json",
    graph_unknown=[0.2, 0.3, 0.4, 0.5],
    min_regions=8,
    max_regions=15,
)

# Run pipeline
pipeline.run_pipeline()
```

## Parameters

### Basic Parameters
- `initial_data`: Path to the initial training dataset file in JSON format. This dataset serves as the starting point for the iterative training process.
- `output_dir`: Directory where all outputs will be saved, including trained models, generated datasets, and evaluation results.
- `n_episodes`: Number of training episodes to run (default: 5). Each episode consists of a full generate-train-evaluate cycle.
- `n_batch`: Number of new samples to generate per episode (default: 10). Higher values increase dataset growth rate but require more generation time.
- `held_out_data`: Path to a held-out dataset in JSON format for evaluation (optional). This dataset is not used for training but only to evaluate model generalization.
- `val_frac`: Fraction of training data to use for validation (default: 0.1).
- `n_episodes`: Controls the number of train-evaluate-generate-retrain cycles. Each episode builds on knowledge from previous episodes, creating a curriculum of increasingly sophisticated training data.
- `sampling_strategy`: Strategy for sampling existing data to use as context when generating new samples. Available options:
  - `uniform`: Random sampling with equal probability (default)
  - `accuracy_weighted`: Samples weighted by inverse accuracy, prioritizing harder examples
  - `loss_weighted`: Samples weighted by loss value, focusing on examples the model struggles with
- `objective`: Metric used to determine the best model across episodes (default: "eval/mean_accuracy")
- `objective_type`: Whether to maximize or minimize the objective (default: "max")
- `eval_metric`: Primary metric for evaluation reporting (default: "eval/mean_accuracy")

## Data Generation Process

The pipeline uses the following process to generate new data:

1. For each episode, it creates a new directory for data generation: `data_gen_episode_{episode}`
2. It samples existing datapoints from the current dataset to use as context/prior
3. It initializes a `DataGenerator` with the specified parameters
4. It generates new graph data using the `TaskGraphGen` class
5. It processes the generated data into a conversation format compatible with the training pipeline
6. If data generation fails, it falls back to placeholder samples based on TaskGraphGen output

## Evaluation Process

The pipeline evaluates trained models in three ways:

1. **Training Data Evaluation**: Computes token-wise cross-entropy loss and accuracy on the training data
2. **Held-out Data Evaluation** (optional): Computes the same metrics on a held-out dataset
3. **Task Evaluation** (optional): Evaluates the model on standard planning tasks using `EvalSample` from `run_eval.py`

The best model across episodes (based on task evaluation accuracy) is symlinked as `best_model` in the output directory.

## Debugging Tips

When debugging the pipeline, you can use the following strategies:

1. **Subsample the Initial Dataset**: Use `--initial_dataset_size N` to train on a smaller dataset
2. **Reduce Episode Count**: Set `--n_episodes` to 1 to quickly test the entire workflow
3. **Generate Fewer Samples**: Use a smaller `--n_batch` value to reduce data generation time
4. **Skip Task Evaluation**: Use `--skip_eval_tasks` to avoid running the evaluation process
5. **Set a Small Number of Epochs**: Use `--epochs 1` for faster training cycles

## Data Format

The expected data format is a JSON file with conversations in the following structure:

```json
[
  {
    "conversations": [
      {
        "role": "user",
        "value": "User message"
      },
      {
        "role": "assistant",
        "value": "Assistant response"
      },
      ...
    ]
  },
  ...
]
```

## Output

For each episode, the pipeline saves:
- The current dataset (`dataset_episode_N.json`)
- The trained model (`iterative_model_episode_N`)
- The new samples generated (`new_samples_episode_N.json`)
- Data generation files in `data_gen_episode_N/`
- A symlink to the best model (`best_model`)

## Metrics

The pipeline logs the following metrics to wandb:
- Training loss and accuracy
- Held-out loss and accuracy (if provided)
- Task evaluation accuracy (if provided)
- Sample-level accuracy histograms
- Dataset size growth
- Best model performance
