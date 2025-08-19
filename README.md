# PRISM

Code for the paper [Distilling On-device Language Models for Robot Planning with Minimal Human Intervention](https://arxiv.org/abs/2506.17486).

PRISM is a framework to distill on-device robot planners. PRISM takes as input a source large langauge model (LLM)-enabled planner and automatically generates training data with which to distill a small langauge model (SLM)

## Installation

Clone this repo, install requirements, and install
```
git clone git@github.com:KumarRobotics/PRISM.git && cd PRISM
python -m pip install -r requirements.txt
python -m pip install -e setup.py
```



Here are the llm-enabled planners supported:
- [SPINE](https://github.com/KumarRobotics/SPINE/tree/feature/prism) (PRISM branch)
- [LLM-Planner](https://github.com/ZacRavichandran/LLM-Planner/tree/main/e2e) (for ALFRED experiments)
- SayCan (with a local implementation)

If collecting data, make sure the relevant planners are installed.

## Generating data

Data generation will
1. Synthesize data using GPT, given some task and graph structure
2. Generate a plan using a GPT-enabled planner
3. Save all generated data

Data generation will save intermediate samples, and the aggregated data will be saved in a file called `formatted.json`. This can be used for unsloth training.

Basic parameters
- `n-samples` number of training samples to generate
- `n-tasks` number of tasks per sample
- `name` name of directory for data logging


### for SPINE

```
python scripts/generate_data_spine.py --help
   -h, --help
  --n-samples N_SAMPLES
  --n-tasks N_TASKS
  --name NAME

```

### for ALFRED

```
python scripts/generate_data_alfred.py --help
  -h, --help            show this help message and exit
  --n-samples N_SAMPLES
  --n-tasks N_TASKS
  --n-objects N_OBJECTS
  --name NAME
```


Other parameters
- `n-objects` number of objects per task

## Training a model

```
usage: train_llama3_llora.py [-h] [--name NAME] [--bit4 [BIT4]] [--data DATA] [--eval_data EVAL_DATA] [--r R]
                             [--base_model BASE_MODEL] [--wandb_project WANDB_PROJECT] [--epochs EPOCHS] [--val_frac VAL_FRAC]
                             [--lora_alpha LORA_ALPHA] [--lora_dropout LORA_DROPOUT]
                             [--target_modules TARGET_MODULES [TARGET_MODULES ...]]
                             [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
                             [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
                             [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--report_to REPORT_TO]
                             [--learning_rate LEARNING_RATE] [--warmup_steps WARMUP_STEPS] [--weight_decay WEIGHT_DECAY]
                             [--debug [DEBUG]]

options:
  -h, --help            show this help message and exit
  --name NAME
  --bit4 [BIT4]
  --data DATA
  --eval_data EVAL_DATA, --eval-data EVAL_DATA
  --r R
  --base_model BASE_MODEL, --base-model BASE_MODEL
  --wandb_project WANDB_PROJECT, --wandb-project WANDB_PROJECT
  --epochs EPOCHS
  --val_frac VAL_FRAC, --val-frac VAL_FRAC
  --lora_alpha LORA_ALPHA, --lora-alpha LORA_ALPHA
  --lora_dropout LORA_DROPOUT, --lora-dropout LORA_DROPOUT
  --target_modules TARGET_MODULES [TARGET_MODULES ...], --target-modules TARGET_MODULES [TARGET_MODULES ...]
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE, --per-device-train-batch-size PER_DEVICE_TRAIN_BATCH_SIZE
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE, --per-device-eval-batch-size PER_DEVICE_EVAL_BATCH_SIZE
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS, --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
  --report_to REPORT_TO, --report-to REPORT_TO
  --learning_rate LEARNING_RATE, --learning-rate LEARNING_RATE
  --warmup_steps WARMUP_STEPS, --warmup-steps WARMUP_STEPS
  --weight_decay WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
  --debug [DEBUG]

```


## Code style

- This project uses `black`, `isort`, and `flake8` for enforcing code style. See `requirements.txt` for version numbers.
- We use `pre-commit` hooks to ensure that all code committed respects the code style.
- After (1) cloning the repo, (2) creating your environment and (3) installing the required
packages, you are strongly encouraged to run `pre-commit install` to set-up pre-commit hooks.


## Referencing

If you find this work helpful, please cite
```
@article{ravichandran_prism,
      title={Distilling On-device Language Models for Robot Planning with Minimal Human Intervention},
      author={Zachary Ravichandran and Ignacio Hounie and Fernando Cladera and Alejandro Ribeiro and George J. Pappas and Vijay Kumar},
      year={2025},
      journal={Conference on Robot Learning (CoRL)},
      url={https://arxiv.org/abs/2506.17486},
}

```
