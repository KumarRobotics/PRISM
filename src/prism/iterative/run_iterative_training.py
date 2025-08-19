#!/usr/bin/env python
"""Run the iterative training pipeline."""
from transformers import HfArgumentParser

from prism.iterative.iterative_training import (IterativePipelineConfig,
                                                run_iterative_training)
from prism.training.train import TrainConfig

if __name__ == "__main__":
    parser = HfArgumentParser((IterativePipelineConfig, TrainConfig))
    pipeline_args, train_args = parser.parse_args_into_dataclasses()
    run_iterative_training(pipeline_args, train_args)
