import wandb
from transformers.trainer_callback import TrainerCallback

from prism.eval.run_eval import eval_model


class MetricsTrackerCallback(TrainerCallback):
    """Custom callback that captures metrics for retrieval"""

    def __init__(self, trainer):
        self.trainer = trainer
        # Flag this as our custom callback
        self._is_metrics_tracker = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Capture metrics during evaluation"""
        if metrics:
            self.trainer.latest_metrics.update(metrics)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture metrics during logging"""
        if logs:
            self.trainer.latest_metrics.update(logs)


class EvalCallback(TrainerCallback):
    def __init__(self, eval_samples):
        self.eval_samples = eval_samples

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            print("Model not provided; skipping evaluation.")
            return control

        # Save current model state to a temporary directory
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            # Run evaluation
            result_correct = eval_model(
                model_path=tmp_dir,
                is_four_bit=False,  # Use same settings as training
                eval_samples=self.eval_samples,
            )

            # Log to wandb
            wandb.log({"eval/accuracy": result_correct, "epoch": state.epoch})
            self.metrics = {
                "eval/accuracy": result_correct,
            }

        return control

    def get_latest_metrics(self):
        return self.metrics


"""'
class EvalCallback(TrainerCallback):
    def __init__(self, eval_samples):
        self.eval_samples = eval_samples

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            print("Model not provided; skipping evaluation.")
            return control

        # Save current model state to a temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            # Run evaluation
            result_correct = eval_model(
                model_path=tmp_dir,
                is_four_bit=False,  # Use same settings as training
                eval_samples=self.eval_samples
            )

            # Log to wandb
            wandb.log({
                "eval/accuracy": result_correct,
                "epoch": state.epoch
            })
            if result_correct > 0.9:
                model.save_pretrained("lora_gpt_r8")

        return control
"""
