from dataclasses import dataclass

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


@dataclass
class MergeConfig:
    base_model_path: str = "unsloth/Llama-3.2-3B-Instruct"
    lora_path: str
    merged_model_path: str


if __name__ == "__main__":
    parser = HfArgumentParser(MergeConfig)
    (config,) = parser.parse_args_into_dataclasses()

    # Load base model (make sure it's the same as what was fine-tuned with Unsloth)
    base_model_path = config.base_mode_path
    lora_path = config.lora_path
    merged_model_path = config.merged_model_path

    model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load LoRA adapter (Unsloth saved it here)
    model = PeftModel.from_pretrained(model, lora_path)

    # Merge LoRA with base model
    model = model.merge_and_unload()

    # Save full model
    model.save_pretrained(merged_model_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_path)

    print(f"Full model saved at: {merged_model_path}")
