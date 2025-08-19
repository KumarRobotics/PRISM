from typing import Optional, Tuple

from transformers import PreTrainedTokenizer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
FOURBIT_MODELS = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
    "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",  # NEW! Llama 3.3 70B!
]  # More models at https://huggingface.co/unsloth

FULL_MODELS = ["unsloth/Llama-3.2-3B-Instruct"]


def from_pretrained(
    path: str,
    max_seq_length: Optional[int] = 2048,
    load_in_4bit: Optional[bool] = True,
    inference: Optional[bool] = False,
) -> Tuple[FastLanguageModel, PreTrainedTokenizer]:
    """Load a model from unsloth.

    Parameters
    ----------
    path : str
        Model path. Can be local or huggingface
    max_seq_length : Optional[int], optional
        For LLM generation, by default 2048
    load_in_4bit : Optional[bool], optional
        Use 4 bit quantized model, by default True
    inference : Optional[bool], optional
        Load inference model, by default False

    Returns
    -------
    Tuple[FastLanguageModel, PreTrainedTokenizer]
        Model and tokenizer
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        # dtype = dtype,
        load_in_4bit=load_in_4bit,
    )
    if inference:
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    return model, tokenizer
