from typing import List

from transformers import DataCollatorForSeq2Seq
from unsloth_zoo.training_utils import fix_zero_training_loss

__all__ = [
    "train_on_responses_only",
    "TurnAwareCollator",
    "get_formatting_prompts_func",
    "retrieve_latest_metrics_from_wandb",
]


# Custom collator that preserves the turn field during batching
class TurnAwareCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer=None,
        model=None,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        label_pad_token_id=-100,
        return_tensors="pt",
        include_turns=True,
        turn_pad_value=-100,
    ):
        super().__init__(
            tokenizer,
            model,
            padding,
            max_length,
            pad_to_multiple_of,
            label_pad_token_id,
            return_tensors,
        )
        self.include_turns = include_turns
        self.turn_pad_value = turn_pad_value

    def __call__(self, features):
        # Store original turn values before super() processing
        original_turns = []
        if self.include_turns:
            for feature in features:
                if "turn" in feature:
                    # Store the complete turn array without any filtering
                    original_turns.append(feature.pop("turn"))
                else:
                    original_turns.append([])

        # Call parent collator to handle padding, etc.
        batch = super().__call__(features)

        # Add turn field back to the batch with matching padding
        if self.include_turns and original_turns:
            # Get the length of input_ids after padding
            input_ids_lengths = [len(ids) for ids in batch["labels"]]

            # Pad turns to match input_ids length
            padded_turns = []
            for i, turns in enumerate(original_turns):
                # Pad or truncate to match input_ids length
                target_length = input_ids_lengths[i]
                if len(turns) < target_length:
                    padded_turns.append(
                        turns + [self.turn_pad_value] * (target_length - len(turns))
                    )
                else:
                    padded_turns.append(turns[:target_length])

            # Convert to tensor if needed
            if self.return_tensors == "pt":
                import torch

                batch["turn"] = torch.tensor(padded_turns)
            else:
                batch["turn"] = padded_turns

        return batch


def _longest_common_sublist(lists):
    """
    Finds the longest common sublist among multiple lists.

    Parameters:
    lists (List[List[int]]): A list of lists.

    Returns:
    List[int]: The longest common sublist. If multiple sublists have the same maximum length,
               one of them is returned. If there's no common sublist, an empty list is returned.
    """
    if not lists:
        return []

    # Find the minimum length among all lists
    min_len = min(len(lst) for lst in lists)
    if min_len == 0:
        return []

    def has_common_sublist(length):
        """
        Checks if there's a common sublist of the given length across all lists.

        Returns:
        (bool, List): Tuple of whether such a sublist exists and the sublist itself.
        """
        common = set()
        first = lists[0]
        # Generate all possible sublists of the given length from the first list
        for i in range(len(first) - length + 1):
            sub = tuple(first[i : i + length])
            common.add(sub)
        pass

        # Iterate over the remaining lists and retain only the common sublists
        for lst in lists[1:]:
            current = set()
            for i in range(len(lst) - length + 1):
                sub = tuple(lst[i : i + length])
                if sub in common:
                    current.add(sub)
            common = current
            if not common:
                return False, []
        pass

        # If common is not empty, return one of the common sublists
        return True, list(common.pop())

    pass

    left, right = 1, min_len
    result = []

    while left <= right:
        mid = left + (right - left) // 2
        exists, sublist = has_common_sublist(mid)
        if exists:
            result = sublist  # Update result with the latest found sublist
            left = mid + 1  # Try to find a longer sublist
        else:
            right = mid - 1  # Try with a shorter length
    pass

    return result


def _find_common_token_ids(component, tokenizer):
    """
    \n### User:\n\n
    \n\n### User:\n\n
    etc
    we need to find the middle most repeatted part.
    Tokenizers can tokenize newlines or spaces as 1 token!
    """
    right_text = ""
    if component.endswith(" "):
        right_text = " "
    elif component.endswith("\n"):
        right_text = "\n"
    left_text = ""
    if component.startswith(" "):
        left_text = " "
    elif component.startswith("\n"):
        left_text = "\n"
    stripped = component.strip()

    # Add current pieces and also newlines
    all_input_ids = []
    for left in range(3):
        for right in range(3):
            x = left * left_text + stripped + right * right_text
            x = tokenizer(x, add_special_tokens=False).input_ids
            all_input_ids.append(x)

            x = left * "\n" + stripped + right * "\n"
            x = tokenizer(x, add_special_tokens=False).input_ids
            all_input_ids.append(x)
        pass
    pass

    # Old longest common substring is replaced with actual longest common list of numbers
    # substring = _old_longest_common_substring([str(x + [0]) for x in all_input_ids])
    # substring = substring.split(", ")[:-1]
    # substring = [int(x) for x in substring if x.isdigit()]
    substring = _longest_common_sublist([x + [0] for x in all_input_ids])

    # If substring is simply [0], this might be just the original single token
    # Fixes https://github.com/unslothai/unsloth/issues/1290
    # Mistral [INST] [/INST] singular tokens breaks since we output [0] but we need [3] [4]
    if substring == [0] and len(all_input_ids[0]) == 1:
        single_token = all_input_ids[0][0]
        # Confirm single token in every single possible match
        if all(single_token in x for x in all_input_ids):
            substring = [single_token]
    pass

    # Also if substring is original input_ids + [0], then leave it as the original one
    # This happens when no newlines / spaces are used in chat template
    # Eg Phi-4 does not use newlines or spaces
    if (
        (len(set(str(x) for x in all_input_ids)) == 1)
        and (len(all_input_ids[0]) + 1 == len(substring))
        and (all_input_ids[0] == substring[:-1])
    ):
        # Use original un-changed substring
        substring = all_input_ids[0]
    pass

    # Also get rest of tokenized string
    original = tokenizer(component, add_special_tokens=False).input_ids
    # Get optional left and right
    for j in range(len(original)):
        if original[j : j + len(substring)] == substring:
            break
    optional_left = original[:j]
    optional_right = original[j + len(substring) :]
    return substring, optional_left, optional_right


pass


def train_on_responses_only(
    trainer,
    instruction_part=None,
    response_part=None,
):
    """
    Trains only on responses and not on the instruction by masking out
    the labels with -100 for the instruction part.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    tokenizer = (
        trainer.processing_class
        if hasattr(trainer, "processing_class")
        else trainer.tokenizer
    )

    if not hasattr(tokenizer, "_unsloth_input_part") or not hasattr(
        tokenizer, "_unsloth_output_part"
    ):
        if instruction_part is None or response_part is None:
            raise ValueError(
                "Unsloth: instruction_part and response_part must be given!"
            )
    elif (instruction_part is not None or response_part is not None) and (
        hasattr(tokenizer, "_unsloth_input_part")
        or hasattr(tokenizer, "_unsloth_output_part")
    ):
        raise ValueError(
            "Unsloth: Your tokenizer already has instruction and response parts set - do not give custom ones!"
        )
    else:
        instruction_part = tokenizer._unsloth_input_part
        response_part = tokenizer._unsloth_output_part
    pass

    # Get most common tokens since tokenizers can tokenize stuff differently!
    Q_must, Q_left, Q_right = _find_common_token_ids(instruction_part, tokenizer)
    A_must, A_left, A_right = _find_common_token_ids(response_part, tokenizer)

    # Store some temporary stuff
    A_first = A_must[0]
    len_A_must = len(A_must)
    A_left_reversed = A_left[::-1]
    A_right_forward = A_right

    Q_first = Q_must[0]
    len_Q_must = len(Q_must)
    Q_left_reversed = Q_left[::-1]
    Q_right_forward = Q_right

    def _train_on_responses_only(examples):
        input_ids_ = examples["input_ids"]
        all_labels = []
        all_turns = []  # New: collect turn indices per sample.
        for input_ids in input_ids_:
            n = len(input_ids)
            n_minus_1 = n - 1
            labels = [-100] * n  # Initialize all turn values to -100.
            turns = [-100] * n
            turn_counter = 0  # Counter for assistant turns.
            j = 0
            while j < n:
                # Find <assistant> marker.
                if (input_ids[j] == A_first) and (
                    input_ids[j : (k := j + len_A_must)] == A_must
                ):
                    # Backtrack for optional left tokens.
                    for optional_left in A_left_reversed:
                        if j < 1:
                            break
                        if optional_left == input_ids[j - 1]:
                            j -= 1
                        else:
                            break
                    # Extend forwards for optional right tokens.
                    for optional_right in A_right_forward:
                        if k >= n_minus_1:
                            break
                        if optional_right == input_ids[k + 1]:
                            k += 1
                        else:
                            break
                    assistant_k = k  # Mark the start of the assistant response.
                    j = assistant_k
                    while j < n:
                        # Find <user> marker or end of sample.
                        if (j == n_minus_1) or (
                            (input_ids[j] == Q_first)
                            and (input_ids[j : (k := j + len_Q_must)] == Q_must)
                        ):
                            # Backtrack for optional left tokens.
                            for optional_left in Q_left_reversed:
                                if j < 1:
                                    break
                                if optional_left == input_ids[j - 1]:
                                    j -= 1
                                else:
                                    break
                            # Extend forwards for optional right tokens.
                            for optional_right in Q_right_forward:
                                if k >= n_minus_1:
                                    break
                                if optional_right == input_ids[k + 1]:
                                    k += 1
                                else:
                                    break
                            user_j = j if j != n_minus_1 else n
                            # Copy assistant response tokens to labels.
                            labels[assistant_k:user_j] = input_ids[assistant_k:user_j]
                            turns[assistant_k:user_j] = [turn_counter] * (
                                user_j - assistant_k
                            )
                            # Set the corresponding turn indices for these tokens.
                            turn_counter += 1
                            break
                        j += 1
                j += 1
            all_labels.append(labels)
            all_turns.append(turns)
            # breakpoint()
        # breakpoint()
        return {"labels": all_labels, "turn": all_turns}

    pass

    if hasattr(trainer, "train_dataset") and trainer.train_dataset is not None:
        trainer.train_dataset = trainer.train_dataset.map(
            _train_on_responses_only, batched=True
        )
    pass

    if hasattr(trainer, "eval_dataset") and trainer.eval_dataset is not None:
        # Eval datasets could be a dict!
        if type(trainer.eval_dataset) is dict:
            for key, value in trainer.eval_dataset.items():
                trainer.eval_dataset[key] = value.map(
                    _train_on_responses_only, batched=True
                )
        else:
            trainer.eval_dataset = trainer.eval_dataset.map(
                _train_on_responses_only, batched=True
            )
        pass
    pass

    fix_zero_training_loss(None, tokenizer, trainer.train_dataset)
    return trainer


def get_formatting_prompts_func(tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    return formatting_prompts_func


def retrieve_latest_metrics_from_wandb(current_run):
    # Retrieve training metrics from wandb
    # Get all metrics from the run
    metrics = {}
    for key in current_run.history().keys():
        metrics[key] = current_run.history()[key].tolist()

    # Get the final evaluation accuracy
    final_eval_accuracy = None
    if "eval/accuracy" in metrics:
        final_eval_accuracy = metrics["eval/accuracy"][-1]

    # Print summary metrics
    print("\nTraining Summary:")
    print(f"Final evaluation accuracy: {final_eval_accuracy:.4f}")

    # Return metrics for further use if needed
    return metrics
