import json
from pathlib import Path
from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion


def try_load_json(file):
    with open(file) as f:
        content = f.read()

    # Not sure why but some files have trailing strings
    # after the json entry
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return json.loads("".join(content.split("\n")[:-4]))


def aggregate(
    root_dir: str, glob_str: str, out_file: str, cutbefore: Optional[int] = 36
) -> None:
    json_files = Path(root_dir).glob(glob_str)

    all_data = []
    for json_file in json_files:
        try:
            data = try_load_json(json_file)

            train_data = data[cutbefore:]

            all_data.append({"conversations": train_data})
        except Exception as ex:
            print(json_file)

    with open(out_file, "w") as f:
        json.dump(all_data, f)


class GPTQueryClient:
    def __init__(self):
        self.client = OpenAI()

    def query_gpt(
        self,
        query: str,
        temperature: Optional[float] = 0.31,
        max_tokens: Optional[int] = 2048,
    ) -> ChatCompletion:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}],
                }
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        return response
