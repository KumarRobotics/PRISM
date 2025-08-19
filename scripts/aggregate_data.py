import argparse
import json
from pathlib import Path

import numpy as np

root = Path.home()


def try_load_json(file):
    with open(file) as f:
        content = f.read()

    # Not sure why but some files have trailing strings
    # after the json entry
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return json.loads("".join(content.split("\n")[:-4]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, help="data dir", default=root)
    parser.add_argument("--rnd-pct", type=int, default=100)
    parser.add_argument("--out", default="formatted-non-iterative.json")
    args = parser.parse_args()

    data_root = args.data_root
    all_data = []
    n_failed = 0

    for data_root in root:
        json_files = list(Path(data_root).glob("sample*json"))
        print(len(json_files))

        np.random.shuffle(json_files)

        n_to_save = len(json_files) * args.rnd_pct // 100

        for idx, json_file in enumerate(json_files):
            if "failed" in str(json_file):
                n_failed += 1
                continue
            try:
                data = try_load_json(json_file)

                debug = 0

                train_data = data[36:]

                all_data.append({"conversations": train_data})

                if idx >= n_to_save:
                    break

            except Exception as ex:
                print(json_file)

    print(f"n failed: {n_failed}")

    print(len(all_data))

    print(f"saving {len(all_data)} samples")

    with open(args.out, "w") as f:
        json.dump(all_data, f)
