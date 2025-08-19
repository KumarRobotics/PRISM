import argparse
import json
from pathlib import Path

from prism.data.data_gen import DataGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--n-tasks", type=int, default=25)
    parser.add_argument("--name", type=str, default="non-iterative-data")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.31)

    args = parser.parse_args()

    Path(args.name).mkdir(parents=True, exist_ok=True)
    with open(f"{args.name}/data_gen_params.json", "w") as f:
        json.dump(vars(args), f)

    log_dir = args.name

    # unknown_pcts = [0, 5] * 10  # [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    unknown_pcts = [0] * 10
    # unknown_pcts = [0, 5, 10, 15] * 10
    n_regions_list = [10, 15, 20] * 10  # np.arange(20, 30, 2)
    n_objects_list = [3, 6, 9] * 10  # np.arange(10, 30, 1)

    # unknown_pcts = [10, 15] * 10
    # n_regions_list = [10, 15, 20] * 10  # np.arange(20, 30, 2)
    # n_objects_list = [3, 6, 9] * 10  # np.arange(10, 30, 1)

    data_generator = DataGenerator(
        graph_unknown=unknown_pcts,
        n_region_list=n_regions_list,
        n_objects_list=n_objects_list,
    )
    data_generator.generate(
        log_dir=log_dir,
        n_samples=args.n_samples,
        n_tasks=args.n_tasks,
        description=args.description,
    )
