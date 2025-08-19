import argparse

from prism.data.alfred.alfred_gen_data import AlfredDataGen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-samples", type=int, default=25)
    parser.add_argument("--n-tasks", type=int, default=4)
    parser.add_argument("--n-objects", type=int, default=5)

    parser.add_argument("--name", type=str, default="fully-known-graph-v1")

    args = parser.parse_args()

    alfred_data_gen = AlfredDataGen(log_dir=args.name)
    alfred_data_gen.generate(
        n_samples=args.n_samples, n_tasks=args.n_tasks, n_objects=args.n_objects
    )
