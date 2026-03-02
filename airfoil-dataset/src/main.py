import argparse
import time
import ast

from dataset import Dataset


def parser_dataset():
    parser = argparse.ArgumentParser(
        description="Generate dataset based on config file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_config_file",
        type=str,
        default="./config/airfoil.json",
        help="Path to the meta .json config file",
    )
    parser.add_argument(
        "--create_configs_pool",
        type=ast.literal_eval,
        default=True,
        help="Create a new pool of configurations",
    )
    parser.add_argument(
        "-z",
        "--zip",
        action="store_true",
        help=(
            "Zip and delete the dataset after creation. "
            "Zip name taken from pool name, does not overwrite existing zips. "
            "(default: False)"
        ),
    )
    parser.add_argument(
        "--ignore_slurm",
        action="store_true",
        help=(
            "If set, always use mpirun instead of srun, even if SLURM is available. "
            "(default: False)"
        ),
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parser_dataset()
    # timer
    start_time = time.time()
    # create dataset
    dataset = Dataset(
        meta_path=args.dataset_config_file,
        create_configs_pool=args.create_configs_pool,
        ignore_slurm=args.ignore_slurm,
    )
    dataset.prep_directories(clear=False)
    dataset.print_info()
    dataset.generate()
    # timer
    elapsed_time = time.time() - start_time
    days = elapsed_time // (24 * 3600)
    elapsed_time = elapsed_time % (24 * 3600)
    hours = elapsed_time // 3600
    elapsed_time %= 3600
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    print(
        f"\n\nTotal walltime: {int(days)}-{int(hours):02}:{int(minutes):02}:{seconds:.2f}"
    )
    if args.zip:
        dataset.zip()


if __name__ == "__main__":
    main()
