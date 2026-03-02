import os
import argparse
import random
import shutil
import sys


def process_files(
    start_index: int,
    count: int,
    target_dir: str,
    shuffled_files: list,
    input_dir: str,
    copy: bool = False,
) -> None:
    for i in range(start_index, start_index + count):
        if i >= len(shuffled_files):
            break
        base_name = os.path.splitext(shuffled_files[i])[0]

        xdmf_file = os.path.join(input_dir, f"{base_name}.xdmf")
        h5_file = os.path.join(input_dir, f"{base_name}.h5")

        if os.path.isfile(xdmf_file) and os.path.isfile(h5_file):
            target_xdmf = os.path.join(target_dir, os.path.basename(xdmf_file))
            target_h5 = os.path.join(target_dir, os.path.basename(h5_file))

            if os.path.exists(target_xdmf) or os.path.exists(target_h5):
                print(
                    (
                        f"ERROR: Files {xdmf_file} or {h5_file} already exist in {target_dir}. "
                        "Aborting to prevent data loss."
                    )
                )
                sys.exit(1)

            if copy:
                shutil.copy(xdmf_file, target_xdmf)
                shutil.copy(h5_file, target_h5)
            else:
                shutil.move(xdmf_file, target_xdmf)
                shutil.move(h5_file, target_h5)
        else:
            print(f"WARNING: Skipping {base_name} because the file pair is incomplete.")


def _parser():
    """Parse command-line arguments for splitting xdmf/h5 dataset into train/test/predict."""
    parser = argparse.ArgumentParser(
        description="Split dataset into train, test, and predict directories."
    )
    parser.add_argument(
        "input_dir", help="Directory containing the .xdmf and .h5 files."
    )
    parser.add_argument(
        "-train",
        "--train_dir",
        default="train",
        help="Directory name for training data. (default: ./train)",
    )
    parser.add_argument(
        "-test",
        "--test_dir",
        default="test",
        help="Directory name for testing/validation data. (default: ./test)",
    )
    parser.add_argument(
        "-predict",
        "--predict_dir",
        default="predict",
        help="Directory name for prediction data. (default: ./predict)",
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of moving them."
    )
    parser.add_argument(
        "--reproduce",
        default=None,
        type=str,
        help=(
            "Reproduce a dataset split from a given directory "
            "(must include train/test/predict subdirectories). "
            "This will look for the same configuration IDs in the given directory "
            "and current directory, split the current directory in the same way. "
            "Non-matching IDs will be ignored.split to respect train/loss/split."
        ),
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=str,
        default="80,10,10",
        help=(
            "Train, test, predict split ratio as a comma-separated string "
            "of three integers (e.g., '80,10,10'). Must add up to 100."
        ),
    )
    parser.add_argument(
        "--sorted",
        action="store_true",
        help="Sort files instead of shuffling before splitting.",
    )
    args = parser.parse_args()
    args.ratio = list(map(lambda x: int(float(x) + 0.5), args.ratio.split(",")))
    return args


def main_split(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = _parser()

    input_dir = args.input_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    predict_dir = args.predict_dir
    copy = args.copy
    ratio = args.ratio
    reproduce_dir = args.reproduce

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory {input_dir} does not exist.")
        sys.exit(1)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".xdmf")])

    if not files:
        print("No .xdmf files found in the input directory. Exiting.")
        sys.exit(1)
    if reproduce_dir is None:
        total_files = len(files)
        train_count = total_files * ratio[0] // 100
        test_count = total_files * ratio[1] // 100
        predict_count = total_files - train_count - test_count

        print(f"Total files: {total_files}")
        print(f"Train: {train_count}, Test: {test_count}, Predict: {predict_count}")

        shuffled_files = files[:]
        if not args.sorted:
            random.shuffle(shuffled_files)

        process_files(0, train_count, train_dir, shuffled_files, input_dir, copy)
        process_files(
            train_count, test_count, test_dir, shuffled_files, input_dir, copy
        )
        process_files(
            train_count + test_count,
            predict_count,
            predict_dir,
            shuffled_files,
            input_dir,
            copy,
        )

        print("Files successfully processed into train, test, and predict directories.")

    else:
        reproduce_dir = args.reproduce
        reproduced_files = []
        # sanity check
        if not os.path.isdir(reproduce_dir):
            print(f"ERROR: Replicate directory {reproduce_dir} does not exist.")
            sys.exit(1)
        reproduce_train_dir = os.path.join(reproduce_dir, "train")
        reproduce_test_dir = os.path.join(reproduce_dir, "test")
        reproduce_predict_dir = os.path.join(reproduce_dir, "predict")
        if not os.path.isdir(reproduce_train_dir):
            print(
                f"ERROR: Replicate train directory {reproduce_train_dir} does not exist."
            )
            sys.exit(1)
        if not os.path.isdir(reproduce_test_dir):
            print(
                f"ERROR: Replicate test directory {reproduce_test_dir} does not exist."
            )
            sys.exit(1)
        if not os.path.isdir(reproduce_predict_dir):
            print(
                f"ERROR: Replicate predict directory {reproduce_predict_dir} does not exist."
            )
            sys.exit(1)

        # reproduce train split
        reproduce_train_files = sorted(
            [
                f
                for f in os.listdir(reproduce_train_dir)
                if f.endswith(".xdmf") or f.endswith(".h5")
            ]
        )
        for file in reproduce_train_files:
            if file.endswith(".xdmf"):
                base_name = os.path.splitext(file)[0]
                xdmf_file = os.path.join(input_dir, f"{base_name}.xdmf")
                h5_file = os.path.join(input_dir, f"{base_name}.h5")

                if os.path.isfile(xdmf_file) and os.path.isfile(h5_file):
                    target_xdmf = os.path.join(train_dir, os.path.basename(xdmf_file))
                    target_h5 = os.path.join(train_dir, os.path.basename(h5_file))
                    if copy:
                        shutil.copy(xdmf_file, target_xdmf)
                        shutil.copy(h5_file, target_h5)
                        reproduced_files.append(os.path.basename(xdmf_file))
                    else:
                        shutil.move(xdmf_file, target_xdmf)
                        shutil.move(h5_file, target_h5)
                        reproduced_files.append(os.path.basename(xdmf_file))
                else:
                    print(
                        f"\tWARNING: Skipping train/{base_name} (absent in current dataset)."
                    )
        # reproduce test split
        reproduce_test_files = sorted(
            [
                f
                for f in os.listdir(reproduce_test_dir)
                if f.endswith(".xdmf") or f.endswith(".h5")
            ]
        )
        for file in reproduce_test_files:
            if file.endswith(".xdmf"):
                base_name = os.path.splitext(file)[0]
                xdmf_file = os.path.join(input_dir, f"{base_name}.xdmf")
                h5_file = os.path.join(input_dir, f"{base_name}.h5")

                if os.path.isfile(xdmf_file) and os.path.isfile(h5_file):
                    target_xdmf = os.path.join(test_dir, os.path.basename(xdmf_file))
                    target_h5 = os.path.join(test_dir, os.path.basename(h5_file))
                    if copy:
                        shutil.copy(xdmf_file, target_xdmf)
                        shutil.copy(h5_file, target_h5)
                        reproduced_files.append(os.path.basename(xdmf_file))
                    else:
                        shutil.move(xdmf_file, target_xdmf)
                        shutil.move(h5_file, target_h5)
                        reproduced_files.append(os.path.basename(xdmf_file))
                else:
                    print(
                        f"\tWARNING: Skipping test/{base_name} (absent in current dataset)."
                    )
        # reproduce predict split
        reproduce_predict_files = sorted(
            [
                f
                for f in os.listdir(reproduce_predict_dir)
                if f.endswith(".xdmf") or f.endswith(".h5")
            ]
        )
        for file in reproduce_predict_files:
            if file.endswith(".xdmf"):
                base_name = os.path.splitext(file)[0]
                xdmf_file = os.path.join(input_dir, f"{base_name}.xdmf")
                h5_file = os.path.join(input_dir, f"{base_name}.h5")

                if os.path.isfile(xdmf_file) and os.path.isfile(h5_file):
                    target_xdmf = os.path.join(predict_dir, os.path.basename(xdmf_file))
                    target_h5 = os.path.join(predict_dir, os.path.basename(h5_file))
                    if copy:
                        shutil.copy(xdmf_file, target_xdmf)
                        shutil.copy(h5_file, target_h5)
                        reproduced_files.append(os.path.basename(xdmf_file))
                    else:
                        shutil.move(xdmf_file, target_xdmf)
                        shutil.move(h5_file, target_h5)
                        reproduced_files.append(os.path.basename(xdmf_file))
                else:
                    print(
                        f"\tWARNING: Skipping predict/{base_name} (absent in current dataset)."
                    )
        # print summary
        reproduced_files = sorted(reproduced_files)
        reproduced_count = len(reproduced_files)
        print(f"\nReproduced files: {reproduced_count}")
        print(
            f"Train: {len(os.listdir(train_dir)) // 2}, "
            f"Test: {len(os.listdir(test_dir)) // 2}, "
            f"Predict: {len(os.listdir(predict_dir)) // 2}."
        )

        # moving/copying remaining files to extra/
        remaining_files = sorted([f for f in files if f not in reproduced_files])
        remaining_count = len(remaining_files)
        if remaining_count > 0:
            extra_dir = os.path.join(os.path.dirname(train_dir), "extra")
            os.makedirs(extra_dir, exist_ok=True)
            if copy:
                for file in remaining_files:
                    base_name = os.path.splitext(file)[0]
                    xdmf_file = os.path.join(input_dir, f"{base_name}.xdmf")
                    h5_file = os.path.join(input_dir, f"{base_name}.h5")
                    shutil.copy(xdmf_file, extra_dir)
                    shutil.copy(h5_file, extra_dir)
            else:
                for file in remaining_files:
                    base_name = os.path.splitext(file)[0]
                    xdmf_file = os.path.join(input_dir, f"{base_name}.xdmf")
                    h5_file = os.path.join(input_dir, f"{base_name}.h5")
                    shutil.move(xdmf_file, extra_dir)
                    shutil.move(h5_file, extra_dir)
            print(f"\nRemaining files: {remaining_count}")
            print(
                "Remaining files will not be split, find them in 'extra/'. "
                "\n\t-> Please copy them manually to train/test/predict directories."
            )
        else:
            print("No remaining files to process.")


if __name__ == "__main__":
    main_split(args=None)
