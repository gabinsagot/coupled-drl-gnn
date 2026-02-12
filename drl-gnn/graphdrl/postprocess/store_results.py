import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm


def merge_case_results(case_path, output_format):
    case_path = os.path.abspath(case_path)
    case_name = os.path.basename(case_path.rstrip("/"))

    merged_rows = []

    # run folders are directories whose name is an integer
    run_folders = [
        d
        for d in os.listdir(case_path)
        if os.path.isdir(os.path.join(case_path, d)) and d.isdigit()
    ]

    if not run_folders:
        print("No valid integer-named run folders found.")
        return

    # sort runs
    run_folders = sorted(run_folders, key=lambda x: int(x))

    print(f"Detected {len(run_folders)} run folders.")
    print("Scanning...")

    # run progress bar
    for run in tqdm(run_folders, desc="Processing runs", unit="run"):

        run_path = os.path.join(case_path, run)

        # envs are folders
        env_folders = [
            d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))
        ]
        env_folders = sorted(env_folders, key=lambda x: int(x))

        total_envs = len(env_folders)
        env_count = 0

        with tqdm(
            total=total_envs,
            desc=f"Process envs of run {run} ",
            unit="env",
            leave=False,
        ) as pbar:

            for env in env_folders:
                env_count += 1

                # update progress bar postfix
                pbar.set_postfix(env=f"{env_count}/{total_envs}")

                env_path = os.path.join(run_path, env)

                # data is in CSV files, discard other files
                csv_files = [
                    f for f in os.listdir(env_path) if f.lower().endswith(".csv")
                ]

                for fname in csv_files:
                    csv_path = os.path.join(env_path, fname)

                    try:
                        df = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"[Skipped] Could not read {csv_path}: {e}")
                        continue
                    df["case_id"] = case_name
                    df["run_id"] = run
                    df["env_id"] = env

                    merged_rows.append(df)

                pbar.update(1)

    if not merged_rows:
        print("No CSV data found in env folders.")
        return

    final_df = pd.concat(merged_rows, ignore_index=True)

    # reorder columns
    cols = ["case_id", "run_id", "env_id"] + [
        c for c in final_df.columns if c not in ("case_id", "run_id", "env_id")
    ]
    final_df = final_df[cols]

    # rename columns for convention
    final_df.rename(
        columns={"Time": "time", "Fx": "fx", "Fy": "fy", "Object": "object"},
        inplace=True,
    )

    # output path
    print("Writing output...")
    output_dir = os.path.dirname(case_path)
    output_file = os.path.join(output_dir, f"{case_name}.{output_format}")

    # safety overwrite check
    if os.path.exists(output_file):
        print(f"ERROR: Output file already exists: {output_file}")
        print("Refusing to overwrite. Please delete the file manually.")
        return

    # write to format
    if output_format == "csv":
        final_df.to_csv(output_file, index=False)
    elif output_format == "parquet":
        final_df.to_parquet(output_file, index=False)
    else:
        print(f"Unknown output format: {output_format}")
        return

    print(f"\nSuccess! Output written to: {output_file}")


def parser():
    parser = argparse.ArgumentParser(
        description=(
            "Process the results of a PBO optimization case by merging run "
            "and environment data into a single dataset for simplicity and storage."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "case_path",
        nargs="+",
        help="One or more paths to case_name directories containing numerical run folders.",
    )
    parser.add_argument(
        "--output_format",
        choices=["csv", "parquet"],
        default="csv",
        help="Choose output format: 'csv' or 'parquet'.",
    )
    args = parser.parse_args()

    # handle multiple paths
    if len(args.case_path) > 1:
        for path in args.case_path:
            merge_case_results(case_path=path, output_format=args.output_format)
        sys.exit(0)

    args.case_path = args.case_path[0]
    return args


def main():
    args = parser()
    merge_case_results(case_path=args.case_path, output_format=args.output_format)


if __name__ == "__main__":
    main()
