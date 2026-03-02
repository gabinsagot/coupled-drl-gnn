import os
import re
import numpy as np
from pathlib import Path
import argparse


def _parser():
    """Parse command line arguments for mesh statistics."""
    parser = argparse.ArgumentParser(
        description="Analyze a dataset's XDMF files for retrieving mesh (num of nodes) statistics."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help=(
            "Directory of dataset to search for .xdmf files. "
            "Will search into dataset splits (train/val/test) recursively, if any."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="If set, prints detailed processing information.",
    )
    args = parser.parse_args()
    return args


def extract_node_count(xdmf_file):
    """
    Extract the number of nodes from an XDMF file.

    Args:
        xdmf_file (str): Path to the XDMF file

    Returns:
        int: Number of nodes, or None if not found
    """
    try:
        with open(xdmf_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Look for the first Dimensions attribute that has the format "NNNN 3"
        # This typically represents node data (NNNN nodes × 3 coordinates)
        pattern = r'Dimensions="(\d+)\s+3"'
        match = re.search(pattern, content)

        if match:
            return int(match.group(1))
        else:
            # Fallback: look for any Dimensions with just a number (scalar node data)
            pattern = r'Dimensions="(\d+)"'
            matches = re.findall(pattern, content)
            if matches:
                # Return the first number found (likely the node count)
                return int(matches[0])

    except Exception as e:
        print(f"Error reading {xdmf_file}: {e}")
    return None


def process_files(xdmf_files, verbose=True, output_file="mesh_statistics.txt"):
    """
    Process a list of XDMF files to extract node counts and compute statistics.
    Args:
        xdmf_files (list of Path): List of paths to XDMF files
        verbose (bool): If True, print detailed output
        output_file (str): Path to save the statistics output
    Returns:
        dict: Dictionary containing statistics and node counts
    """
    if verbose:
        print(f"Processing files {len(xdmf_files)} XDMF files...")

    node_counts = []
    failed_files = []

    for xdmf_file in xdmf_files:
        node_count = extract_node_count(xdmf_file)
        if node_count is not None:
            node_counts.append(node_count)
            if verbose:
                print(f"{xdmf_file}: {node_count} nodes")
        else:
            failed_files.append(xdmf_file)
            if verbose:
                print(f"{xdmf_file}: Could not extract node count")

    if not node_counts:
        raise ValueError("No valid node counts found in the provided XDMF files.")

    # convert to numpy array for easy stats
    node_counts = np.array(node_counts)

    # statistics
    mean_nodes = np.mean(node_counts)
    std_nodes = np.std(node_counts)
    min_nodes = np.min(node_counts)
    max_nodes = np.max(node_counts)
    median_nodes = np.median(node_counts)

    if verbose:
        print("\n" + "=" * 60)
        print("MESH NODE STATISTICS")
        print("=" * 60)
        print(f"Total files processed: {len(node_counts)}")
        print(f"Failed to process: {len(failed_files)}")
        print()
        print(f"Average number of nodes: {mean_nodes:.2f}")
        print(f"Standard deviation: {std_nodes:.2f}")
        print(f"Minimum nodes: {min_nodes}")
        print(f"Maximum nodes: {max_nodes}")
        print(f"Median nodes: {median_nodes:.1f}")
        print(f"Variance: {std_nodes**2:.2f}")
        print()
        print("Distribution:")
        print(f"  < 1000 nodes: {np.sum(node_counts < 1000)} files")
        print(
            f"  1000-1500 nodes: {np.sum((node_counts >= 1000) & (node_counts < 1500))} files"
        )
        print(
            f"  1500-2000 nodes: {np.sum((node_counts >= 1500) & (node_counts < 2000))} files"
        )
        print(f"  >= 2000 nodes: {np.sum(node_counts >= 2000)} files")

    if failed_files:
        print("\nFiles that could not be processed:")
        for f in failed_files:
            print(f"  {f}")

    # Save results to a file
    with open(output_file, "w") as f:
        f.write("MESH NODE STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total files processed: {len(node_counts)}\n")
        f.write(f"Average number of nodes: {mean_nodes:.2f}\n")
        f.write(f"Standard deviation: {std_nodes:.2f}\n")
        f.write(f"Variance: {std_nodes**2:.2f}\n")
        f.write(f"Minimum nodes: {min_nodes}\n")
        f.write(f"Maximum nodes: {max_nodes}\n")
        f.write(f"Median nodes: {median_nodes:.1f}\n")

    if verbose:
        print(f"\nDetailed results saved to: {output_file}")


def main_mesh_stats(args=None):
    """Main function to process all XMF files and compute statistics."""
    if args is None:
        args = _parser()

    directory = Path(args.directory)
    output_file = os.path.join(directory, "mesh_statistics.txt")
    xdmf_files = list(directory.glob("**/*.xdmf"))
    if not xdmf_files:
        raise FileNotFoundError(f"No .xdmf files found in directory {directory}")
    process_files(xdmf_files, verbose=args.verbose, output_file=output_file)


if __name__ == "__main__":
    main_mesh_stats()
