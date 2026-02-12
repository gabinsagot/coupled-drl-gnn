import numpy as np


def read_file(
    filename: str,
    iter_list: list,
    r: list,
    theta: list,
    wr: float,
    wtheta: float,
    nenv: int,
    nepmax: int,
    nactions: int,
):
    """
    Read reward and action data from a file and populate the provided lists.

    Args:
        filename (str): The name of the pbo reward file to read.
        iter_list (list): List to store iteration numbers.
        r (list): List to store reward values.
        theta (list): List of lists to store action values.
        wr (float): Reward scaling factor.
        wtheta (float): Action scaling factor.
        nenv (int): Number of parallel environments/individuals per episode/generation.
        nepmax (int): Maximum number of episodes/generations.
        nactions (int): Number of actions.
    Returns:
        None
    """
    with open(filename, "r") as file:
        for line in file:
            values = line.split()
            _, iter_tmp = float(values[0]), float(values[1])
            if iter_tmp <= nepmax * nenv and (
                len(iter_list) == 0 or iter_tmp > iter_list[-1]
            ):
                r_tmp = float(values[2])
                theta_tmp = list(map(float, values[3 : 3 + nactions]))
                iter_list.append(iter_tmp)
                r.append(r_tmp / wr)
                for i in range(nactions):
                    theta[i].append(theta_tmp[i] * wtheta)


def discretize_actions(theta: list, discretize_set: np.ndarray):
    """
    Discretize action values to the nearest values in a given discrete set.
    """
    for i in range(len(theta)):
        for j in range(len(theta[i])):
            theta[i][j] = float(
                discretize_set[np.abs(discretize_set - theta[i][j]).argmin()]
            )


def compute_discretize_set(action_step: float, action_range: tuple) -> np.ndarray:
    """
    Compute a set of discrete action values evenly spaced within a given range.

    Args:
        action_step (float): Step size between discrete actions.
        action_range (tuple): Tuple containing the minimum and maximum action values.

    Returns:
        np.ndarray: Array of discrete action values.
    """
    total_actions = int((action_range[1] - action_range[0]) / action_step) + 1
    return np.linspace(action_range[0], action_range[1], total_actions)


def calc_slideav(myxav: list, x: list, nlines: int, nav: int):
    """
    Calculate the sliding average of a list of values.

    Args:
        myxav (list): List to store the sliding average values.
        x (list): List of values to calculate the sliding average from.
        nlines (int): Total number of lines (length of x).
        nav (int): Number of points to average over.
    """
    for i in range(nav - 1):
        val = sum(x[: i + 1]) / (i + 1)
        myxav.append(val)

    for i in range(nav - 1, nlines):
        val = sum(x[i - nav + 1 : i + 1]) / nav
        myxav.append(val)


def calc_av(mean: list, rms: list, y: list, nep: int, nenv: int):
    """
    Calculate the mean and RMS of the last 'nep' episodes/generations from the list 'y'.

    Args:
        mean (list): List to store the mean value.
        rms (list): List to store the RMS value.
        y (list): List of values to calculate the mean and RMS from.
        nep (int): Number of episodes/generations from end considered in computation.
        nenv (int): Number of parallel environments/individuals per episode/generation.
    """
    sub_y = y[-nep * nenv :]
    mean_val = np.mean(sub_y)
    rms_val = np.sqrt(np.mean((sub_y - mean_val) ** 2))

    mean[0] = mean_val
    rms[0] = rms_val


def _parser():
    """
    Argument parser for the postprocessing script.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Postprocessing script for raw PBO reward file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", type=str, help="Reward file path from pbo output.")
    parser.add_argument(
        "--name",
        type=str,
        default="reward_pbo.txt",
        help="Output file name for the processed reward (.txt format, file extension automatically added).",
    )
    parser.add_argument(
        "--nactions",
        type=int,
        default=6,
        help="Number of actions taken at each environment/individual.",
    )
    parser.add_argument(
        "--wr",
        type=float,
        default=1,
        help="Reward scaling factor, R/wr.",
    )
    parser.add_argument(
        "--wa",
        type=float,
        default=1,
        help="Action scaling factor, theta*wa.",
    )
    parser.add_argument(
        "--discrete_action_step",
        type=float,
        default=None,
        help="Step size for discretizing actions, if None actions are not discretized.",
    )
    parser.add_argument(
        "--discrete_action_range",
        type=float,
        nargs=2,
        default=None,
        help="Range (min max) for discretizing actions, if None actions are not discretized.",
    )
    parser.add_argument(
        "--nep",
        type=int,
        default=100,
        help="Maximum number of episodes to process.",
    )
    parser.add_argument(
        "--nenv",
        type=int,
        default=8,
        help="Number of parallel environments/individuals per episode/generation.",
    )
    parser.add_argument(
        "--nav",
        type=int,
        default=50,
        help="Number of environments/individuals for sliding average.",
    )
    parser.add_argument(
        "--final_slice",
        type=int,
        default=5,
        help="Number of episodes/generations from end considered in computation of final mean and RMS.",
    )
    return parser.parse_args()


def process_pbo_data(
    data_file: str,
    output_name: str,
    nepmax: int,
    nenv: int,
    nactions: int,
    wr: float,
    wtheta: float,
    discrete_action_step: float | None,
    discrete_action_range: tuple[float, float] | None,
    nav: int,
    final_slice: int,
):
    """
    Process PBO reward data and generate output file with moving averages.

    Args:
        data_file: Path to the input data file.
        output_name: Name of the output file.
        nepmax: Maximum number of episodes to process.
        nenv: Number of parallel environments/individuals per episode/generation.
        nactions: Number of actions taken at each environment/individual.
        wr: Reward scaling factor.
        wtheta: Action scaling factor.
        discrete_action_step: Step size for discretizing actions, if None actions are not discretized.
        discrete_action_range: Range (min, max) for discretizing actions, if None actions
        nav: Number of environments/individuals for sliding average.
        final_slice: Number of episodes/generations from end considered in computation of final mean and RMS.
    """
    iter_list, r = [], []
    theta = [[] for _ in range(nactions)]
    print(f"\nLoaded PBO rewards and info from datafile: {data_file}\n")

    read_file(
        filename=data_file,
        iter_list=iter_list,
        r=r,
        theta=theta,
        wr=wr,
        wtheta=wtheta,
        nenv=nenv,
        nepmax=nepmax,
        nactions=nactions,
    )
    nlines = len(iter_list)
    print(f"Total environments: {nlines}")

    # discretize actions if requested
    if discrete_action_step is not None and discrete_action_range is not None:
        discretize_set = compute_discretize_set(
            action_step=discrete_action_step, action_range=discrete_action_range
        )
        discretize_actions(theta=theta, discretize_set=discretize_set)
        print(
            f"Actions discretized with step {discrete_action_step} in range {discrete_action_range}"
        )

    # moving avg
    rav = []
    theta_av = [[] for _ in range(nactions)]
    calc_slideav(myxav=rav, x=r, nlines=nlines, nav=nav)
    for i in range(nactions):
        calc_slideav(myxav=theta_av[i], x=theta[i], nlines=nlines, nav=nav)

    # last final_slice episodes
    rmean, rrms = [0], [0]
    theta_mean = [[0] for _ in range(nactions)]
    theta_rms = [[0] for _ in range(nactions)]
    calc_av(mean=rmean, rms=rrms, y=r, nep=final_slice, nenv=nenv)
    for i in range(nactions):
        calc_av(
            mean=theta_mean[i], rms=theta_rms[i], y=theta[i], nep=final_slice, nenv=nenv
        )

    # print results
    print(f"\nLast {final_slice} episodes: (avg) (rms) (rms/avg)")
    rms_mean_ratio = rrms[0] / rmean[0] if rmean[0] != 0 else 0
    print(f"    reward {rmean[0]:.6f} {rrms[0]:.6f} {rms_mean_ratio:.6f}")
    for i in range(nactions):
        rms_mean_ratio_theta = (
            theta_rms[i][0] / theta_mean[i][0] if theta_mean[i][0] != 0 else 0
        )
        print(
            f"    theta{i+1} {theta_mean[i][0]:.6f} {theta_rms[i][0]:.6f} {rms_mean_ratio_theta:.6f}"
        )

    # write output file
    output_file_name = (
        output_name if output_name.endswith(".txt") else output_name + ".txt"
    )
    print(f"\nOutput file: {output_file_name}\n")
    with open(output_file_name, "w") as file:
        for i in range(nlines):
            file.write(
                f"{iter_list[i]/nenv} {r[i]:.7f} "
                + " ".join(f"{theta[j][i]:.7f}" for j in range(nactions))
                + " "
                + f"{rav[i]:.7f} "
                + " ".join(f"{theta_av[j][i]:.7f}" for j in range(nactions))
                + "\n"
            )
    return ()


def main():
    # Parser
    args = _parser()

    # Call the function
    process_pbo_data(
        data_file=args.file,
        output_name=args.name,
        nepmax=args.nep,
        nenv=args.nenv,
        nactions=args.nactions,
        wr=args.wr,
        wtheta=args.wa,
        discrete_action_step=args.discrete_action_step,
        discrete_action_range=args.discrete_action_range,
        nav=args.nav,
        final_slice=args.final_slice,
    )


if __name__ == "__main__":
    main()
