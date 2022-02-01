import pickle
import argparse
import sys
import os
import pandas as pd
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_commandline_args(args_list=None):
    """
    """
    # Create main parser
    parser = argparse.ArgumentParser(description="")
    # Add commandline options
    add_parser_args(parser)
    # Parse commandline arguments
    args = parser.parse_args(args_list)
    return args

def add_parser_args(parser):
    """
    """
    parser.add_argument("-data_dir", "--data_dir",
        required=True,
        type=str,
        help="")
    parser.add_argument("-name", "--name",
        required=True,
        type=str,
        help="")

def get_name(file):
    name = os.path.basename(file).split(".")[0]
    if "seedmat07" == name:
        return "seedmat08"
    elif "seedmat08" == name:
        return "seedmat08"
    elif "seedmat09" == name:
        return "seedmat09"
    elif "tripinfo" == name:
        return "tripinfo"
    elif "assignmat" == name:
        return "assignmat"
    elif "summary" == name:
        return "summary"
    elif "gencon" == name:
        return "gencon"
    elif "odmat" == name:
        return "odmat"
    elif "simdata" == name:
        return "simdata"
    else:
        raise ValueError("TODO")

def load_data(args):
    """
    """
    file_list = [
        os.path.join(args.data_dir, file) 
        for file in os.listdir(args.data_dir) if "csv" in file.split(".")
    ]
    print("The following files were read in:")
    for file in file_list:
        print(file)
    print()
    df_dict = {}
    for file in file_list:
        name = get_name(file)
        df = pd.read_csv(file, index_col=0)
        df_dict[name] = df
    return df_dict

def plot_scen_data(df_dict):
    # Layout settings
    sns.set_palette(sns.color_palette("Blues_d"))
    fig, ax = plt.subplots(
        nrows=3,
        ncols=1,
        sharey=False,
        sharex=False,
        figsize = (8, 8),
    )
    # Plot stuff...
    start = 0
    end = 4500
    key = "tripinfo"
    df = df_dict["tripinfo"]

    fig_width_pt = 408.0  # Get this from LaTeX using \memssage{\showthe\columnwidth}
    inches_per_pt = 1.0 / 72.27                # Convert pt to inches
    golden_mean = (sqrt(5) - 1.0) / 2.0        # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt   # width in inches
    fig_height = fig_width * golden_mean       # height in inches
    fig_size = [fig_width, fig_height]
    params = {
        "backend": "ps",
        "text.usetex": True,
        "figure.figsize": fig_size,
    }
    size = fig.get_size_inches()
    # fig.set_size_inches(20, 20)
    # fig.set_size_inches(0.75 * size[0], 0.75 * size[1])
    fig.subplots_adjust(wspace=0.25, hspace=0.5)
    sns.set_context("paper")
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)
    plt.rcParams["legend.loc"] = "upper right"

    x = df.loc[df["depart"] <= end, "depart"].values
    y = df.loc[df["depart"] <= end, "routeLength"].values
    hist, bins = np.histogram(y, bins = 16)
    width = 0.90 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax[0].bar(center, hist, align = "center", width = width)
    ax[0].set_xlabel("Trip Length (meter)")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Trip Length Distribution", fontsize="large")
    ax[0].axvline(
        np.mean(y),
        label = "Mean Trip Length: " + \
            str(np.round(np.mean(y), 2)),
        color = "red",
        linestyle = "--"
    )
    ax[0].legend(fontsize = "small")
    # ax[0].set_xticks([])
    # ax[0].set_xticklabels()

    y = df.loc[df["depart"] <= end, "duration"].values
    hist, bins = np.histogram(y, bins = 16)
    width = 0.90 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax[1].bar(center, hist, align = "center", width = width)
    ax[1].set_xlabel("Trip Duration (seconds)")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Trip Duration Distribution", fontsize="large")
    ax[1].axvline(
        np.mean(y),
        label = "Mean Trip Duration: " + \
            str(np.round(np.mean(y), 2)),
        color = "red",
        linestyle = "--"
    )
    ax[1].legend(fontsize = "small")

    # y = df.loc[df["depart"] <= end, "depart"].values
    # hist, bins = np.histogram(y, bins = 16)
    # width = 0.90 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # ax[2].bar(center, hist, align = "center", width = width)
    # ax[2].set_xlabel("Trip Departure Time (seconds)")
    # ax[2].set_ylabel("Frequency")
    # ax[2].set_title("Departure Time Distribution", fontsize="large")
    # ax[2].axvline(
    #     np.mean(y),
    #     label = "Mean Trip Departure Time: " + \
    #         str(np.round(np.mean(y), 2)), color = "red", linestyle = "--"
    # )
    # ax[2].legend(fontsize = "small")

    y = df.loc[df["depart"] <= end, "timeLoss"].values
    ax[2].scatter(x, y, alpha = 0.25)
    ax[2].set_xlabel("Trip Departure Time (seconds)")
    ax[2].set_ylabel("Trip Time Lost (seconds)")
    ax[2].set_title("Time Lost Due to Congestion", fontsize="large")
    ax[2].axhline(
        np.mean(y),
        label = "Mean Trip Time Lost: " + \
            str(np.round(np.mean(y), 2)),
        color = "red",
        linestyle = "--"
    )
    ax[2].legend(fontsize = "small")

    fig.subplots_adjust(wspace=0.05, hspace=0.5)
    plt.tight_layout()
    plt.savefig("scen_od56" + args.name + ".png")
    plt.savefig("scen_od56" + args.name + ".pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":
    args = get_commandline_args()
    df_dict = load_data(args)
    plot_scen_data(df_dict)