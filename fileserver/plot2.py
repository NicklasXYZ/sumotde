from posixpath import split
from typing import Dict, Any, Tuple
import os
import pickle
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from numpy.core.fromnumeric import size
from sklearn.metrics import mean_squared_error
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()
# import SeabornFig2Grid as sfg
from math import sqrt



def comparison_plot_estobs(loaded_data):
    c = 1.25

    XLIM = 250

    grouping = {}
    for key in list(loaded_data.keys()):
        split_key = key.split(".")[0].split("_")
        seedmat = get_name(split_key[-1])
        method = split_key[0].split("-")
        method_name = method[1]
        if method_name not in list(grouping.keys()):
            grouping[method_name] = {}
        observed_counts = loaded_data[key]["merged_simdata"]["counts_x"].values
        estimated_counts = loaded_data[key]["merged_simdata"]["counts_y"].values
        rmse_counts = np.sqrt(mean_squared_error(estimated_counts, observed_counts)) 
        observed_odmat = loaded_data[key]["merged_odmat"]["value_x"].values
        estimated_odmat = loaded_data[key]["merged_odmat"]["value_y"].values
        rmse_odmat = np.sqrt(mean_squared_error(estimated_odmat, observed_odmat)) 
        grouping[method_name][seedmat] = {
            "observed_simdata": observed_counts,
            "estimated_simdata": estimated_counts,
            "observed_odmat": observed_odmat,
            "estimated_odmat": estimated_odmat,
            "rmse_simdata": np.round(rmse_counts, 4),
            "rmse_odmat": np.round(rmse_odmat, 4),
        }

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        # sharey=False,
        # sharex=False,
        figsize = (8, 8)
    )
    plt.rcParams["legend.loc"] = "upper left"

    max_0 = 35
    max_1 = 350
    alpha = 0.30

    method_name = "assignmat"
    expr = "E1-3" 

    method_name = "spsa"
    expr = "E4-6" 

    method_name = "knn"
    expr = "E7-9" 

    method_name = "fnn"
    expr = "E10-12" 

    seedmat = "None"
    axes[0][0].scatter(
        grouping[method_name][seedmat]["observed_odmat"],
        grouping[method_name][seedmat]["estimated_odmat"],
        alpha=alpha,
        label=f"RMSE: {grouping[method_name][seedmat]['rmse_odmat']:.4f}",
    )
    axes[0][0].legend(fontsize = "small")
    axes[0][0].set_title("OD Pairs. Using Seed Vector: " + seedmat,  fontsize="small")
    axes[0][0].set_xlim([0 - 2, max_0])
    axes[0][0].set_ylim([0 - 2, max_0])
    axes[0][0].plot(
        np.linspace(0, int(max_0), 100),
        np.linspace(0, int(max_0), 100),
        alpha=0.65,
        linestyle = "--",
        color="red",
    )

    axes[0][1].scatter(
        grouping[method_name][seedmat]["observed_simdata"],
        grouping[method_name][seedmat]["estimated_simdata"],
        alpha=alpha,
        label=f"RMSE: {grouping[method_name][seedmat]['rmse_simdata']:.4f}",
    )
    axes[0][1].legend(fontsize = "small")
    axes[0][1].set_title("Arc Counts. Using Seed Vector: " + seedmat,  fontsize="small")
    axes[0][1].set_xlim([0 - 20, max_1])
    axes[0][1].set_ylim([0 - 20, max_1])
    axes[0][1].plot(
        np.linspace(0, int(max_1), 100),
        np.linspace(0, int(max_1), 100),
        alpha=0.65,
        linestyle = "--",
        color="red",
    )


    seedmat = "LD"
    axes[1][0].scatter(
        grouping[method_name][seedmat]["observed_odmat"],
        grouping[method_name][seedmat]["estimated_odmat"],
        alpha=alpha,
        label=f"RMSE: {grouping[method_name][seedmat]['rmse_odmat']:.4f}",
    )
    axes[1][0].legend(fontsize = "small")
    axes[1][0].set_title("OD Pairs. Using Seed Vector: " + seedmat,  fontsize="small")
    axes[1][0].set_xlim([0 - 2, max_0])
    axes[1][0].set_ylim([0 - 2, max_0])

    axes[1][0].plot(
        np.linspace(0, int(max_0), 100),
        np.linspace(0, int(max_0), 100),
        alpha=0.65,
        linestyle = "--",
        color="red",
    )

    axes[1][1].scatter(
        grouping[method_name][seedmat]["observed_simdata"],
        grouping[method_name][seedmat]["estimated_simdata"],
        alpha=alpha,
        label=f"RMSE: {grouping[method_name][seedmat]['rmse_simdata']:.4f}",
    )
    axes[1][1].legend(fontsize = "small")

    axes[1][1].set_title("Arc Counts. Using Seed Vector: " + seedmat,  fontsize="small")
    axes[1][1].set_xlim([0 - 20, max_1])
    axes[1][1].set_ylim([0 - 20, max_1])
    axes[1][1].plot(
        np.linspace(0, int(max_1), 100),
        np.linspace(0, int(max_1), 100),
        alpha=0.65,
        linestyle = "--",
        color="red",
    )

    seedmat = "HD"
    axes[2][0].scatter(
        grouping[method_name][seedmat]["observed_odmat"],
        grouping[method_name][seedmat]["estimated_odmat"],
        alpha=alpha,
        label=f"RMSE: {grouping[method_name][seedmat]['rmse_odmat']:.4f}",
    )
    axes[2][0].legend(fontsize = "small")

    axes[2][0].set_title("OD Pairs. Using Seed Vector: " + seedmat,  fontsize="small")
    axes[2][0].set_xlim([0 - 2, max_0])
    axes[2][0].set_ylim([0 - 2, max_0])
    axes[2][0].plot(
        np.linspace(0, int(max_0), 100),
        np.linspace(0, int(max_0), 100),
        alpha=0.65,
        linestyle = "--",
        color="red",
    )

    axes[2][1].scatter(
        grouping[method_name][seedmat]["observed_simdata"],
        grouping[method_name][seedmat]["estimated_simdata"],
        alpha=alpha,
        label=f"RMSE: {grouping[method_name][seedmat]['rmse_simdata']:.4f}",
    )
    axes[2][1].legend(fontsize = "small")
    axes[2][1].set_title("Arc Counts. Using Seed Vector: " + seedmat,  fontsize="small")
    axes[2][1].set_xlim([0 - 20, max_1])
    axes[2][1].set_ylim([0 - 20, max_1])
    axes[2][1].plot(
        np.linspace(0, int(max_1), 100),
        np.linspace(0, int(max_1), 100),
        alpha=0.65,
        linestyle = "--",
        color="red",
    )

    for i in range(3):
        for j in range(2):
            leg = axes[i][j].legend(handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False)

   # axes[j][0].scatter(odmat["value_x"], odmat["value_y"], alpha=0.25, label = "RMSE: " + str(np.round(dict[exp]["rmse_odmat"][0], 2)))
    # axes[j][0].scatter(odmat["value_x"], odmat["value_y"], alpha=0.25, label = "RMSE: " + str(np.round(tmp_dict[exp]["rmse_odmat"][0], 2)) + ", NRMSE: " + str(np.round(tmp_dict[exp]["nrmse_odmat"][0], 2)))
    # axes[j][0].scatter(odmat["value_x"], odmat["value_y"], alpha=0.25, label = "RMSE: " + str(np.round(tmp_dict[exp]["rmse_odmat"][0], 2)))


    sns.set_palette(sns.color_palette("Blues_d"))
    sns.set_context("paper")
    sns.set_style("darkgrid", {"axes.facecolor": "1.0"})
    plt.grid(True)
    plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5)
    fig.suptitle(f"Ground Truth & Estimated Values. Experiments {expr}.", y=1)# + str(args.experiments) + ".", y = 1)
    # fig.suptitle("Ground Truth & Estimated Values", y = 1) #. Experiments E" + str(args.experiments) + ".", y = 1)
    fig.text(0.5, 0.02, "Ground Truth Values", ha="center", va="center", size="large")
    fig.text(0.015, 0.5, "Estimated Values", ha="center", va="center", rotation="vertical", size="large")
    # plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5)

    plt.savefig(f"estobs_{expr}.pdf")


def comparison_plot_hist_1(loaded_data):
    c = 1.25
    sns.set_palette(sns.color_palette("Blues_d"))
    sns_cols = sns.color_palette("Blues_d")
    colors = {"None": sns_cols[0], "LD": sns_cols[1], "MD": sns_cols[2], "HD": sns_cols[3]}
    markers = ["1", "+", "o", ".", "3", "*", "2", "4", "5", "6"]
    linestyles = {"None": "solid", "LD": "dotted", "MD": "dashed", "HD": "dashdot"}

    grouping = {}
    for key in list(loaded_data.keys()):
        split_key = key.split(".")[0].split("_")
        seedmat = get_name(split_key[-1])
        method = split_key[0].split("-")
        method_name = method[1]
        if method_name not in list(grouping.keys()):
            grouping[method_name] = {}
        # observed_counts = loaded_data[key]["merged_simdata"]["counts_x"].values
        # estimated_counts = loaded_data[key]["merged_simdata"]["counts_y"].values
        # rmse_counts = np.sqrt(mean_squared_error(estimated_counts, observed_counts)) 
        # observed_odmat = loaded_data[key]["merged_odmat"]["value_x"].values
        # estimated_odmat = loaded_data[key]["merged_odmat"]["value_y"].values
        # rmse_odmat = np.sqrt(mean_squared_error(estimated_odmat, observed_odmat)) 
        of_value_history = loaded_data[key]["of_history"]
        of_value_history = [
            _sum_of_values(of_values=of_values)
            for of_values in of_value_history
        ]
        length1 = len(of_value_history); x = [i for i in range(length1)]
        grouping[method_name][seedmat] = {
            "of_history": of_value_history, 
            "x": x,
            # "observed_simdata": observed_counts,
            # "estimated_simdata": estimated_counts,
            # "observed_odmat": observed_odmat,
            # "estimated_odmat": estimated_odmat,
            # "rmse_simdata": np.round(rmse_counts, 4),
            # "rmse_odmat": np.round(rmse_odmat, 4),
        }

    method_name = "assignmat"
    expr = "E1-3" 

    method_name = "spsa"
    expr = "E4-6" 

    method_name = "knn"
    expr = "E7-9" 

    method_name = "fnn"
    expr = "E10-12" 


    fig, axes = plt.subplots(nrows = 3, ncols = 1, sharey = True, sharex = True, figsize = (6, 8))

    size = 0.5
    alpha = 0.75
    linewidth = 2.25

    seedmat = "None"
    axes[0].plot(
        grouping[method_name][seedmat]["x"],
        grouping[method_name][seedmat]["of_history"],
        label = seedmat,
        linestyle = linestyles[seedmat],
        color = colors[seedmat],
        linewidth=linewidth,
    )
    axes[0].scatter(
        grouping[method_name][seedmat]["x"],
        grouping[method_name][seedmat]["of_history"],
        color = colors[seedmat],
        marker="o",
        s=size,
        alpha=alpha
    )

    axes[0].legend(fontsize = "small")
    axes[0].set_title("Seed Vector: None")

    seedmat = "LD"
    axes[1].plot(
        grouping[method_name][seedmat]["x"],
        grouping[method_name][seedmat]["of_history"],
        label = seedmat,
        linestyle = linestyles[seedmat],
        color = colors[seedmat],
        linewidth=linewidth,
    )
    axes[1].scatter(
        grouping[method_name][seedmat]["x"],
        grouping[method_name][seedmat]["of_history"],
        color = colors[seedmat],
        marker="o",
        s=size,
        alpha=alpha
    )

    axes[1].legend(fontsize = "small")
    axes[1].set_title("Seed Vector: LD")

    seedmat = "HD"
    axes[2].plot(
        grouping[method_name][seedmat]["x"],
        grouping[method_name][seedmat]["of_history"],
        label = seedmat,
        linestyle = linestyles[seedmat],
        color = colors[seedmat],
        linewidth=linewidth,
    )
    axes[2].scatter(
        grouping[method_name][seedmat]["x"],
        grouping[method_name][seedmat]["of_history"],
        color = colors[seedmat],
        marker="o",
        s=size,
        alpha=alpha
    )

    axes[2].legend(fontsize = "small")
    axes[2].set_title("Seed Vector: HD")

    axes[0].set_ylim([0.0, 2.25])
    axes[1].set_ylim([0.0, 2.25])
    axes[2].set_ylim([0.0, 2.25])
    # loc = plticker.MultipleLocator(base = 2.0) # this locator puts ticks at regular intervals
    # axes[0].xaxis.set_major_locator(loc)
    # axes[1].xaxis.set_major_locator(loc)
    # loc = plticker.MultipleLocator(base = 10.0) # this locator puts ticks at regular intervals

    plt.rcParams["legend.loc"] = "upper right"
    plt.grid(True)
    sns.set_style("darkgrid", {"axes.facecolor": "1.0"})
    sns.set_context("paper")


    fig.suptitle(f"Objective Function Value History. Experiments {expr}.", y = 0.9425)

    fig.text(0.5, 0.0240, "Objective Function Evaluation", ha="center", va="center", size="large")
    fig.text(0.015, 0.5, "Objective Function Value", ha="center", va="center", rotation="vertical", size="large")

    # plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=0.0)
    # plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=10.0)
    plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5)
    # plt.tight_layout(w_pad=2.5, h_pad=2.5)

    plt.savefig(f"iters_{expr}" + ".pdf")#, pad_inches = 10)
    # plt.show()



def _sum_of_values(of_values: Dict[str, Tuple[float, float]]) -> float:
    return np.sum(
        [weight * of_value for weight, of_value in of_values.values()]
    )

def comparison_plot_of_value_history(
    loaded_data: Dict[str, Any],
    # det_subset: pd.DataFrame
    ) -> None:
    c = 1.25
    XLIM = 205
    YLIM = 2.0

    # Organice data for plotting purposes
    # Organize data by seedmatrix used
    grouping = {}
    for key in list(loaded_data.keys()):
        of_value_history = loaded_data[key]["of_history"]
        # of_value_history = [
        #     _sum_of_values(of_values=of_values)
        #     for of_values in of_value_history
        # ]
        of_value_history_term_f1 = [
            of_values["odmat"][0] * of_values["odmat"][1] 
            for of_values in of_value_history
        ]
        of_value_history_term_f2 = [
            of_values["counts"][0] * of_values["counts"][1] 
            for of_values in of_value_history
        ]

        # if np.max(of_value_history) > max_value:
        #     max_value = np.max(of_value_history)
        split_key = key.split(".")[0].split("_")
        seedmat = get_name(split_key[-1])
        method = split_key[0].split("-")


        if method[1] not in grouping:
            grouping[method[1]] = {
                "HD": {},
                "LD": {},
                "None": {},
            }
        grouping[method[1]][seedmat] = {
            "f1": of_value_history_term_f1,
            "f2": of_value_history_term_f2,
        }
    
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        sharey=True,
        sharex=True,
        figsize=(8, 4),
    )

    def plot_row_pair(_method, _seedmat, _axes, i, j):
        y1 = grouping[_method][_seedmat]["f1"]
        y2 = grouping[_method][_seedmat]["f2"]
        length1 = len(y1); x1 = [i for i in range(length1)]
        length2 = len(y2); x2 = [i for i in range(length2)]
        axes[i  ][j].plot(x1, y1, label = f"{method}, f1")
        axes[i+1][j].plot(x2, y2, label = f"{method}, f2")
        
    plot_row_pair("assignmat", "HD",   axes, 0, 0)
    plot_row_pair("assignmat", "LD",   axes, 0, 1)
    plot_row_pair("assignmat", "None", axes, 0, 2)


    # fig.suptitle("Objective Function Value History. Experiments E" + str(args.experiments) + ".", y = 1)
    fig.suptitle("Objective Function Value History")#, y = 1.5)
    # fig.suptitle("Objective Function Value History. OD Pairs: " + args.name, y = 1)
    # fig.text(0.5, 0.02, "Objective Function Evaluation", ha="center", va="center", size="large")
    fig.text(0.5, 0.02, "Objective Function Evaluation", ha="center", va="center", size="large")
    fig.text(0.015, 0.5, "Objective Function Value", ha="center", va="center", rotation="vertical", size="large")

    sns.set_context("paper")
    sns.set_style("darkgrid", {"axes.facecolor": "1.0"})
    plt.grid(True)
    plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5)
    # plt.tight_layout()
    plt.savefig("OF_progress.pdf")
    # plt.show()

def comparison_method(
    loaded_data: Dict[str, Any],
    det_subset: pd.DataFrame
    ) -> None:
    c = 1.25

    # Organice data for plotting purposes
    # Organize data by seedmatrix used
    x = []
    y = []
    labels = []
    for key in list(loaded_data.keys()):
        _split_key = key.split(".")[0]
        split_key = _split_key.split("_")
        seedmat = get_name(split_key[-1])
        method = split_key[0].split("-")
        observed_counts = loaded_data[key]["merged_simdata"]["counts_x"].values
        esimtated_counts = loaded_data[key]["merged_simdata"]["counts_y"].values
        rmse_counts = np.sqrt(mean_squared_error(esimtated_counts, observed_counts)) 
        observed_values = loaded_data[key]["merged_odmat"]["value_x"].values
        esimtated_values = loaded_data[key]["merged_odmat"]["value_y"].values
        rmse_odmat = np.sqrt(mean_squared_error(esimtated_values, observed_values)) 
        print(
            "Method     : ", method,
            "Seedmat    : ", seedmat,
            "RMSE COUNTS: ", rmse_counts,
            "RMSE ODMAT : ", rmse_odmat,
            "OF EVALS   : ", len(loaded_data[key]["of_history"]),
        )
        x.append(rmse_counts)
        y.append(rmse_odmat)
        labels.append(_split_key)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=1,
        sharey=True,
        sharex=True,
        figsize=(8, 4),
    )
    for i in range(len(labels)):
        axes.legend(fontsize = "small")
        axes.scatter(x[i], y[i], label=labels[i], s=30)
    axes.set_xlim(0, np.max(x) + 10)
    axes.set_ylim(0, np.max(y) + 10)
    axes.set_xlabel("RMSE COUNTS")
    axes.set_ylabel("RMSE ODMAT")
    sns.set_context("paper")
    sns.set_style("darkgrid", {"axes.facecolor": "1.0"})
    plt.grid(True)
    # plt.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5)
    plt.tight_layout()
    plt.show()


def read_xml_file(xml_file):
    # Use the python .XML parser to read the input file
    parser = etree.XMLParser(dtd_validation=False, no_network=True)
    root = etree.parse(xml_file, parser).getroot()
    return root


def read_det_subset_file(det_subset_file: str) -> pd.DataFrame:
    root = read_xml_file(det_subset_file)
    subset_det = []
    for child in root:
        if child.tag == "additional":
            for subchild in child:
                if subchild.tag == "inductionLoop":
                    subset_det_attribs = {}
                    for attribute in subchild.attrib:
                        if attribute == "id":
                            subset_det_attribs["edge"] = \
                                subchild.attrib["id"]
                        else:
                            subset_det_attribs[attribute] = \
                                subchild.attrib[attribute]
                    subset_det.append(subset_det_attribs)
    df = pd.DataFrame(data=subset_det, dtype=object)
    return df


def load_data(data_dir, det_subset_df):
    files = [
        f for f in os.listdir(data_dir) if os.path.isfile(
            os.path.join(data_dir, f)
        )
    ]
    # Set column types otherwise pandas pares ID's as ints...
    simdata_dtypes = {
        "begin": float,
        "end": float,
        "link": str,
        "counts_x": int,
        "counts_y": int,
    } 
    pd.set_option('display.max_rows', None)

    loaded_files = {}
    for file in files:
        with open(os.path.join(data_dir, file), mode="r") as _f:
            f = json.load(_f)
            f["merged_odmat"] = pd.read_json(f["merged_odmat"])
            simdata =  pd.read_json(f["merged_simdata"], dtype=simdata_dtypes)
            simdata = simdata.loc[simdata["link"].isin(det_subset_df["edge"])]
            # Value is 0 if NA, as no data was ever parsed with those attributes
            simdata = simdata.fillna(0.0)
            f["merged_simdata"] = simdata
            loaded_files[file] = f
    return loaded_files


def get_name(name):
    if name == "seedmat00":
        name = "None"
    elif name == "None":
        name = "None"
    elif name == "seedmat07":
        name = "LD"
    elif name == "seedmat08":
        name = "MD"
    elif name == "seedmat09":
        name = "HD"
    return name


if __name__ == "__main__":
    data_dir = "static/run05"
    det_subset_file = "input_files/network/sumo-grid-network-od132_100.det_subset.xml"
    det_subset_df = read_det_subset_file(det_subset_file=det_subset_file)

    loaded_data = load_data(data_dir, det_subset_df)

    # Plot estimated against observed quantities
    comparison_plot_estobs(loaded_data)

    # Plot the objective function value history
    # comparison_plot_of_value_history(
    #     loaded_data=loaded_data,
    #     det_subset=det_subset_df,
    # )
