import argparse
import os
import logging
import sys

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from baselines_lab.utils.tensorboard import TrainingInformation
from baselines_lab.utils import config_util
from utils.tensorboard import Plotter

tags = ["episode_length/eval_ep_length_mean"]
discrete_fieldnames = ["Run", "ON", "RN", "TP", "DEL", "RND", "GR", "Best", "Avg", "Drop"]
continuous_fieldnames = ["Run", "ON", "RN", "TP", "DEL", "RND", "GR", "Int Norm", "PO", "Best", "Avg", "Drop"]


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'k', 'M', 'G', 'T'][magnitude])


def process_file(file):
    config = config_util.read_config(str(file.joinpath("config.yml")))
    if "Continuous" in config["env"]["name"]:
        return process_continuous_file(file, config)
    elif "Discrete" in config["env"]["name"]:
        return process_discrete_file(file, config)
    else:
        raise ValueError("Unknown env type {}!".format(config["env"]["name"]))


def process_continuous_file(file, config):
    file_data = process_discrete_file(file, config)
    env_kwargs = config["env"]["reward_kwargs"]
    if env_kwargs.get("positive_only", False):
        file_data["PO"] = "X"
    if env_kwargs.get("normalize", True):
        file_data["Int Norm"] = "X"
    return file_data


def process_discrete_file(file, config):
    file_data = {}

    env_kwargs = config["env"]["reward_kwargs"]
    if env_kwargs.get("time_penalty", True):
        file_data["TP"] = "X"
    if env_kwargs.get("dynamic_episode_length", False):
        file_data["DEL"] = "X"
    gathering_reward = env_kwargs.get("gathering_reward", 0.0)
    if gathering_reward > 0.0:
        file_data["GR"] = "{:.2f}".format(gathering_reward)
    if config["env"].get("normalize", False):
        if config["env"]["normalize"].get("norm_obs", True):
            file_data["ON"] = "X"
        if config["env"]["normalize"].get("norm_reward", True):
            file_data["RN"] = "X"
    if config["env"].get("curiosity", False):
        scale = config["env"]["curiosity"].get("intrinsic_reward_weight", 1.0)
        file_data["RND"] = "{:.2f}".format(scale)

    info = TrainingInformation(str(file))
    drop_train, drop_test, avg, min = info.log_key_points(drop_level=0.05, max_step=6000000)

    file_data["Best"] = min
    file_data["Avg"] = avg
    if env_kwargs.get("dynamic_episode_length", False):
        file_data["Drop"] = int(drop_test)
    else:
        file_data["Drop"] = int(drop_train)
    file_data["Run"] = file.name
    return file_data


def to_latex_frame(frame: pd.DataFrame, value_cols, max_min):
    # Get positions of best values
    bold_positions = []
    for col, op in zip(value_cols, max_min):
        if op == "max":
            bold_positions.append(frame[col].idxmax())
        elif op == "min":
            bold_positions.append(frame[col].idxmin())
        else:
            raise ValueError("Unknown operation {}".format(op))

    # Format numbers
    for col in value_cols:
        if frame[col].dtype == np.int64:
            frame[col] = frame[col].apply(human_format)
        else:
            frame[col] = frame[col].apply(lambda x: "{:.2f}".format(x))

    # Make best values bold
    for row, col in zip(bold_positions, value_cols):
        frame.at[row, col] = "\\textbf{" + frame[col][row] + "}"

    # Drop Run ID
    frame.drop("Run", axis=1, inplace=True)

    return frame


def to_latex(frame : pd.DataFrame, location: str, n_value_cols=3):
    value_cols = list(frame.columns)[-n_value_cols:]
    frame = to_latex_frame(frame, value_cols, ["min"]*n_value_cols)
    indent = "    "
    property_cols = len(frame.columns) - n_value_cols

    with open(location, "w") as f:

        f.write("\\begin{table}[ht]\n")
        f.write(indent + "\\begin{center}\n")
        # f.write(indent * 2 + "\\begin{tabular}{| " + " | ".join(["c"] * (len(frame.columns) - n_value_cols)) + " | " + " | ".join(["L"]*n_value_cols) + " |}\n")
        f.write(indent * 2 + "\\begin{tabular}{r" + "c" * (len(frame.columns) - n_value_cols) + "r" * n_value_cols + "}\n")
        f.write(indent * 3 + "\\toprule\n")
        f.write(indent * 3 + " & \\multicolumn{" + str(property_cols) + "}{c}{Reward Component} & \multicolumn{2}{c}{Episode Length} & \\\\\n")
        f.write(indent * 3 + "\\cmidrule(lr){2-" + str(property_cols+1) + "}" + "\\cmidrule(lr){" + str(property_cols+2) + "-" + str(len(frame.columns)) + "}\n")
        f.write(indent * 3)
        f.write("\\multicolumn{1}{c}{Idx} & ")
        for i, col in enumerate(frame.columns):
            f.write("\\multicolumn{{1}}{{c}}{{{col}}}".format(col=col))
            if i < len(frame.columns) - 1:
                f.write(" & ")
            else:
                f.write("\\\\\n")
        #f.write(indent * 3 + " & ".join(list(frame.columns)) + "\n")
        f.write(indent * 3 + "\\midrule\n")
        for i, row in frame.iterrows():
            f.write(indent * 3 + str(i+1) + " & " + " & ".join(["" if pd.isna(x) else str(x) for x in row.values]) + " \\\\\n")
        f.write(indent * 3 + "\\bottomrule\n")
        f.write(indent * 2 + "\\end{tabular}\n")
        f.write(indent + "\\end{center}\n")
        f.write("\\end{table}")


def parse_args(args):
    parser = argparse.ArgumentParser("Run script for baselines lab stats module.")

    parser.add_argument("config_file", type=str, help="Location of the lab config file. May be a list or a directory.")
    return parser.parse_args(args=args)


def make_table(config, directories):
    pass


def make_figure(config, directories):
    file_format = config.get("format", "pdf")
    output_dir = config.get("output", "./logs")
    tags = config.get("tags", ["episode_length/ep_length_mean"])
    names = config.get("names", ["Episode Length"])
    if len(tags) != len(names):
        raise ValueError("There must be a name for each tag and vice versa!")
    plot_avg_only = config.get("plot_avg_only", False)
    smoothing = config.get("smoothing", 0.9)
    alias = config.get("alias", None)


    plot = Plotter(file_format, directories, output_dir)
    plot.make_plot(tags=tags, names=names, plot_avg_only=plot_avg_only, smoothing=smoothing, alias=alias)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(logging.INFO)
    file = open(args.config_file, "r")
    config = yaml.load(file)
    file.close()

    directories = []
    for dir in config["runs"]:
        filelist = os.listdir(dir)
        if "config.yml" in filelist:  # Directory is a run dir
            directories.append(dir)
        else:  # Directory contains multiple run dirs.
            directories.extend([f.path for f in os.scandir(dir) if f.is_dir()])

    for job in config["jobs"]:
        type = job["type"]
        if type == "figure":
            make_figure(job, directories)
        elif type == "table":
            make_table(job, directories)


if __name__ == "__main__":
    main()

    # log_dir_path = Path("F:\\Uni\\2020_Semester XIV\\Learning_Archive\\Reward_Experimente\\VesselMaze02\\discrete")
    #
    # directories = [f.path for f in os.scandir(log_dir_path) if f.is_dir()]
    # #fieldnames = ["Run", "Obs Norm", "Rew Norm", "TP", "DEL", "RND", "GR", "Best", "Avg", "Drop"]
    # fieldnames = discrete_fieldnames
    # sort = 6 # 6/8
    #
    # entries = []
    # for dir in directories:
    #     entries.append(process_file(Path(dir)))
    # dataframe = pd.DataFrame(entries, columns=fieldnames)
    # dataframe.sort_values(fieldnames[1:sort], inplace=True, ascending=False, na_position="first")
    # dataframe.reset_index(drop=True, inplace=True)
    # #to_latex_frame(dataframe, ["Best", "Avg", "Drop"], ["min", "min", "min"])
    # #dataframe.to_csv(str(log_dir_path.joinpath('output.csv')), index_label="Index")
    # to_latex(dataframe, str(log_dir_path.joinpath('output.tex')))

