import os
import logging
import pandas as pd
from pathlib import Path
from baselines_lab.utils.tensorboard import TrainingInformation
from baselines_lab.utils import config_util

tags = ["episode_length/eval_ep_length_mean"]
discrete_fieldnames = ["Run", "Obs Norm", "Rew Norm", "TP", "DEL", "RND", "GR", "Best", "Avg", "Drop"]
continuous_fieldnames = ["Run", "Obs Norm", "Rew Norm", "TP", "DEL", "RND", "GR", "Int Norm", "PO", "Best", "Avg", "Drop"]


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


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
            file_data["Obs Norm"] = "X"
        if config["env"]["normalize"].get("norm_reward", True):
            file_data["Rew Norm"] = "X"
    if config["env"].get("curiosity", False):
        scale = config["env"]["curiosity"].get("intrinsic_reward_weight", 1.0)
        file_data["RND"] = "{:.2f}".format(scale)

    info = TrainingInformation(str(file))
    drop_train, drop_test, avg, min = info.log_key_points(drop_level=0.05, max_step=6000000)

    file_data["Best"] = "{:.2f}".format(min)
    file_data["Avg"] = "{:.2f}".format(avg)
    if env_kwargs.get("dynamic_episode_length", False):
        file_data["Drop"] = human_format(int(drop_test))
    else:
        file_data["Drop"] = human_format(int(drop_train))
    file_data["Run"] = file.name
    return file_data


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    log_dir_path = Path("F:\\Uni\\2020_Semester XIV\\Learning_Archive\\Reward_Experimente\\VesselMaze02\\continuous")

    directories = [f.path for f in os.scandir(log_dir_path) if f.is_dir()]
    #fieldnames = ["Run", "Obs Norm", "Rew Norm", "TP", "DEL", "RND", "GR", "Best", "Avg", "Drop"]
    fieldnames = continuous_fieldnames
    sort = 8 # 6/8

    entries = []
    for dir in directories:
        entries.append(process_file(Path(dir)))
    dataframe = pd.DataFrame(entries, columns=fieldnames)
    dataframe.sort_values(fieldnames[1:sort], inplace=True, ascending=False, na_position="first")
    dataframe.reset_index(drop=True, inplace=True)
    dataframe.to_csv(str(log_dir_path.joinpath('output.csv')), index_label="Index")

