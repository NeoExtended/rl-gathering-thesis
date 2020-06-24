from abc import ABC, abstractmethod
from pathlib import Path

from typing import List, Dict, Any, Tuple, Optional
from datetime import timedelta

import pandas as pd
import numpy as np

from baselines_lab.utils import config_util
from baselines_lab.utils.tables.latex import LatexTableWriter
from baselines_lab.utils.tensorboard import TrainingInformation


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'k', 'M', 'G', 'T'][magnitude])


class TableGenerator(ABC):
    """
    Used to create latex tables for the thesis. Not really a part of the rest of the lab.
    """

    def __init__(self, files: List[Path], best: bool = True, avg: bool = True, drop: bool = True, run_id: bool = False, time: bool = False):
        self.files = files
        self.best = best
        self.avg = avg
        self.drop = drop
        self.run_id = run_id
        self.value_start = 0
        self.time = time

    def make_table(self, output: str, drop_level: float = 0.05, max_step:Optional[int] = None, format: str = "tex"):
        rows = []
        for file in self.files:
            config = config_util.read_config(str(file.joinpath("config.yml")))
            row = self._process_config(config)
            info = TrainingInformation(str(file))
            drop_train, drop_test, average, min, time = info.log_key_points(drop_level=drop_level, max_step=max_step)

            if self.best:
                row["Best"] = min
            if self.avg:
                row["Avg"] = average
            if self.drop:
                if config["env"].get("dynamic_episode_length", False):
                    row["Drop"] = int(drop_test)
                else:
                    row["Drop"] = int(drop_train)
            if self.run_id:
                row["Run"] = file.name
            if self.time:
                row["Time"] = time

            rows.append(row)

        fieldnames = self._get_fieldnames()
        table_format = ["c"]*len(fieldnames)
        sort_len = len(fieldnames)
        sort_start, fieldnames, table_format = self._extend_fieldnames(fieldnames, table_format)
        table_format.insert(0, "r")  # Index
        self.value_start = sort_start + sort_len + 1
        dataframe = pd.DataFrame(rows, columns=fieldnames)

        dataframe.sort_values(fieldnames[sort_start:sort_start+sort_len], inplace=True, ascending=False, na_position="first")
        dataframe.reset_index(drop=True, inplace=True)
        if format == "tex":

            self._format_frame(dataframe, fieldnames[sort_start+sort_len:], ["min"]*(len(fieldnames)-sort_start+sort_len))
            self._to_latex(output, dataframe, table_format, fieldnames)
        elif format == "csv":
            dataframe.to_csv(output, index_label="Idx")

    def _extend_fieldnames(self, fieldnames: List[str], table_format: List[str]) -> Tuple[int, List[str], List[str]]:
        sort_start = 0
        if self.best:
            table_format.append("r")
            fieldnames.append("Best")
        if self.avg:
            table_format.append("r")
            fieldnames.append("Avg")
        if self.drop:
            table_format.append("r")
            fieldnames.append("Drop")
        if self.run_id:
            sort_start = 1
            table_format.insert(0, "r")
            fieldnames.insert(0, "Run")
        if self.time:
            table_format.append("r")
            fieldnames.append("Time")
        return sort_start, fieldnames, table_format

    @abstractmethod
    def _process_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        pass

    @abstractmethod
    def _get_fieldnames(self) -> List[str]:
        pass

    def _get_header(self) -> Optional[List[Tuple[str, int, int]]]:
        if self.avg and self.best:
            return [("Episode Length", self.value_start+1, self.value_start+2)]
        elif self.avg or self.best:
            return [("Episode Length", self.value_start+1, self.value_start+1)]
        return []

    def _to_latex(self, output: str, frame: pd.DataFrame, table_format, fieldnames):
        header = self._get_header()
        writer = LatexTableWriter(indent=4)
        writer.begin_table("htp")
        writer.begin_center()
        writer.begin_tabular("".join(table_format))
        writer.add_toprule()
        if header:
            writer.add_spanned_header(header)
        fieldnames.insert(0, "Idx")
        writer.add_dense_header(fieldnames)
        writer.add_midrule()

        for i, row in frame.iterrows():
            values = ["" if pd.isna(x) else str(x) for x in row.values]
            values.insert(0, str(i+1))
            writer.add_row(values)
        writer.add_bottomrule()
        writer.end_tabular()
        writer.end_center()
        writer.end_table()
        writer.write(output)

    def _format_frame(self, frame: pd.DataFrame, value_cols, max_min):
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
            if col == "Time":
                frame[col] = frame[col].apply(lambda x: str(timedelta(seconds=int(x))) + "h")
            else:
                if frame[col].dtype == np.int64:
                    frame[col] = frame[col].apply(human_format)
                else:
                    frame[col] = frame[col].apply(lambda x: "{:.2f}".format(x))

        # Make best values bold
        for row, col in zip(bold_positions, value_cols):
            frame.at[row, col] = "\\textbf{" + frame[col][row] + "}"

        return frame

    @staticmethod
    def make_generator(type, **kwargs):
        if type == "reward":
            return RewardTableGenerator(**kwargs)
        elif type == "observation_size":
            return ObservationSizeTableGenerator(**kwargs)
        else:
            raise ValueError("Unknown table type {}.".format(type))


class ObservationSizeTableGenerator(TableGenerator):
    def _process_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        data = {}
        wrappers = config['env']['wrappers']
        found_wrapper = False
        for wrapper in wrappers:
            if isinstance(wrapper, dict):
                if 'WarpGrayscaleFrame' in list(wrapper.keys())[0]:
                    data['Frame Size'] = "({width}, {height})".format(**wrapper['env.wrappers.WarpGrayscaleFrame'])
                    found_wrapper = True
                elif 'NoObsWrapper' in list(wrapper.keys())[0]:
                    data['Frame Size'] = "-"
                    found_wrapper = True
            else:
                if 'WarpGrayscaleFrame' in wrapper:
                    data['Frame Size'] = "(84, 84)"
                    found_wrapper = True
                elif 'NoObsWrapper' in wrapper:
                    data['Frame Size'] = "-"
                    found_wrapper = True

        if not found_wrapper:
            if 'Maze0318' in config['env']['name']:
                data['Frame Size'] = "(100, 100)"
            elif 'VesselMaze02' in config['env']['name']:
                data['Frame Size'] = "(130, 80)"
            elif 'Maze0122' in config['env']['name']:
                data['Frame Size'] = "(380, 300)"
            else:
                raise ValueError("Unknown env")

        return data

    def _get_fieldnames(self) -> List[str]:
        return ['Frame Size']


class RewardTableGenerator(TableGenerator):
    def __init__(self, files: List[Path]):
        super(RewardTableGenerator, self).__init__(files)
        self.mode = None

    def _get_header(self) -> Optional[List[Tuple[str, int, int]]]:
        stop = 6 if self.mode == "discrete" else 8
        header = [("Reward Component", 2, stop+1)]
        header.extend(super()._get_header())

        return header

    def _process_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        if "Continuous" in config["env"]["name"]:
            self.mode = "continuous"
            return self._process_continuous_config(config)
        elif "Discrete" in config["env"]["name"]:
            self.mode = "discrete"
            return self._process_discrete_config(config)
        else:
            raise ValueError("Unknown env type {}!".format(config["env"]["name"]))

    def _get_fieldnames(self) -> List[str]:
        if self.mode == "continuous":
            return ["ON", "RN", "TP", "DEL", "RND", "GR", "Int Norm", "PO"]
        elif self.mode == "discrete":
            return ["ON", "RN", "TP", "DEL", "RND", "GR"]
        else:
            raise ValueError("Unknown env type!")

    def _process_continuous_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        file_data = self._process_discrete_config(config)
        env_kwargs = config["env"]["reward_kwargs"]
        if env_kwargs.get("positive_only", False):
            file_data["PO"] = "X"
        if env_kwargs.get("normalize", True):
            file_data["Int Norm"] = "X"
        return file_data

    def _process_discrete_config(self, config: Dict[str, Any]) -> Dict[str, str]:
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

        return file_data
