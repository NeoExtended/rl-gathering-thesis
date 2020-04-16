import logging
import os

import imageio
import numpy as np

from baselines_lab.utils import get_timestamp


class GifRecorder:
    """
    Records and saves a gif. Gifs are saved to a given directory with a timestamp as name.
    :param path: Output directory for all gifs.
    """
    def __init__(self, path, name_prefix=""):
        self.path = path
        self.images = []
        self.name_prefix = name_prefix

    def record(self, image):
        self.images.append(image)

    def reset(self):
        self.images = []

    def close(self):
        if len(self.images) > 0:
            gif_path = os.path.join(self.path, "{}{}.gif".format(self.name_prefix, get_timestamp()))
            logging.info("Saving gif to {}".format(gif_path))
            imageio.mimsave(gif_path, [np.array(img) for img in self.images], fps=30)
            self.reset()
