import os
import imageio
import numpy as np
import gym
import cv2
from stable_baselines.common.vec_env import VecEnvWrapper
from utils import get_timestamp

class WarpGrayscaleFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Warp grayscale frames to 84x84 as done in the Nature paper and later work.

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class VecGifRecorder(VecEnvWrapper):
    """
    Records videos from the environment in gif format.
    :param env: (VecEnv) Environment to record from.
    :param output_directory: (str) Output directory for the gifs. Individual files will be named with a timestamp
    """
    def __init__(self, env, output_directory):
        VecEnvWrapper.__init__(self, env)
        self.images = []
        self.path = output_directory
        self.timestamp = get_timestamp()

    def reset(self):
        obs = self.venv.reset()
        self.images = []
        self.timestamp = get_timestamp()
        self.images.append(self.venv.render(mode="rgb_array"))
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.images.append(self.venv.render(mode="rgb_array"))
        return obs, rews, dones, infos

    def close(self):
        gif_path = os.path.join(self.path, "{}.gif".format(self.timestamp))
        imageio.mimsave(gif_path, [np.array(img) for img in self.images], fps=30)
        VecEnvWrapper.close(self)
