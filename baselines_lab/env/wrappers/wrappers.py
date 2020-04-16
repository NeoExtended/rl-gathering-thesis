import cv2
import gym
import numpy as np
from stable_baselines.common.tile_images import tile_images
from stable_baselines.common.vec_env import VecEnvWrapper

from baselines_lab.utils.recorder import GifRecorder


class WarpGrayscaleFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """
        Warp grayscale frames to 84x84 as done in the Nature paper and later work.

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
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
    :param record_obs: (bool) If true the recorder records observations instead of the rgb_array output of the env.
    """
    def __init__(self, env, output_directory, record_obs=False):
        VecEnvWrapper.__init__(self, env)
        if record_obs:
            self.recorder = GifRecorder(path=output_directory, name_prefix="obs_")
        else:
            self.recorder = GifRecorder(path=output_directory)
        self.record_obs = record_obs

    def reset(self):
        obs = self.venv.reset()
        self.recorder.reset()
        self._record(obs)
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self._record(obs)
        return obs, rews, dones, infos

    def close(self):
        self.recorder.close()
        VecEnvWrapper.close(self)

    def _record(self, obs):
        if self.record_obs:
            self.recorder.record(tile_images(obs))
        else:
            self.recorder.record(self.venv.render(mode="rgb_array"))


class VecScaledFloatFrame(VecEnvWrapper):
    """
    Scales image observations to [0.0, 1.0]. May be less memory efficient due to float conversion.
    """
    def __init__(self, env, dtype=np.float16):
        self.dtype= dtype
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=env.observation_space.shape, dtype=dtype)
        VecEnvWrapper.__init__(self, env, observation_space=self.observation_space)

    def reset(self):
        obs = self.venv.reset()
        return self._scale_obs(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self._scale_obs(obs), rews, dones, infos

    def _scale_obs(self, obs):
        return np.array(obs).astype(self.dtype) / 255.0