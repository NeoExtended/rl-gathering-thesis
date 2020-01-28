import gym
import cv2
from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.common.tile_images import tile_images
from utils.recorder import GifRecorder
from env.evaluation import EpisodeInformationAggregator

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


class EvaluationWrapper(gym.Wrapper):
    """
    Evaluation class for goal based environments (reaching max_episode_steps is counted as fail)
    :param env: (gym.Env or gym.Wrapper) The environment to wrap.
    :param path: (str) Path to save the evaluation results to.
    """

    def __init__(self, env, path=None):
        gym.Wrapper.__init__(self, env)
        self.aggregator = EpisodeInformationAggregator(num_envs=1, path=path)

    def step(self, action):
        obs, rew, done, info = gym.Wrapper.step(self, action)
        self.aggregator.step([rew], [done], [info])
        return obs, rew, done, info

    def close(self):
        gym.Wrapper.close(self)
        self.aggregator.close()


class VecEvaluationWrapper(VecEnvWrapper):
    """
    Evaluation class for vectorized goal based environments (reaching max_episode_steps is counted as fail)
    :param env: (VecEnv or VecEnvWrapper) The environment to wrap.
    :param path: (str) Path to save the evaluation results to.
    """

    def __init__(self, env, path=None):
        VecEnvWrapper.__init__(self, env)
        num_envs = env.unwrapped.num_envs
        self.aggregator = EpisodeInformationAggregator(num_envs=num_envs, path=path)

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.aggregator.step(rews, dones, infos)
        return obs, rews, dones, infos

    def close(self):
        VecEnvWrapper.close(self)
        self.aggregator.close()
