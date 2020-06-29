import cv2
import gym
import numpy as np
from stable_baselines.common.tile_images import tile_images
from stable_baselines.common.vec_env import VecEnvWrapper

from baselines_lab.utils.recorder import GifRecorder, ImageSequenceRecorder


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


class NoObsWrapper(gym.Wrapper):
    def __init__(self, env, rew_is_obs=False):
        """
        Deletes the observation from the env and replaces it with a counter which increases with each step.
        """
        gym.Wrapper.__init__(self, env)
        self.counter = np.array([0])
        self.rew_is_obs = rew_is_obs
        if rew_is_obs:
            self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64)

    def step(self, action):
        obs, rew, done, info = super(NoObsWrapper, self).step(action)
        self.counter[0] += 1
        if self.rew_is_obs:
            return np.array(self.counter[0], rew), rew, done, info
        else:
            return self.counter, rew, done, info

    def reset(self, **kwargs):
        super(NoObsWrapper, self).reset(**kwargs)
        self.counter[0] = 0

        if self.rew_is_obs:
            return np.array([self.counter[0], 0])
        else:
            return self.counter


class VecImageRecorder(VecEnvWrapper):
    """
    Records videos from the environment in gif format.
    :param env: (VecEnv) Environment to record from.
    :param output_directory: (str) Output directory for the gifs. Individual files will be named with a timestamp
    :param record_obs: (bool) If true the recorder records observations instead of the rgb_array output of the env.
    """
    def __init__(self, env, output_directory, record_obs=False, format: str = "gif", unvec=False, reduction=4):
        VecEnvWrapper.__init__(self, env)
        prefix = "obs_" if record_obs else ""
        self.recorders = []
        self.reduction = reduction
        self.last = reduction
        if unvec:
            for i in range(self.num_envs):
                self.recorders.append(self._create_recorder(output_directory, prefix="{}_{}".format(i, prefix), format=format))
        else:
            self.recorders.append(self._create_recorder(output_directory, prefix, format))

        self.unvec = unvec
        self.record_obs = record_obs
        self.unvec = unvec

    def _create_recorder(self, output_dir, prefix, format):
        if format == "gif":
            recorder = GifRecorder(output_dir, prefix)
        elif format == "png":
            recorder = ImageSequenceRecorder(output_dir, prefix)
        else:
            raise ValueError("Unkown image format {}".format(format))
        return recorder

    def reset(self):
        obs = self.venv.reset()
        for recorder in self.recorders:
            recorder.reset()
        self._record(obs)
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self._record(obs)
        return obs, rews, dones, infos

    def close(self):
        for recorder in self.recorders:
            recorder.close()
        VecEnvWrapper.close(self)

    def _record(self, obs):
        self.last += 1
        if self.last - self.reduction < 0:
            return
        else:
            self.last = 0

        if self.record_obs:
            if self.unvec:
                for i, recorder in enumerate(self.recorders):
                    recorder.record(obs[i])
            else:
                self.recorders[0].record(tile_images(obs))
        else:
            if self.unvec:
                images = self.venv.env_method("render", mode="rgb_array")
                for i, recorder in enumerate(self.recorders):
                    recorder.record(images[i])
            else:
                self.recorders[0].record(self.venv.render(mode="rgb_array"))


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


class VecStepSave(VecEnvWrapper):
    def __init__(self, env):
        super(VecStepSave, self).__init__(env)
        self.last_obs = None
        self.last_rews = None
        self.last_infos = None
        self.last_dones = None

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.last_obs = obs
        self.last_rews = rews
        self.last_dones = dones
        self.last_infos = infos
        return obs, rews, dones, infos