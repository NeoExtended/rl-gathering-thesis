import logging

from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import mlp_extractor, nature_cnn
from stable_baselines.common.input import observation_input
from stable_baselines.common import tf_util, tf_layers
from stable_baselines.common.running_mean_std import RunningMeanStd

import tensorflow as tf
import numpy as np

class CuriosityWrapper(VecEnvWrapper):

    def __init__(self, env, network="cnn", intrinsic_reward_weight = 1.0, buffer_size=65536, train_freq=16384, gradient_steps=4, batch_size=4096, learning_starts=100):
        super().__init__(env)


        self.network_type = network

        self.buffer = ReplayBuffer(buffer_size)
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.intrinsic_reward_weight = intrinsic_reward_weight

        # TODO: Parameters
        self.filter_extrinsic_reward = True
        self.clip_rewards = True
        self.clip_rews = 1
        self.clip_obs = 5
        self.norm_obs = True
        self.last_action = None
        self.last_obs = None
        self.steps = 0
        self.last_update = 0
        self.epsilon = 1e-8
        self.gamma = 0.99
        self.int_rwd_rms = RunningMeanStd(shape=(), epsilon=self.epsilon)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret = np.zeros(self.num_envs)

        self._setup_model()

        self.intrinsic_sum = np.zeros(self.num_envs)

    def _setup_model(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf_util.make_session(num_cpu=None, graph=self.graph)

            self.observation_ph, self.processed_obs = observation_input(self.venv.observation_space, scale=(self.network_type == "cnn"))

            with tf.variable_scope("target_model"):
                self.target_network = nature_cnn(self.processed_obs)
                self.target_network = tf_layers.linear(self.target_network, "out", 512)

            with tf.variable_scope("predictor_model"):
                self.predictor_network = nature_cnn(self.processed_obs)
                self.predictor_network = tf_layers.linear(self.predictor_network, "out", 512)

            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.predictor_network, tf.stop_gradient(self.target_network)), axis=1)
                #self.loss = tf.losses.mean_squared_error(labels=self.target_network, predictions=self.predictor_network, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

            learning_rate = 0.01

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                self.training_op = optimizer.minimize(self.loss)
            #tf.variables_initializer().run(self.sess)
            tf.global_variables_initializer().run(session=self.sess)
            #tf.initialize_all_variables().run(session=self.sess)

    def reset(self):
        obs = self.venv.reset()
        self.last_obs = obs
        return obs

    def step_async(self, actions):
        super().step_async(actions)
        self.last_action = actions
        self.steps += self.num_envs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()

        if self.clip_rewards:
            rews = np.clip(rews, -self.clip_rews, self.clip_rews)

        self.buffer.extend(self.last_obs, self.last_action, rews, obs, dones)

        if self.filter_extrinsic_reward:
            rews = np.zeros(rews.shape)

        self.obs_rms.update(obs)
        obs_n = self.normalize_obs(obs)

        loss = self.sess.run([self.loss],{self.observation_ph : obs_n})

        if self.steps > self.train_freq * 10:
            self.update_mean(loss)
            intrinsic_reward = np.array(loss) / np.sqrt(self.int_rwd_rms.var + self.epsilon)
        else:
            intrinsic_reward = np.array(np.zeros(self.num_envs))
        #logging.info(loss)

        self.intrinsic_sum += np.squeeze(intrinsic_reward)

        reward = np.squeeze(rews + self.intrinsic_reward_weight * intrinsic_reward)

        if self.steps > self.learning_starts and self.steps - self.last_update > self.train_freq:
            self.last_update = self.steps
            self.learn()
            #logging.info("{} {}".format(self.intrinsic_sum, np.array(self.intrinsic_sum) / self.train_freq))
            self.intrinsic_sum = np.zeros(self.num_envs)

        return obs, reward, dones, infos

    def close(self):
        VecEnvWrapper.close(self)

    def learn(self):
        #logging.info("Training predictor")

        for _ in range(self.gradient_steps):
            obs_batch, act_batch, rews_batch, next_obs_batch, done_mask = self.buffer.sample(self.batch_size)
            obs_batch = self.normalize_obs(obs_batch)
            self.sess.run(self.training_op, {self.observation_ph : obs_batch})

    def update_mean(self, reward):
        self.ret = self.gamma * self.ret + reward
        self.int_rwd_rms.update(self.ret)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        if self.norm_obs:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs,
                          self.clip_obs)
        return obs
