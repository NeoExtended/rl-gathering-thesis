from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv, FrameStack
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecFrameStack

from env.gym_maze.envs.MazeBase import MazeBase
import gym

if __name__ == '__main__':
    seed = 42
    set_global_seeds(seed)
    num_env = 4
    start_index = 0

    #env = MazeBase(map_file='../mapdata/map0318.csv', goal=[82, 80], goal_range=10)



    # env = make_atari_env('BreakoutNoFrameskip-v4', num_env=num_env, seed=seed)

    # env = gym.make('maze0318-v0')

    def make_env(rank):
        def _mk():
            # env = gym.make('BreakoutNoFrameskip-v4')
            env = gym.make('Maze0318-v0')  # Creates env with FrameSkip(4) and NoopRandom
            env = MaxAndSkipEnv(env, skip=4)
            env.seed(seed + rank)
            env = FrameStack(env, 4)
            return env
            # return wrap_deepmind(env)

        return _mk


    env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)

    #model = PPO2.load("mazemodel", env=env)
    model = PPO2(CnnPolicy, env, verbose=1, cliprange_vf=-1)
    model.learn(total_timesteps=100000)
    model.save("mazemodel")

    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()

    print("test")
