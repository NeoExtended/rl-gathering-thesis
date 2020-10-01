from baselines_lab import env
import timeit
import gym
gym.register(id="Maze0318h-v0",
             entry_point="baselines_lab.env.maze0318_huang:Maze0318Env",
             max_episode_steps=2000)
gym.register(id="Maze0122h-v0",
             entry_point="baselines_lab.env.maze0122_huang:Maze0122Env",
             max_episode_steps=2000)
gym.register(id="Maze0518h-v0",
             entry_point="baselines_lab.env.maze0518_huang:Maze0518Env",
             max_episode_steps=2000)

def step(env):
    obs, rew, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()

if __name__ == "__main__":

    print("Maze0318")
    env_huang_0318 = gym.make("Maze0318h-v0")
    env_0318 = gym.make("Maze0318Discrete-v0")
    env_0318.reset()
    env_huang_0318.reset()
    print(timeit.timeit(lambda: step(env_0318), number=100000))
    print(timeit.timeit(lambda: step(env_huang_0318), number=100000))

    print("Maze0518")
    env_huang_0518 = gym.make("Maze0518h-v0")
    env_0518 = gym.make("Maze0518Discrete-v0")
    env_huang_0518.reset()
    env_0518.reset()
    print(timeit.timeit(lambda: step(env_0518), number=100000))
    print(timeit.timeit(lambda: step(env_huang_0518), number=100000))

    print("Maze0122")
    env_huang_0122 = gym.make("Maze0122h-v0")
    env_0122 = gym.make("Maze0122Discrete-v0")
    env_huang_0122.reset()
    env_0122.reset()
    print(timeit.timeit(lambda: step(env_0122), number=100000))
    print(timeit.timeit(lambda: step(env_huang_0122), number=100000))

