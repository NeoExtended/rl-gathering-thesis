import gym

from baselines_lab.algorithms.gathering.move_to_extreme_algorithm import MoveToExtremeAlgorithm
from baselines_lab.algorithms.gym_maze_wrapper import GymMazeWrapper

if __name__ == "__main__":
    print("Hello")
    env = gym.make("Maze0318Continuous-v0", n_particles=256)
    env.seed(82)
    env = GymMazeWrapper(env)

    alg = MoveToExtremeAlgorithm(env)
    alg.run()
    print(alg.get_number_of_movements())
    print("Finished")