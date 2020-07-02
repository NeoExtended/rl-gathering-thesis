import numpy as np
import timeit
from baselines_lab.env.gym_maze.maze_generators import InstanceReader, RRTGenerator


action_map = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1), # {E, SE, S, SW, W, NW, N, NE}
              4: (0, -1), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1)}


def update_locations_vanilla(freespace, locations):
    action = np.random.randint(0, 8)
    new_locations = []
    delta = action_map[action]
    for loc in locations:
        if freespace[loc[0] + delta[0]][loc[1] + delta[1]]:
            new_locations.append([loc[0] + delta[0], loc[1] + delta[1]])
        else:
            new_locations.append(loc)
    return new_locations


def update_locations_numpy(freespace, locations):
    action = np.random.randint(0, 8)
    new_locations = locations + action_map[action]
    valid_locations = freespace[tuple(new_locations.T)]
    new_locations = np.where(valid_locations[:, np.newaxis], new_locations, locations)
    return new_locations


def update_locations_numpy_2(freespace, locations):
    action = np.random.randint(0, 8)
    new_locations = locations + action_map[action]
    valid_locations = freespace.ravel()[(new_locations[:, 1] + new_locations[:, 0] * freespace.shape[1])]
    new_locations = np.where(valid_locations[:, np.newaxis], new_locations, locations)
    return new_locations


if __name__ == "__main__":
    #generator = InstanceReader("../../../../mapdata/map0318.csv")
    generator = RRTGenerator(width=100, height=100)
    generator.seed(42)
    freespace = generator.generate()

    locations = np.transpose(np.nonzero(freespace))
    choice = np.random.choice(len(locations), 64, replace=False)
    particle_locations = locations[choice, :]  # Particle Locations are in y, x (row, column) order

    print("Testing Collision Detection")
    print("Vanilla Time: {:.2f}s".format(timeit.timeit(lambda: update_locations_vanilla(freespace.tolist(), particle_locations.tolist()), number=100000)))
    print("Numpy Simple Time: {:.2f}s".format(timeit.timeit(lambda: update_locations_numpy(freespace, particle_locations), number=100000)))
    print("Numpy Flat Time: {:.2f}s".format(timeit.timeit(lambda: update_locations_numpy_2(freespace, particle_locations), number=100000)))



