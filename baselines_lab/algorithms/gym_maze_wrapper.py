import numpy as np

from baselines_lab.env.gym_maze.envs.maze_base import MazeBase


class GymMazeWrapper:
    def __init__(self, env: MazeBase, allow_diagonal: bool=False):
        self.env = env
        self.env.reset()

        if allow_diagonal:
            self._allowed_operations = [(1,0), (-1,0), (0, 1), (0, -1), (-1,-1), (1,1),(-1,1), (1,-1)]
        else:
            self._allowed_operations = [(1,0), (-1,0), (0, 1), (0, -1)]

    def is_valid_position(self, coord):
        row, column = coord
        m = self.env.freespace
        if not 0 <= row < m.shape[0] or not 0 <= column < m.shape[1]:
            return False
        return m[coord] == 1

    def get_neighbored_coords(self, coord):
        nbrs = []
        for op in self.get_operations():
            n = (coord[0] + op[0], coord[1] + op[1])
            if n != coord and self.is_valid_position(n):
                nbrs.append(n)
        return nbrs

    def get_particle_locations(self):
        return self.env.particle_locations

    def get_operations(self):
        return self._allowed_operations

    def size(self):
        return np.sum(self.env.freespace)

    def step(self, action):
        action_number = self.env.rev_action_map[action]
        return self.env.step(action_number)

    @property
    def matrix(self):
        return self.env.freespace

    def simulate_action(self, action):
        dy, dx = action
        particle_locations = np.copy(self.env.particle_locations)

        new_locations = particle_locations + [dx, dy]
        valid_locations = (self.env.freespace.ravel()[(new_locations[:, 1] + new_locations[:, 0] * self.env.freespace.shape[1])]).reshape(-1, 1)  # Border does not need to be checked as long as all maps have borders.
        return np.where(valid_locations, new_locations, particle_locations)