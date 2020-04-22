import random

from algorithms.utils.distance_map import DistanceMap


class PairFinderAlgorithm:
    def __init__(self):
        self._env = None

    def set_environment(self, env):
        self._env = env

    def __call__(self, particles):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__


class MostDistancedPairFinderAlgorithm(PairFinderAlgorithm):

    def _arbitrary_particle(self, particles):
        for i in range(0, particles.shape[0]):
            for j in range(0, particles.shape[1]):
                if particles[i, j] == 1:
                    return (i, j)

    def _max_distanced_particle(self, particles, p):
        dm = DistanceMap(self._env, p)
        particle_coords = ((i, j)
                           for i in range(0, particles.shape[0])
                           for j in range(0, particles.shape[1])
                           if particles[i, j] == 1)
        max_p = max(particle_coords, key=lambda x: dm.distance(x))
        return max_p, dm.distance(max_p)

    def __call__(self, particles):
        a = self._arbitrary_particle(particles)
        b, dist = self._max_distanced_particle(particles, a)
        a = b
        b, dist = self._max_distanced_particle(particles, a)
        return a, b


class RandomPairFinderAlgorithm(PairFinderAlgorithm):

    def __call__(self, particles):
        coords = [(i, j) for i in range(particles.shape[0]) for j in
                  range(particles.shape[1]) if particles[i, j] == 1]
        if len(coords) == 1:
            return coords[0], coords[0]
        assert len(coords) > 0
        pair = random.sample(coords, k=2)
        return pair[0], pair[1]
