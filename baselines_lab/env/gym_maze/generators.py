import math
from abc import ABC, abstractmethod
from collections import deque
from typing import List

import cv2
import numpy as np
from scipy.spatial.distance import cdist


class InstanceGenerator(ABC):
    """
    Base class for random instance generators.
    :param width: (int) Width of the instance.
    :param height: (int) Height of the instance.
    """
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    @abstractmethod
    def generate(self) -> np.ndarray:
        pass


class Node(np.ndarray):
    """
    Helper class to store graph information for the RTTGenerator.
    """
    def __new__(cls, shape, dtype=int, buffer=None, offset=0, strides=None, order=None):
        obj = super(Node, cls).__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.prev = None
        obj.next = []
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.prev = getattr(obj, 'prev', np.array([]))
        self.next = getattr(obj, 'next', [])


class RRTGenerator(InstanceGenerator):
    """
    This generator uses Rapidly-exploring random trees(RRT) to create random instances with blood vessel like shape.
    https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree

    :param width: (int) Instance width
    :param height: (int) Instance height
    :param n_nodes: (int) Number of nodes for the RTT (higher values will result in more vessels)
    :param max_length: (float) Maximum length for a new edge (controls how straight the vessels will be)
    :param n_loops: (int) Number of loops created at the end of the vessels
    :param thickness: (float) Factor for overall vessel thickness
    :param border: (bool) Weather or not to set all border pixels to blocked.
    """

    def __init__(self, width: int, height: int, n_nodes: int = 1000, max_length: float = 10.0, n_loops: int = 10,
                 thickness: float = 1.0, border: bool = True) -> None:
        super(RRTGenerator, self).__init__(width, height)
        self.n_nodes = n_nodes
        self.max_length = max_length
        self.n_loops = n_loops
        self.thickness = thickness
        self.border = border

        if self.border:
            self.width -= 1
            self.height -= 1

    def generate(self) -> np.ndarray:
        nodes = self._generate_tree()
        self._calculate_flow(nodes)
        self._create_loops(nodes)
        maze = self._draw_graph(nodes)
        return self._create_border(maze)

    def _generate_tree(self) -> List[Node]:
        nodes = []
        nodes.append(np.array([np.random.randint(self.height), np.random.randint(self.width)]).view(Node))
        # nodes.append(np.array([imgy/2, imgx/2], dtype=int).view(Node))

        for i in range(self.n_nodes):
            rand = [np.random.randint(self.height), np.random.randint(self.width)]
            dist = cdist(np.atleast_2d(rand), np.atleast_2d(nodes))
            min_dist = np.min(dist)
            min_node_index = np.argmin(dist)
            min_node = nodes[min_node_index]

            if min_dist < self.max_length:
                new_node = np.array(rand).view(Node)
            else:
                theta = math.atan2(rand[1] - min_node[1], rand[0] - min_node[0])
                new_node = np.array([min_node[0] + self.max_length * math.cos(theta), min_node[1] + self.max_length * math.sin(theta)], dtype=int).view(Node)
            new_node.prev = min_node
            min_node.next.append(new_node)
            nodes.append(new_node)

        return nodes

    def _calculate_flow(self, nodes: List[Node]) -> None:
        end_nodes = []
        for node in nodes:
            if len(node.next) == 0:
                end_nodes.append(node)
            node.flow = 0

        current_nodes = deque()
        current_nodes.extend(end_nodes)
        while current_nodes:
            node = current_nodes.pop()
            if node.prev.size > 0:
                node.prev.flow += 1
                current_nodes.append(node.prev)

    def _create_loops(self, nodes: List[Node]) -> None:
        end_nodes = []
        for node in nodes:
            if len(node.next) == 0:
                end_nodes.append(node)

        for i in range(self.n_loops):
            rand = [np.random.randint(self.height), np.random.randint(self.width)]
            dist = cdist(np.atleast_2d(rand), np.atleast_2d(end_nodes))
            first_min = end_nodes[np.argmin(dist)]
            second_min = end_nodes[np.argpartition(dist, 2)[0][2]]

            first_min.next.append(second_min)
            second_min.next.append(first_min)

    def _draw_graph(self, nodes: List[Node]) -> np.ndarray:
        maze = np.zeros((self.height, self.width))

        for node in nodes:
            for next_node in node.next:
                cv2.line(maze, (node[1], node[0]), (next_node[1], next_node[0]), (1), int(max(2.0, np.sqrt(next_node.flow))*self.thickness))
        #structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #maze = cv2.morphologyEx(maze, cv2.MORPH_CLOSE, structure)
        return maze

    def _create_border(self, maze: np.ndarray) -> np.ndarray:
        if self.border:
            return np.pad(maze, pad_width=1, mode='constant', constant_values=0)
        else:
            return maze
