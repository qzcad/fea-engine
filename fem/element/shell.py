from typing import List

import numpy as np

from mesh.node import Node


class ShellElement:
    def __init__(self, nodes: List[Node]):
        self._nodes = nodes
        if len(nodes) < 3:
            raise Exception("At least 3 nodes are necessary to transform element.")
        a = nodes[0].coords
        b = nodes[1].coords
        c = nodes[2].coords
        self._cosine_matrix = self.cosine(a, b, c)
        self._plane_nodes = [
            Node(
                coords=np.dot(self._cosine_matrix, node.coords - a),
                node_type=node.node_type,
                id=node.id
            ) for node in nodes
        ]

    def nodes(self):
        return self._nodes

    def plane_nodes(self):
        return self._plane_nodes

    def cosine_matrix(self):
        return self._cosine_matrix

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray, c: np.ndarray):
        """
            Build the direction cosine matrix for the triangle defined by three vertices

            :param a: Coordinates of the first vertex as `np.array`
            :param b: Coordinates of the second vertex as `np.array`
            :param c: Coordinates of the third vertex as `np.array`
            :return: The direction cosine matrix as `np.array`
            """
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        vx = ab / np.linalg.norm(ab, ord=2)
        vz = n / np.linalg.norm(n, ord=2)
        vy = np.cross(vz, vx)
        # print n, n[0] / norm(n, ord=2), n[1] / norm(n, ord=2), n[2] / norm(n, ord=2), norm(n, ord=2)
        l = np.array([
            [vx[0], vx[1], vx[2]],
            [vy[0], vy[1], vy[2]],
            [vz[0], vz[1], vz[2]]
        ])
        return l
