from typing import List

import numpy as np

from mesh.node import Node


class PlaneBeamTransformer:
    def __init__(self, nodes: List[Node]):
        self._nodes = nodes
        if len(nodes) < 2:
            raise Exception("At least 2 nodes are necessary to transform element.")
        a = nodes[0].coords
        b = nodes[1].coords
        ab = b - a
        l = np.linalg.norm(ab)
        x = ab[0]
        y = ab[1]
        self._origin = a
        self._transform_matrix = np.array(
            [
                [x / l,  y / l],
                [-y / l, x / l]
            ]
        )
        self._local_nodes = [
            Node(
                coords=self.to_local(node.coords),
                node_type=node.node_type,
                id=node.id
            ) for node in nodes
        ]
        self._inv_transform_matrix = np.linalg.inv(self._transform_matrix)

    def nodes(self):
        return self._nodes

    def local_nodes(self):
        return self._local_nodes

    def transform_matrix(self):
        return self._transform_matrix

    def inv_transform_matrix(self):
        return self._inv_transform_matrix

    def to_local(self, coords: np.ndarray):
        return np.dot(self._transform_matrix, coords - self._origin)

    def to_global(self, coords: np.ndarray):
        return np.dot(self._inv_transform_matrix, coords) + self._origin


class ShellElementTransformer:
    def __init__(self, nodes: List[Node]):
        self._nodes = nodes
        if len(nodes) < 3:
            raise Exception("At least 3 nodes are necessary to transform element.")
        a = nodes[0].coords
        b = nodes[1].coords
        c = nodes[2].coords
        self._transform_matrix = self.cosine(a, b, c)
        self._local_nodes = [
            Node(
                coords=np.dot(self._transform_matrix, node.coords - a),
                node_type=node.node_type,
                id=node.id
            ) for node in nodes
        ]

    def nodes(self):
        return self._nodes

    def local_nodes(self):
        return self._local_nodes

    def transform_matrix(self):
        return self._transform_matrix

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
        cosine_matrix = np.array([
            [vx[0], vx[1], vx[2]],
            [vy[0], vy[1], vy[2]],
            [vz[0], vz[1], vz[2]]
        ])
        return cosine_matrix
