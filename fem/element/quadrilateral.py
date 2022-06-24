from typing import List

import numpy as np

from fem.element.element import FeaElement
from fem.quadrature.quadrature import QuadraturePoint
from mesh.node import Node


class IsoQuad4(FeaElement):
    """The plane 4-nodes isoparametric element for a quadrilateral."""

    def __init__(self, nodes: List[Node]):
        """
        Create an isoparametric element for a quadrilateral.

        :param nodes: a list of nodes in the counterclockwise direction
        """
        super().__init__(nodes)
        if len(nodes) != 4:
            raise Exception("IsoQuad4 requires 4 nodes to be built.")
        self._jacobian = 0.0
        self._node_number = 4
        self._shapes = []
        self._derivatives = []
        self._x = np.array([node.x for node in self._nodes])
        self._y = np.array([node.y for node in self._nodes])

    def build(self, point: QuadraturePoint):
        xi = point.xi
        eta = point.eta
        self._shapes = [
            (1.0 - xi) * (1.0 - eta) / 4.0,
            (1.0 + xi) * (1.0 - eta) / 4.0,
            (1.0 + xi) * (1.0 + eta) / 4.0,
            (1.0 - xi) * (1.0 + eta) / 4.0
        ]  # bilinear shape functions
        shape_dxi = np.array([
            -(1.0 - eta) / 4.0,
            (1.0 - eta) / 4.0,
            (1.0 + eta) / 4.0,
            -(1.0 + eta) / 4.0
        ])  # derivatives of the shape functions in the first parametric direction
        shape_deta = np.array([
            -(1.0 - xi) / 4.0,
            -(1.0 + xi) / 4.0,
            (1.0 + xi) / 4.0,
            (1.0 - xi) / 4.0
        ])  # derivatives of the shape functions in the second parametric direction
        jacobi = np.array([
            [np.sum(shape_dxi * self._x), np.sum(shape_dxi * self._y)],
            [np.sum(shape_deta * self._x), np.sum(shape_deta * self._y)]
        ])  # Jacobi matrix
        self._jacobian = np.linalg.det(jacobi)
        inverted_jacobi = np.linalg.inv(jacobi)
        shape_dx = inverted_jacobi[0, 0] * shape_dxi + inverted_jacobi[0, 1] * shape_deta
        shape_dy = inverted_jacobi[1, 0] * shape_dxi + inverted_jacobi[1, 1] * shape_deta
        self._derivatives = [[shape_dx], [shape_dy]]

    def jacobian(self) -> float:
        return self._jacobian

    def shapes(self) -> np.ndarray:
        return np.array(self._shapes)

    def derivatives(self) -> np.ndarray:
        return np.array(self._derivatives)


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


class IsoQuad4S(IsoQuad4):
    """The surface 4-nodes isoparametric element for a quadrilateral."""

    def __init__(self, nodes: List[Node]):
        super().__init__(nodes)
        a = nodes[0].coords
        b = nodes[1].coords
        c = nodes[2].coords
        self._cosine_matrix = cosine(a, b, c)
        local_coordinates = [self._cosine_matrix.dot(node.coords - a) for node in nodes]  # type: List[np.ndarray]
        self._x = [lc[0] for lc in local_coordinates]
        self._y = [lc[1] for lc in local_coordinates]

    def cosine_matrix(self):
        return self._cosine_matrix
