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
        self._node_number = len(nodes)
        if self._node_number != 4:
            raise Exception("IsoQuad4 requires 4 nodes to be built.")
        self._jacobian = 0.0
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
        self._derivatives = [shape_dx, shape_dy]

    def jacobian(self) -> float:
        return self._jacobian

    def shapes(self) -> np.ndarray:
        return np.array(self._shapes)

    def derivatives(self) -> np.ndarray:
        return np.array(self._derivatives)


class IsoQuad8(FeaElement):
    """The plane 8-nodes isoparametric element for a quadrilateral."""

    def __init__(self, nodes: List[Node]):
        """
        Create an isoparametric element for a quadrilateral.

        :param nodes: a list of nodes in the counterclockwise direction
        """
        super().__init__(nodes)
        self._node_number = len(nodes)
        if self._node_number != 8:
            raise Exception("IsoQuad8 requires 8 nodes to be built.")
        self._jacobian = 0.0
        self._shapes = []
        self._derivatives = []
        self._x = np.array([node.x for node in self._nodes])
        self._y = np.array([node.y for node in self._nodes])
        self._weights = np.array(
            [
                [-0.25,    0,    0,  0.25,    0.25,    0.25, -0.25, -0.25],
                [-0.25,    0,    0, -0.25,    0.25,    0.25, -0.25,  0.25],
                [-0.25,    0,    0,  0.25,    0.25,    0.25,  0.25,  0.25],
                [-0.25,    0,    0, -0.25,    0.25,    0.25,  0.25, -0.25],
                [0.5,      0, -0.5,     0,    -0.5,       0,   0.5,     0],
                [0.5,    0.5,    0,     0,       0,    -0.5,      0, -0.5],
                [0.5,      0,  0.5,     0,    -0.5,       0,   -0.5,    0],
                [0.5,   -0.5,    0,     0,       0,    -0.5,      0,  0.5]
            ],
            dtype=float
        )

    def build(self, point: QuadraturePoint):
        xi = point.xi
        eta = point.eta
        n = np.array(
            [1, xi, eta, xi * eta, xi * xi, eta * eta, xi * xi * eta, xi * eta * eta],
            dtype=float
        )  # 1,	x,	y,	xy,	x^2,	y^2,	x^2 y,	x y^2
        dndxi = np.array(
            [0, 1, 0, eta, 2.0 * xi, 0, 2.0 * xi * eta, eta * eta],
            dtype=float
        )  # 0,	1,	0,	y,	2x,	0,	2xy,	y^2
        dndeta = np.array(
            [0, 0, 1, xi, 0, 2.0 * eta, xi * xi, 2.0 * xi * eta],
            dtype=float
        )  # 0,	0,	1,	x,	0,	2y,	x^2,	2xy

        self._shapes = np.dot(self._weights, n)  # shape functions
        shape_dxi = np.dot(self._weights, dndxi)  # derivatives of the shape functions in the first parametric direction
        shape_deta = np.dot(self._weights, dndeta)  # derivatives of the shape functions in the second parametric direction
        jacobi = np.array([
            [np.sum(shape_dxi * self._x), np.sum(shape_dxi * self._y)],
            [np.sum(shape_deta * self._x), np.sum(shape_deta * self._y)]
        ])  # Jacobi matrix
        self._jacobian = np.linalg.det(jacobi)
        inverted_jacobi = np.linalg.inv(jacobi)
        shape_dx = inverted_jacobi[0, 0] * shape_dxi + inverted_jacobi[0, 1] * shape_deta
        shape_dy = inverted_jacobi[1, 0] * shape_dxi + inverted_jacobi[1, 1] * shape_deta
        self._derivatives = [shape_dx, shape_dy]

    def jacobian(self) -> float:
        return self._jacobian

    def shapes(self) -> np.ndarray:
        return np.array(self._shapes)

    def derivatives(self) -> np.ndarray:
        return np.array(self._derivatives)
