from typing import List

import numpy as np

from fem.element.element import FeaElement
from fem.quadrature.quadrature import QuadraturePoint
from mesh.element import Element
from mesh.mesh import Mesh
from mesh.node import Node


class IsoQuad4(FeaElement):
    def __init__(self, mesh: Mesh, element: Element):
        super().__init__(mesh, element)
        self._jacobian = 0.0
        self._node_number = 4
        self._shapes = []
        self._derivatives = []

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
        x = np.array([node.x for node in self._element.nodes])
        y = np.array([node.y for node in self._element.nodes])
        jacobi = np.array([
            [np.sum(shape_dxi * x), np.sum(shape_dxi * y)],
            [np.sum(shape_deta * x), np.sum(shape_deta * y)]
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

    def nodes(self) -> List[Node]:
        return self._element.nodes
