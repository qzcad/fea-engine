from typing import List

import numpy as np

from fem.element.element import FeaElement
from fem.quadrature.quadrature import QuadraturePoint
from mesh.node import Node


class IsoBeam2(FeaElement):
    """1D 2-nodes isoparametric element for a beam."""
    def __init__(self, nodes: List[Node]):
        super().__init__(nodes)
        self._node_number = len(nodes)
        if self._node_number != 2:
            raise Exception("IsoBeam2 requires 2 nodes to be built.")
        self._jacobian = 0.0
        self._shapes = []
        self._derivatives = []
        self._x = np.array([node.x for node in self._nodes])

    def build(self, point: QuadraturePoint):
        xi = point.xi
        self._shapes = [
            (1.0 - xi) / 2.0,
            (1.0 + xi) / 2.0
        ]  # linear shape functions
        shape_dxi = np.array([-0.5, 0.5])  # derivatives of the shape functions in the first parametric direction
        jacobi = np.sum(shape_dxi * self._x)
        self._jacobian = jacobi  # j = l / 2
        inverted_jacobi = 1.0 / jacobi  # j^-1 = 2 / l
        shape_dx = inverted_jacobi * shape_dxi
        self._derivatives = shape_dx  # [-1 / l; 1 / l]

    def jacobian(self) -> float:
        return self._jacobian

    def shapes(self) -> np.ndarray:
        return np.array(self._shapes)

    def derivatives(self) -> np.ndarray:
        return np.array(self._derivatives)


class IsoBeam3(FeaElement):
    """1D 3-nodes isoparametric element for a beam. Order is following a__(a+b)/2__b: 0 - a, 1 - b, 2 - (a+b)/2"""

    def __init__(self, nodes: List[Node]):
        """
        Create a 3 nodes beam element.

        :param nodes: a list of nodes, node[0] - is the left node (a), node[1] is the right node (b), node[2] - is the middle of the interval [a; b]
        """
        super().__init__(nodes)
        self._node_number = len(nodes)
        if self._node_number != 3:
            raise Exception("IsoBeam3 requires 3 nodes to be built.")
        self._jacobian = 0.0
        self._shapes = []
        self._derivatives = []
        self._x = np.array([node.x for node in self._nodes])

    def build(self, point: QuadraturePoint):
        xi = point.xi
        self._shapes = [
            xi * (xi - 1.0) / 2.0,
            xi * (xi + 1.0) / 2.0,
            1.0 - xi * xi
        ]  # quadratic shape functions
        shape_dxi = np.array([
            xi - 0.5,
            xi + 0.5,
            -2.0 * xi
        ])  # derivatives of the shape functions in the first parametric direction
        jacobi = np.sum(shape_dxi * self._x)
        self._jacobian = jacobi  # j = l / 2
        inverted_jacobi = 1.0 / jacobi  # j^-1 = 2 / l
        shape_dx = inverted_jacobi * shape_dxi
        self._derivatives = shape_dx  # [-1 / l; 1 / l]

    def jacobian(self) -> float:
        return self._jacobian

    def shapes(self) -> np.ndarray:
        return np.array(self._shapes)

    def derivatives(self) -> np.ndarray:
        return np.array(self._derivatives)
