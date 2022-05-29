from typing import Callable, Union, List, Tuple, Iterable

import numpy as np

from mesh.creators.creator import MeshCreator
from mesh.element import Element
from mesh.mesh import Mesh
from mesh.node import NodeType


class TransfiniteGridCreator(MeshCreator):
    def __init__(
            self,
            top: Callable[[float], Iterable[float]],
            bottom: Callable[[float], Iterable[float]],
            left: Callable[[float], Iterable[float]],
            right: Callable[[float], Iterable[float]],
            num_x: int,
            num_y: int
    ):
        self._top = top
        self._bottom = bottom
        self._left = left
        self._right = right
        self._num_x = num_x
        self._num_y = num_y

    def create(self) -> Mesh:
        xi_values = np.linspace(0.0, 1.0, self._num_x)
        eta_values = np.linspace(0.0, 1.0, self._num_y)
        rb0 = np.array(self._bottom(0.0))
        rb1 = np.array(self._bottom(1.0))
        rt0 = np.array(self._top(0.0))
        rt1 = np.array(self._top(1.0))
        mesh = Mesh()
        nodes = []
        for i, xi in enumerate(xi_values):
            row = []
            rt = np.array(self._top(xi))
            rb = np.array(self._bottom(xi))
            for j, eta in enumerate(eta_values):
                rl = np.array(self._left(eta))
                rr = np.array(self._right(eta))
                p = (1.0 - xi) * rl + xi * rr + (1.0 - eta) * rb + eta * rt - (1.0 - xi) * (1.0 - eta) * rb0 - \
                    (1.0 - xi) * eta * rt0 - xi * (1.0 - eta) * rb1 - xi * eta * rt1
                # if i == 0:
                #     p = rl
                # elif i == self._num_x - 1:
                #     p = rr
                # elif j == 0:
                #     p = rb
                # elif j == self._num_y - 1:
                #     p = rt
                node_type = NodeType.BORDER if i == 0 or i == self._num_x - 1 or j == 0 or j == self._num_y - 1 else NodeType.INTERNAl
                row.append(
                    mesh.append_point(coords=p, node_type=node_type, check=False)
                )
            nodes.append(row)
        for i in range(self._num_x - 1):
            for j in range(self._num_y - 1):
                mesh.append_element(
                    Element(
                        [
                            nodes[i][j],
                            nodes[i + 1][j],
                            nodes[i + 1][j + 1],
                            nodes[i][j + 1]
                        ]
                    )
                )
        return mesh
