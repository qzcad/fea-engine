from typing import Tuple

import numpy as np

from mesh.creators.creator import MeshCreator
from mesh.element import Element
from mesh.mesh import Mesh
from mesh.node import NodeType


class PlaneGridCreator(MeshCreator):
    def __init__(self, x: float, y: float, width: float, height: float, num_x: int, num_y: int):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._num_x = num_x
        self._num_y = num_y

    def create(self) -> Mesh:
        x = np.linspace(self._x, self._x + self._width, self._num_x)
        y = np.linspace(self._y, self._y + self._height, self._num_y)
        mesh = Mesh()
        nodes = []
        for i in range(self._num_x):
            row = []
            for j in range(self._num_y):
                node_type = NodeType.BORDER if i == 0 or i == self._num_x - 1 or j == 0 or j == self._num_y - 1 else NodeType.INTERNAl
                row.append(
                    mesh.append_point(coords=(x[i], y[j]), node_type=node_type, check=False)
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


if __name__ == "__main__":
    creator = PlaneGridCreator(0, 0, 1, 2, 600, 1001)
    mesh = creator.create()
    print([len(mesh.get_adjacent(n)) for n in mesh.nodes])
