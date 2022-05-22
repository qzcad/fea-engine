from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from scipy.spatial import distance

import numpy as np


class NodeType(Enum):
    UNDEFINED = -2
    EXTERNAL = -1
    BORDER = 0
    INTERNAl = 1
    FIXED = 2


class Node:
    def __init__(self, coords: Iterable[float], node_type: NodeType, id: int):
        self._coords = np.array(coords, dtype=float)  # type: np.ndarray
        self._node_type = node_type
        self._id = id

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @coords.setter
    def coords(self, c: Iterable[float]):
        self._coords = np.array(c, dtype=float)

    @property
    def node_type(self):
        return self._node_type

    @node_type.setter
    def node_type(self, nt: NodeType):
        self._node_type = nt

    @property
    def x(self) -> float:
        return self._coords[0] if len(self._coords) > 0 else 0.0

    @x.setter
    def x(self, v):
        if len(self._coords) > 0:
            self._coords[0] = v

    @property
    def y(self) -> float:
        return self._coords[1] if len(self._coords) > 1 else 0.0

    @y.setter
    def y(self, v):
        if len(self._coords) > 1:
            self._coords[1] = v

    @property
    def z(self) -> float:
        return self._coords[2] if len(self._coords) > 2 else 0.0

    @z.setter
    def z(self, v):
        if len(self._coords) > 2:
            self._coords[2] = v

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, i: int):
        self._id = i

    def to_node(self, node: Node):
        return distance.euclidean(self._coords, node._coords)

    def to_point(self, coords: Iterable[float]):
        return distance.euclidean(self._coords, coords)

    def vector(self, to_node: Node):
        return to_node._coords - self._coords

    def __str__(self):
        return f"{str(self._coords)}-{self._node_type}"
