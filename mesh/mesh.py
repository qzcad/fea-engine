from typing import List, Iterable, Dict

from mesh.element import Element
from mesh.node import Node, NodeType


class Mesh:
    def __init__(self, epsilon: float = 1.0E-8):
        self._nodes = []  # type: List[Node]
        self._elements = []  # type: List[Element]
        self._adjacent = {}  # type: Dict[Node, List[Element]]
        self._epsilon = epsilon
        self._node_id = 0

    @property
    def nodes(self):
        return self._nodes

    @property
    def elements(self):
        return self._elements

    @property
    def epsilon(self):
        return self.epsilon

    @epsilon.setter
    def epsilon(self, e: float):
        self._epsilon = e

    def append_point(self, coords: Iterable[float], node_type: NodeType, check: bool = True) -> Node:
        if check:
            node = next((n for n in self._nodes if n.to_point(coords) < self._epsilon), None)
            if node is not None:
                return node
        node = Node(coords, node_type, self._node_id)
        self._node_id += 1
        self._nodes.append(node)
        # self._adjacent.append([])
        self._adjacent[node] = []
        return node

    def append_element(self, element: Element):
        self._elements.append(element)
        for node in element.nodes:
            # i = self._nodes.index(node)
            self._adjacent[node].append(element)

    def get_adjacent(self, node: Node) -> List[Element]:
        # i = self._nodes.index(node)
        return self._adjacent[node]

    def get_neighbors(self, node: Node) -> List[Node]:
        adjacent = self.get_adjacent(node)
        return list(set(node for element in adjacent for node in element.nodes))

