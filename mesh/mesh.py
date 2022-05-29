from __future__ import annotations
from typing import List, Iterable, Dict

import numpy as np

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

    def power(self, node: Node):
        return len(self._adjacent[node])

    def get_neighbors(self, node: Node) -> List[Node]:
        adjacent = self.get_adjacent(node)
        return list(set(node for element in adjacent for node in element.nodes))

    def reset_node_id(self):
        for i, node in enumerate(self._nodes):
            node.id = i

    def sizes(self):
        x = [self._nodes[0].x, self._nodes[0].x]
        y = [self._nodes[0].y, self._nodes[0].y]
        z = [self._nodes[0].z, self._nodes[0].z]
        for n in self._nodes:
            if n.x < x[0]:
                x[0] = n.x
            if n.x > x[1]:
                x[1] = n.x
            if n.y < y[0]:
                y[0] = n.y
            if n.y > y[1]:
                y[1] = n.y
            if n.z < z[0]:
                z[0] = n.z
            if n.x > x[1]:
                z[1] = n.z
        return x[1] - x[0], y[1] - y[0], z[1] - z[0]

    def origin(self):
        x = self._nodes[0].x
        y = self._nodes[0].y
        z = self._nodes[0].z
        for n in self._nodes:
            if n.x < x:
                x = n.x
            if n.y < y:
                y = n.y
            if n.z < z:
                z = n.z
        return x, y, z

    def mean_edge_length(self):
        lengths = []
        for e in self.elements:
            lengths += [edge[0].to_node(edge[1]) for edge in e.edges()]
        return np.mean(lengths)

    def reverse_elements(self):
        for e in self._elements:
            e.reverse()

    def copy(self) -> Mesh:
        copy_mesh = Mesh()
        nodes_map = {}
        for node in self._nodes:
            new_node = copy_mesh.append_point(coords=node.coords, node_type=node.node_type, check=False)
            nodes_map[node] = new_node
        for element in self._elements:
            copy_mesh.append_element(Element([nodes_map[n] for n in element.nodes]))
        return copy_mesh
