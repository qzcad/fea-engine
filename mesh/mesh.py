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
            node = next((n for n in self._nodes if np.allclose(n.coords, coords)), None)
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
        """
        Return the list of adjacent elements. The list is composed by all elements that include the node.
        If the node isn't in the mesh then return an empty list.

        :param node: the node from the mesh
        :return: a list of elements
        """
        return self._adjacent.get(node, [])

    def power(self, node: Node):
        """
        Calculate the power of the node. A power of a node is a number of elements adjacent in the node.
        If the node isn't in the mesh then return 0.

        :param node: the node from the mesh
        :return: the power of the node
        """
        return len(self.get_adjacent(node))

    def get_moore(self, node: Node) -> List[Node]:
        """
        Return the Moore neighborhood nodes for the node.
        The Moore neighborhood is composed by all nodes from adjacent elements.
        If the node isn't in the mesh then return an empty list.

        :param node: the node from the mesh
        :return: a list of nodes
        """
        adjacent = self.get_adjacent(node)
        return list(set(node for element in adjacent for node in element.nodes))

    def reset_node_id(self):
        """
        Reset IDs of nodes. The method consequently associates numbers from [0; count of nodes) with nodes.
        The order is from the first added node to the last added node.
        """
        for i, node in enumerate(self._nodes):
            node.id = i

    def sizes(self):
        """
        Calculate 3D sizes of the mesh: width is a size along X, height is a size along Y, depth is a size along Z.

        :return: width, height, depth
        """
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
        """
        Calculate the origin of the mesh in the 3D space. An origin is a point with the minimal X, the minimal Y, the minimal Z

        :return: the minimal X, the minimal Y, the minimal Z
        """
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
        """
        Calculate the mean length of an edge in the mesh.

        :return: the mean length of an edge
        """
        return np.mean([edge[0].to_node(edge[1]) for e in self.elements for edge in e.edges()])

    def reverse_elements(self):
        """
        Reverse all elements in the mesh.
        """
        for e in self._elements:
            e.reverse()

    def copy(self) -> Mesh:
        """
        Clone the mesh. The clone of the mesh is the mesh with the same coordinates of nodes and with the same topology of elements.

        :return: the clone of the mesh
        """
        copy_mesh = Mesh()
        nodes_map = {}
        for node in self._nodes:
            new_node = copy_mesh.append_point(coords=node.coords, node_type=node.node_type, check=False)
            nodes_map[node] = new_node
        for element in self._elements:
            copy_mesh.append_element(Element([nodes_map[n] for n in element.nodes]))
        return copy_mesh
