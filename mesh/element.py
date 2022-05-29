from __future__ import annotations

from typing import List, Tuple

from mesh.node import Node


class Element:
    def __init__(self, nodes: List[Node]):
        self._nodes = nodes

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, n: List[Node]):
        self._nodes = n

    def neighbors(self, node: Node) -> List[Node]:
        """
        The default implementation of neighbors search for the node in the plane element.

        :param node: node from the element
        :return: two nodes are the previous node and the next node (useful for plane elements)
        """
        i = self._nodes.index(node)  # throws ValueError if node isn't in the self._nodes
        return [self._nodes[i - 1], self._nodes[(i + 1) % len(self._nodes)]]

    def edges(self) -> List[Tuple[Node, Node]]:
        """
        The default implementation of the edges list for the node in the plane element.

        :return: The list of pair (the previous node; the next node)
        """
        return [(self._nodes[i - 1], self._nodes[i]) for i in range(len(self._nodes))]

    def reverse(self):
        """
        The method reverses the order of the nodes
        """
        self._nodes = list(reversed(self._nodes))

    def __len__(self) -> int:
        """
        Calculates the number of nodes in the element.

        :return: the number of nodes in the element
        """
        return len(self._nodes)
