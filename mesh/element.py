from __future__ import annotations

from typing import List

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
        The default implementation of neighbors search for the node.

        :param node: node from the element
        :return: two nodes are the previous node and the next node (useful for plane elements
        """
        i = self._nodes.index(node)  # throws ValueError if node isn't in the self._nodes
        return [self._nodes[i - 1], self._nodes[(i + 1) % len(self._nodes)]]

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
