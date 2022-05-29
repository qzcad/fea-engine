from typing import List

from mesh.creators.creator import MeshCreator
from mesh.element import Element
from mesh.mesh import Mesh
from mesh.node import NodeType


class SimpleUnion(MeshCreator):
    def __init__(self, meshes: List[Mesh], epsilon: float = 1.0E-8):
        self._meshes = meshes
        self._epsilon = epsilon

    def create(self) -> Mesh:
        final_mesh = Mesh(self._epsilon)
        is_first = True
        for m in self._meshes:
            m.reset_node_id()
            old2new = {}
            for node in m.nodes:
                new_node = final_mesh.append_point(
                    coords=node.coords,
                    node_type=node.node_type,
                    check=(not is_first and (node.node_type == NodeType.BORDER or node.node_type == NodeType.FIXED))
                )
                old2new[node.id] = new_node
            for element in m.elements:
                final_mesh.append_element(Element([old2new[node.id] for node in element.nodes]))
            is_first = False
        return final_mesh
