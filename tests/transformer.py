from math import sqrt
from unittest import TestCase

import numpy as np

from fem.element.transformer import PlaneBeamTransformer
from mesh.node import Node, NodeType


class TestIsoQuad(TestCase):
    def test_plane_beam_transformer(self):
        nodes = [
            Node(
                coords=[0, 0],
                node_type=NodeType.BORDER,
                id=0
            ),
            Node(
                coords=[0, 4],
                node_type=NodeType.BORDER,
                id=1
            ),
            Node(
                coords=[0, 2],
                node_type=NodeType.BORDER,
                id=2
            )
        ]
        l = nodes[1].to_point(nodes[0].coords)
        transformer = PlaneBeamTransformer(nodes)
        local_nodes = transformer.local_nodes()
        for node, local in zip(nodes, local_nodes):
            global_coords = transformer.to_global(local.coords)
            print(f"{node} -> {local} -> {global_coords}")
            self.assertAlmostEqual(node.x, global_coords[0])
            self.assertAlmostEqual(node.y, global_coords[1])
        self.assertAlmostEqual(local_nodes[0].x, 0)
        self.assertAlmostEqual(local_nodes[0].y, 0)
        self.assertAlmostEqual(local_nodes[1].x, l)
        self.assertAlmostEqual(local_nodes[1].y, 0)
        self.assertAlmostEqual(local_nodes[2].x, l / 2)
        self.assertAlmostEqual(local_nodes[2].y, 0)
        inv = np.linalg.inv(transformer.transform_matrix())

