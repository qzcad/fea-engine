from unittest import TestCase

import numpy as np

from fem.element.quadrilateral import IsoQuad4, IsoQuad8
from fem.quadrature.legendre import QuadrilateralQuadrature
from mesh.node import Node, NodeType


class TestIsoQuad(TestCase):
    def setUp(self) -> None:
        self.q = 1.0
        self.order = 3
        self.w = 0.5
        self.h = 1
        self.s = self.w * self.h
        self.nodes = [
            Node(
                coords=[-0.5 * self.w, -0.5 * self.h],
                node_type=NodeType.BORDER,
                id=0
            ),
            Node(
                coords=[0.5 * self.w, -0.5 * self.h],
                node_type=NodeType.BORDER,
                id=1
            ),
            Node(
                coords=[0.5 * self.w, 0.5 * self.h],
                node_type=NodeType.BORDER,
                id=2
            ),
            Node(
                coords=[-0.5 * self.w, 0.5 * self.h],
                node_type=NodeType.BORDER,
                id=3
            )
        ]

    def test_quad4(self):
        element = IsoQuad4(self.nodes)
        quadrature = QuadrilateralQuadrature(self.order)
        integral = np.zeros(4)
        for point in quadrature.points():
            element.build(point)
            integral = integral + point.weight * self.q * element.shapes() * element.jacobian()
            self.assertAlmostEqual(self.s / 4.0, element.jacobian())
        for i in integral:
            self.assertAlmostEqual(self.q * self.s / 4.0, i)
        self.assertAlmostEqual(self.q * self.s, np.sum(integral))

    def test_quad8(self):
        nodes = [n for n in self.nodes]
        nodes.append(Node(coords=[0.0, -0.5 * self.h], node_type=NodeType.BORDER, id=4))
        nodes.append(Node(coords=[0.5 * self.w, 0.0], node_type=NodeType.BORDER, id=5))
        nodes.append(Node(coords=[0.0, 0.5 * self.h], node_type=NodeType.BORDER, id=6))
        nodes.append(Node(coords=[-0.5 * self.w, 0.0], node_type=NodeType.BORDER, id=7))
        element = IsoQuad8(nodes)
        quadrature = QuadrilateralQuadrature(self.order)
        integral = np.zeros(8)
        for point in quadrature.points():
            element.build(point)
            integral = integral + point.weight * self.q * element.shapes() * element.jacobian()
            self.assertAlmostEqual(self.s / 4.0, element.jacobian())
        self.assertAlmostEqual(self.q * self.s, np.sum(integral))

