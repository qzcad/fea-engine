from unittest import TestCase

import numpy as np

from fem.element.beam import IsoBeam2, IsoBeam3
from fem.quadrature.legendre import IntervalQuadrature
from mesh.node import Node, NodeType


class TestIsoQuad(TestCase):
    def setUp(self) -> None:
        self.q = 1.0
        self.order = 3
        self.w = 1.0
        self.nodes = [
            Node(
                coords=[-0.5 * self.w, 0],
                node_type=NodeType.BORDER,
                id=0
            ),
            Node(
                coords=[0.5 * self.w, 0],
                node_type=NodeType.BORDER,
                id=1
            )
        ]

    def test_beam2(self):
        element = IsoBeam2(self.nodes)
        quadrature = IntervalQuadrature(self.order)
        integral = np.zeros(2)
        for point in quadrature.points():
            element.build(point)
            integral = integral + point.weight * self.q * element.shapes() * element.jacobian()
            self.assertAlmostEqual(self.w / 2.0, element.jacobian())
        for i in integral:
            self.assertAlmostEqual(self.q * self.w / 2.0, i)
        self.assertAlmostEqual(self.q * self.w, np.sum(integral))

    def test_beam3(self):
        nodes = [n for n in self.nodes]
        nodes.append(Node(coords=[0, 0], node_type=NodeType.BORDER, id=2))
        element = IsoBeam3(nodes)
        quadrature = IntervalQuadrature(self.order)
        integral = np.zeros(3)
        for point in quadrature.points():
            element.build(point)
            integral = integral + point.weight * self.q * element.shapes() * element.jacobian()
            self.assertAlmostEqual(self.w / 2.0, element.jacobian())
        print(integral)
        self.assertAlmostEqual(self.q * self.w, np.sum(integral))
