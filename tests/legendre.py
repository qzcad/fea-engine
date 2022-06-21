from unittest import TestCase

from fem.quadrature.legendre import IntervalQuadrature, TriangleQuadrature, TetrahedronQuadrature, \
    QuadrilateralQuadrature, HexahedronQuadrature


class TestLegendre(TestCase):
    def setUp(self) -> None:
        self.max_order = 6

    def test_interval(self):
        for order in range(self.max_order):
            quadrature = IntervalQuadrature(order)
            points = quadrature.points()
            self.assertAlmostEqual(sum(point.weight for point in points), 2.0)
            self.assertAlmostEqual(sum(point.xi * point.weight for point in points), 0.0)

    def test_triangle(self):
        for order in range(self.max_order):
            quadrature = TriangleQuadrature(order)
            points = quadrature.points()
            self.assertAlmostEqual(sum(point.weight for point in points), 0.5)
            self.assertAlmostEqual(sum((point.xi + point.eta) * point.weight for point in points), 1.0 / 3.0)

    def test_tetrahedron(self):
        for order in range(self.max_order):
            quadrature = TetrahedronQuadrature(order)
            points = quadrature.points()
            self.assertAlmostEqual(sum(point.weight for point in points), 1.0 / 6.0)
            self.assertAlmostEqual(sum((point.xi + point.eta + point.mu) * point.weight for point in points), 1.0 / 8.0)

    def test_quadrilateral(self):
        for order in range(self.max_order):
            quadrature = QuadrilateralQuadrature(order)
            points = quadrature.points()
            self.assertAlmostEqual(sum(point.weight for point in points), 4.0)
            self.assertAlmostEqual(sum((point.xi + point.eta) * point.weight for point in points), 0.0)

    def test_hexahedron(self):
        for order in range(self.max_order):
            quadrature = HexahedronQuadrature(order)
            points = quadrature.points()
            self.assertAlmostEqual(sum(point.weight for point in points), 8.0)
            self.assertAlmostEqual(sum((point.xi + point.eta + point.mu) * point.weight for point in points), 0.0)
