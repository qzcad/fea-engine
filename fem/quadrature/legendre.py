from math import sqrt
from typing import List

from numpy import array, ndarray

from fem.quadrature.quadrature import Quadrature, QuadraturePoint


class IntervalQuadrature(Quadrature):
    """Gauss-Legendre rules of the interval [-1; 1]"""

    def points(self) -> List[QuadraturePoint]:
        if self._order <= 1:
            p = [
                QuadraturePoint([0.0], 2.0)
            ]
        elif self._order == 2:
            p = [
                QuadraturePoint([-1.0 / sqrt(3.0)], 1.0),
                QuadraturePoint([1.0 / sqrt(3.0)], 1.0)
            ]
        elif self._order == 3:
            p = [
                QuadraturePoint([-sqrt(3.0 / 5.0)], 5.0 / 9.0),
                QuadraturePoint([0.0], 8.0 / 9.0),
                QuadraturePoint([sqrt(3.0 / 5.0)], 5.0 / 9.0)
            ]
        elif self._order == 4:
            p = [
                QuadraturePoint([-sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0)], (18.0 - sqrt(30.0)) / 36.0),
                QuadraturePoint([-sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0)], (18.0 + sqrt(30.0)) / 36.0),
                QuadraturePoint([sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0)],  (18.0 + sqrt(30.0)) / 36.0),
                QuadraturePoint([sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0)],  (18.0 - sqrt(30.0)) / 36.0)
            ]
        else:
            p = [
                QuadraturePoint([-(1.0 / 3.0) * sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0))], (322.0 - 13.0 * sqrt(70.0)) / 900.0),
                QuadraturePoint([-(1.0 / 3.0) * sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0))], (322.0 + 13.0 * sqrt(70.0)) / 900.0),
                QuadraturePoint([0.0], 128.0 / 225.0),
                QuadraturePoint([(1.0 / 3.0) * sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0))], (322.0 + 13.0 * sqrt(70.0)) / 900.0),
                QuadraturePoint([(1.0 / 3.0) * sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0))], (322.0 - 13.0 * sqrt(70.0)) / 900.0)
            ]
        return p


class TriangleQuadrature(Quadrature):
    """Gauss-Legendre rules of the unit triangle"""

    def points(self) -> List[QuadraturePoint]:
        if self._order <= 1:
            p = [
                QuadraturePoint([1.0 / 3.0, 1.0 / 3.0], 1.0 / 2.0)
            ]
        elif self._order == 2:
            p = array([
                QuadraturePoint([1.0 / 6.0, 1.0 / 6.0], 1.0 / 6.0),
                QuadraturePoint([2.0 / 3.0, 1.0 / 6.0], 1.0 / 6.0),
                QuadraturePoint([1.0 / 6.0, 2.0 / 3.0], 1.0 / 6.0)
            ])
        elif self.order == 3:
            p = array([
                QuadraturePoint([1.0 / 3.0, 1.0 / 3.0], -9.0 / 32.0),
                QuadraturePoint([3.0 / 5.0, 1.0 / 5.0], 25.0 / 96.0),
                QuadraturePoint([1.0 / 5.0, 3.0 / 5.0], 25.0 / 96.0),
                QuadraturePoint([1.0 / 5.0, 1.0 / 5.0], 25.0 / 96.0)
            ])
        else:
            p = array([
                QuadraturePoint([0.0, 0.0], 1.0 / 40.0),
                QuadraturePoint([0.5, 0.0], 1.0 / 15.0),
                QuadraturePoint([1.0, 0.0], 1.0 / 40.0),
                QuadraturePoint([0.5, 0.5], 1.0 / 15.0),
                QuadraturePoint([0.0, 1.0], 1.0 / 40.0),
                QuadraturePoint([0.0, 0.5], 1.0 / 15.0),
                QuadraturePoint([1.0 / 3.0, 1.0 / 3.0], 9.0 / 40.0)
            ])
        return p


class TetrahedronQuadrature(Quadrature):
    """Gauss-Legendre rules of the unit tetrahedron"""

    def points(self) -> List[QuadraturePoint]:
        if self._order <= 1:
            p = [
                QuadraturePoint([0.25, 0.25, 0.25], 1.0 / 6.0)
            ]
        elif self.order == 2:
            a = (5.0 + 3.0 * sqrt(5.0)) / 20.0
            b = (5.0 - sqrt(5.0)) / 20.0
            p = [
                QuadraturePoint([a, b, b], 0.25 / 6.0),
                QuadraturePoint([b, a, b], 0.25 / 6.0),
                QuadraturePoint([b, b, a], 0.25 / 6.0),
                QuadraturePoint([b, b, b], 0.25 / 6.0)
            ]
        elif self.order == 3:
            p = [
                QuadraturePoint([1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0], -4.0 / 30.0),
                QuadraturePoint([1.0 / 2.0, 1.0 / 6.0, 1.0 / 6.0], 9.0 / 120.0),
                QuadraturePoint([1.0 / 6.0, 1.0 / 2.0, 1.0 / 6.0], 9.0 / 120.0),
                QuadraturePoint([1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0], 9.0 / 120.0),
                QuadraturePoint([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], 9.0 / 120.0)
            ]
        else:
            a = (1.0 + sqrt(5.0 / 14.0)) / 4.0
            b = (1.0 - sqrt(5.0 / 14.0)) / 4.0
            p = [
                QuadraturePoint([1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0], -74.0 / 5625.0),
                QuadraturePoint([11.0 / 14.0, 1.0 / 14.0, 1.0 / 14.0], 343.0 / 45000.0),
                QuadraturePoint([1.0 / 14.0, 11.0 / 14.0, 1.0 / 14.0], 343.0 / 45000.0),
                QuadraturePoint([1.0 / 14.0, 1.0 / 14.0, 11.0 / 14.0], 343.0 / 45000.0),
                QuadraturePoint([1.0 / 14.0, 1.0 / 14.0, 1.0 / 14.0], 343.0 / 45000.0),
                QuadraturePoint([a, a, b], 56.0 / 2250.0),
                QuadraturePoint([a, b, a], 56.0 / 2250.0),
                QuadraturePoint([b, a, a], 56.0 / 2250.0),
                QuadraturePoint([a, b, b], 56.0 / 2250.0),
                QuadraturePoint([b, a, b], 56.0 / 2250.0),
                QuadraturePoint([b, b, a], 56.0 / 2250.0)
            ]
        return p


class QuadrilateralQuadrature(Quadrature):
    """Gauss-Legendre rules of the quad [-1; 1] x [-1; 1]"""

    def points(self) -> List[QuadraturePoint]:
        interval_quadrature = IntervalQuadrature(self._order)
        interval_points = interval_quadrature.points()
        p = [QuadraturePoint([xi_point.xi, eta_point.xi], weight=xi_point.weight * eta_point.weight)
             for xi_point in interval_points for eta_point in interval_points]
        return p


class HexahedronQuadrature(Quadrature):
    """Gauss-Legendre rules of the hexahedron [-1; 1] x [-1; 1] x [-1; 1]"""

    def points(self) -> List[QuadraturePoint]:
        interval_quadrature = IntervalQuadrature(self._order)
        interval_points = interval_quadrature.points()
        p = [QuadraturePoint([xi_point.xi, eta_point.xi, mu_point.xi], weight=xi_point.weight * eta_point.weight * mu_point.weight)
             for xi_point in interval_points for eta_point in interval_points for mu_point in interval_points]
        return p
