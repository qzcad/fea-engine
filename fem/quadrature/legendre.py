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
                QuadraturePoint([1.0 / 2.0,  1.0 / 2.0], 1.0 / 6.0),
                QuadraturePoint([0.0,        1.0 / 2.0], 1.0 / 6.0),
                QuadraturePoint([1.0 / 2.0,  0.0],       1.0 / 6.0)
            ])
        else:
            p = array([
                QuadraturePoint([1.0 / 6.0,  1.0 / 6.0], 1.0 / 6.0),
                QuadraturePoint([2.0 / 3.0,  1.0 / 6.0], 1.0 / 6.0),
                QuadraturePoint([1.0 / 6.0,  2.0 / 3.0], 1.0 / 6.0)
            ])
        return p


class TetrahedronQuadrature(Quadrature):
    """Gauss-Legendre rules of the unit tetrahedron"""

    def points(self) -> List[QuadraturePoint]:
        if self._order <= 1:
            p = [
                QuadraturePoint([0.25, 0.25, 0.25], 1.0 / 6.0)
            ]
        elif self._order == 2:
            p = [
                QuadraturePoint([(5.0 + 3.0 * sqrt(5.0)) / 20.0, (5.0 - sqrt(5.0)) / 20.0, (5.0 - sqrt(5.0)) / 20.0], 0.25 / 6.0),
                QuadraturePoint([(5.0 - sqrt(5.0)) / 20.0, (5.0 + 3.0 * sqrt(5.0)) / 20.0, (5.0 - sqrt(5.0)) / 20.0], 0.25 / 6.0),
                QuadraturePoint([(5.0 - sqrt(5.0)) / 20.0, (5.0 - sqrt(5.0)) / 20.0, (5.0 + 3.0 * sqrt(5.0)) / 20.0], 0.25 / 6.0),
                QuadraturePoint([(5.0 - sqrt(5.0)) / 20.0, (5.0 - sqrt(5.0)) / 20.0, (5.0 - sqrt(5.0)) / 20.0],       0.25 / 6.0)
            ]
        else:
            p = [
                QuadraturePoint([1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0], -2.0 / 15.0),
                QuadraturePoint([1.0 / 4.0, 1.0 / 6.0, 1.0 / 6.0], 3.0 / 40.0),
                QuadraturePoint([1.0 / 6.0, 1.0 / 4.0, 1.0 / 6.0], 3.0 / 40.0),
                QuadraturePoint([1.0 / 6.0, 1.0 / 6.0, 1.0 / 4.0], 3.0 / 40.0),
                QuadraturePoint([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], 3.0 / 40.0)
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
