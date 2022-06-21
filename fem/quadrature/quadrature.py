from abc import ABC, abstractmethod
from typing import List


class QuadraturePoint:
    """The abstract of a point with a weight"""

    def __init__(self, point: List[float], weight: float):
        self._point = point
        self._weight = weight

    @property
    def point(self):
        """Coordinates of the quadrature point"""
        return self._point

    @property
    def weight(self):
        """The weight value of the quadrature point"""
        return self._weight

    @property
    def xi(self):
        """The first coordinate point of the quadrature point"""
        return self._point[0]

    @property
    def eta(self):
        """The second coordinate point of the quadrature point"""
        return self._point[1]

    @property
    def mu(self):
        """The third coordinate point of the quadrature"""
        return self._point[2]

    @property
    def dimension(self):
        return len(self._point)


class Quadrature(ABC):
    def __init__(self, order: int):
        self._order = order

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, val: int):
        self._order = val

    @abstractmethod
    def points(self) -> List[QuadraturePoint]:
        pass
