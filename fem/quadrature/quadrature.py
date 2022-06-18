from abc import ABC, abstractmethod
from typing import List


class QuadraturePoint:
    """The abstract of a point with a weight"""

    def __init__(self, point: List[float], weight: float):
        self._point = point
        self._weight = weight

    @property
    def point(self):
        return self._point

    @property
    def weight(self):
        return self._weight

    @property
    def xi(self):
        return self._point[0]

    @property
    def eta(self):
        return self._point[1]

    @property
    def mu(self):
        return self._point[3]

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
