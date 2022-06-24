from abc import ABC, abstractmethod
from typing import List

import numpy as np

from fem.quadrature.quadrature import QuadraturePoint
from mesh.node import Node


class FeaElement(ABC):
    def __init__(self, nodes: List[Node]):
        self._nodes = nodes

    @abstractmethod
    def build(self, point: QuadraturePoint):
        pass

    @abstractmethod
    def jacobian(self) -> float:
        pass

    @abstractmethod
    def shapes(self) -> np.ndarray:
        pass

    @abstractmethod
    def derivatives(self) -> np.ndarray:
        pass

    def nodes(self) -> List[Node]:
        return self._nodes
