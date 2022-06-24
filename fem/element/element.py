from abc import ABC, abstractmethod
from typing import List

import numpy as np

from fem.quadrature.quadrature import QuadraturePoint
from mesh.element import Element
from mesh.mesh import Mesh
from mesh.node import Node


class FeaElement(ABC):
    def __init__(self, mesh: Mesh, element: Element):
        self._mesh = mesh
        self._element = element

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

    @abstractmethod
    def nodes(self) -> List[Node]:
        pass
