from abc import ABC, abstractmethod

from mesh.mesh import Mesh


class Renderer(ABC):
    @abstractmethod
    def render(self, mesh: Mesh):
        pass
