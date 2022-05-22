from abc import ABC, abstractmethod

from mesh.mesh import Mesh


class MeshCreator(ABC):
    @abstractmethod
    def create(self) -> Mesh:
        pass
