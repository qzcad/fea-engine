from mesh.mesh import Mesh
from render.file.file_renderer import FileRenderer


class PlaneTextRenderer(FileRenderer):
    def __init__(self, filepath: str):
        super().__init__(filepath)

    def render(self, mesh: Mesh):
        with open(self._filepath, "w") as text_file:
            mesh.reset_node_id()
            print(3, file=text_file)  # dimension
            element_nodes = set(len(e.nodes) for e in mesh.elements)
            if len(element_nodes) > 1:
                raise Exception("meshes with mixed types elements are not supported by the text renderer")
            print(next(iter(element_nodes)), file=text_file)  # number of nodes in elements
            print(1, file=text_file)  # number of faces per element
            print(len(mesh.nodes), file=text_file)
            for node in mesh.nodes:
                print(f"{node.x} {node.y} {node.z} {node.node_type.value}", file=text_file)
            print(len(mesh.elements), file=text_file)
            print(0, file=text_file)
            for element in mesh.elements:
                ids = [str(n.id) for n in element.nodes]
                print(" ".join(ids), file=text_file)

