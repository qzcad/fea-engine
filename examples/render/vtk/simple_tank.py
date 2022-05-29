from math import sin, pi, cos, sqrt

from mesh.creators.plane_grid import PlaneGridCreator
from mesh.creators.transfinite import TransfiniteGridCreator
from mesh.creators.union import SimpleUnion
from mesh.node import NodeType
from render.txt.plane import PlaneTextRenderer
from render.vtk.plane import PlaneVtkRenderer


def right_mesh(r: float, n: int):
    def bottom(t: float):
        return r / 2 + t * r / 2, 0.0

    def top(t: float):
        return cos(pi / 4) * r / 2 + t * (r * cos(pi / 4.0) - cos(pi / 4) * r / 2), \
               sin(pi / 4) * r / 2 + t * (r * sin(pi / 4.0) - sin(pi / 4) * r / 2)

    def left(t: float):
        return r / 2 + t * (cos(pi / 4) * r / 2 - r / 2), t * r / 2 * sin(pi / 4)

    def right(t: float):
        phi = t * pi / 4.0
        return r * cos(phi), r * sin(phi)

    creator = TransfiniteGridCreator(top, bottom, left, right, n, n)
    return creator.create()


def central_mesh(r: float, n: int):
    def bottom(t: float):
        return t * r / 2, 0.0

    def top(t: float):
        return t * r / 2 * cos(pi / 4.0), r / 2 + t * (sin(pi / 4) * r / 2 - r / 2)

    def left(t: float):
        return 0, t * r / 2

    def right(t: float):
        return r / 2 + t * (r / 2 * cos(pi / 4.0) - r / 2), t * sin(pi / 4) * r / 2

    creator = TransfiniteGridCreator(top, bottom, left, right, n, n)
    return creator.create()


def top_mesh(r: float, n: int):
    def bottom(t: float):
        return t * cos(pi / 4.0) * r / 2, r / 2 + t * (sin(pi / 4) * r / 2 - r / 2)

    def top(t: float):
        phi = pi / 2.0 + t * (pi / 4.0 - pi / 2.0)
        return r * cos(phi), r * sin(phi)

    def left(t: float):
        return 0, r / 2 + t * r / 2

    def right(t: float):
        return r / 2 * cos(pi / 4.0) + t * (r * cos(pi / 4.0) - r / 2 * cos(pi / 4.0)), \
               r / 2 * sin(pi / 4.0) + t * (r * sin(pi / 4.0) - r / 2 * sin(pi / 4.0))

    creator = TransfiniteGridCreator(top, bottom, left, right, n, n)
    return creator.create()


if __name__ == "__main__":
    r = 3.900 / 2
    R = 2.500
    l1 = 1.767
    o1 = l1 + sqrt(R ** 2 - r ** 2)
    L = 16.105
    l2 = 2.122
    o2 = L - l2 - sqrt(R ** 2 - r ** 2)
    N = 50
    creator = SimpleUnion([central_mesh(r, N), right_mesh(r, N), top_mesh(r, N)])
    bottom = creator.create()
    top = bottom.copy()
    bottom.reverse_elements()
    for node in bottom.nodes:
        coords = node.coords
        z = -sqrt(R ** 2 - node.x ** 2 - node.y ** 2) + o1
        node.coords = [node.x, node.y, z]
    for node in top.nodes:
        coords = node.coords
        z = sqrt(R ** 2 - node.x ** 2 - node.y ** 2) + o2
        node.coords = [node.x, node.y, z]
    creator = PlaneGridCreator(0, l1, 1.0, L - l2 - l1, N * 2 - 1, N * 4)
    central = creator.create()
    for node in central.nodes:
        coords = node.coords
        phi = node.x * pi / 2
        x = r * cos(phi)
        y = r * sin(phi)
        z = node.y
        node.coords = [x, y, z]
    creator = PlaneGridCreator(0, 0, 1.0, l1, N * 2 - 1, N)
    bottom_j = creator.create()
    for node in bottom_j.nodes:
        coords = node.coords
        phi = node.x * pi / 2
        x = r * cos(phi)
        y = r * sin(phi)
        z = node.y
        node.coords = [x, y, z]
    creator = PlaneGridCreator(0, L - l2, 1.0, l2, N * 2 - 1, N)
    top_j = creator.create()
    for node in top_j.nodes:
        coords = node.coords
        phi = node.x * pi / 2
        x = r * cos(phi)
        y = r * sin(phi)
        z = node.y
        node.coords = [x, y, z]
    creator = SimpleUnion([top, top_j, bottom, bottom_j, central])
    mesh = creator.create()
    for node in mesh.nodes:
        node.node_type = NodeType.BORDER
    renderer = PlaneVtkRenderer("Rectangular Grid", values=[mesh.power(n) for n in mesh.nodes])
    renderer.render(mesh)
    text_render = PlaneTextRenderer(f"tank_n{N}.txt")
    text_render.render(mesh)
