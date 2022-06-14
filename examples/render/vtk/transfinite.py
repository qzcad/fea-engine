from math import sin, pi, cos

from mesh.creators.transfinite import TransfiniteGridCreator
from render.graphic.vtk.plane import PlaneVtkRenderer

if __name__ == "__main__":
    R = 1.0

    def bottom(t: float):
        return R / 2 + t * R / 2, 0.0

    def top(t: float):
        return cos(pi / 4) * R / 2 + t * (R * cos(pi / 4.0) - cos(pi / 4) * R / 2), sin(pi / 4) * R / 2 + t * (R * sin(pi / 4.0) - sin(pi / 4) * R / 2)

    def left(t: float):
        return R / 2 + t * (cos(pi / 4) * R / 2 - R / 2), t * R / 2 * sin(pi / 4)

    def right(t: float):
        phi = t * pi / 4.0
        return R * cos(phi), R * sin(phi)

    creator = TransfiniteGridCreator(top, bottom, left, right, 25, 25)
    mesh = creator.create()
    renderer = PlaneVtkRenderer("Rectangular Grid", values=[n.x for n in mesh.nodes])
    renderer.render(mesh)
