from mesh.creators.plane_grid import PlaneGridCreator
from render.vtk.plane import PlaneRenderer

if __name__ == "__main__":
    creator = PlaneGridCreator(0, 0, 10, 20, 4, 8)
    mesh = creator.create()
    renderer = PlaneRenderer("Rectangular Grid", values=[n.x for n in mesh.nodes])
    renderer.render(mesh)
