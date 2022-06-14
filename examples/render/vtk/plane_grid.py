from mesh.creators.plane_grid import PlaneGridCreator
from render.graphic.vtk.plane import PlaneVtkRenderer

if __name__ == "__main__":
    creator = PlaneGridCreator(0, 0, 10, 20, 4, 8)
    mesh = creator.create()
    renderer = PlaneVtkRenderer("Rectangular Grid", values=[n.x / 10.0 for n in mesh.nodes])
    renderer.render(mesh)
