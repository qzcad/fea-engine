from __future__ import annotations

from collections.abc import Iterable
from typing import List, Tuple

import numpy as np
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolygon, vtkPolyData, vtkCellData, vtkPointData
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

from mesh.mesh import Mesh
from render.file.file_renderer import FileRenderer


class VtkXmlRenderer(FileRenderer):
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self._cell_scalars = []
        self._point_scalars = []

    def add_cell_scalar(self, scalar: Iterable[float], name: str):
        array = numpy_support.numpy_to_vtk(np.array(scalar))
        array.SetName(name)
        self._cell_scalars.append(array)

    def add_cell_vector(self, vectors: List[Tuple[float, float, float]], name: str):
        array = vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfComponents(3)
        array.SetNumberOfTuples(len(vectors))
        for i, v in enumerate(vectors):
            array.SetTuple(i, v)
        self._cell_scalars.append(array)

    def clear_cell_data(self):
        self._cell_scalars.clear()

    def add_point_scalar(self, scalar: Iterable[float], name: str):
        array = numpy_support.numpy_to_vtk(np.array(scalar))
        array.SetName(name)
        self._point_scalars.append(array)

    def add_point_vector(self, vectors: List[Tuple[float, float, float]], name: str):
        array = vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfComponents(3)
        array.SetNumberOfTuples(len(vectors))
        for i, v in enumerate(vectors):
            array.SetTuple(i, v)
        self._point_scalars.append(array)

    def clear_point_data(self):
        self._point_scalars.clear()

    def render(self, mesh: Mesh):
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(self._filepath)
        points = vtkPoints()
        for i, node in enumerate(mesh.nodes):
            points.InsertNextPoint([node.x, node.y, node.z])
            node.id = i
        cells_array = vtkCellArray()
        for element in mesh.elements:
            polygon = vtkPolygon()
            nodes_number = len(element)
            polygon.GetPointIds().SetNumberOfIds(nodes_number)
            for i, node in enumerate(element.nodes):
                polygon.GetPointIds().SetId(i, node.id)
            cells_array.InsertNextCell(polygon)
        poly_data = vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(cells_array)
        cell_data = poly_data.GetCellData()  # type: vtkCellData
        point_data = poly_data.GetPointData()  # type: vtkPointData
        for s in self._cell_scalars:
            cell_data.AddArray(s)
        for s in self._point_scalars:
            point_data.AddArray(s)
        writer.SetInputData(poly_data)
        writer.Write()
