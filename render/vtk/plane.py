from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from vtkmodules.vtkCommonCore import vtkPoints, vtkLookupTable, vtkFloatArray, vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolygon, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkMaskPoints, vtkPolyDataNormals, vtkGlyph3D
from vtkmodules.vtkFiltersModeling import vtkBandedPolyDataContourFilter
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkInteractionWidgets import vtkScalarBarWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkActor, \
    vtkPolyDataMapper, vtkSelectVisiblePoints, vtkActor2D
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkRenderingLabel import vtkLabeledDataMapper

from mesh.mesh import Mesh
from render.renderer import Renderer


class PlaneVtkRenderer(Renderer):
    def __init__(
            self,
            window_name: str,
            colors_count: int = 64,
            use_gray: bool = False,
            background=(0.95, 0.95, 0.95),
            show_mesh: bool = True,
            mesh_color=(0.25, 0.25, 0.25),
            show_normals: bool = True,
            show_axes: bool = True,
            values: Iterable[float] = [],
            contours_count: int = 0,
            use_cell_data: bool = False,
            show_labels: bool = True,
            scalarbar_title: str = ""
    ):
        self._colors_count = colors_count
        self._lut = vtkLookupTable()
        self._lut.SetNumberOfTableValues(self._colors_count)
        self._use_gray = use_gray
        self._lut.SetHueRange(0.666666667, 0.0)
        if use_gray:
            self._lut.SetValueRange(0.0, 1.0)
            self._lut.SetSaturationRange(0.0, 0.0)  # no color saturation
            self._lut.SetRampToLinear()
        self._lut.Build()
        self._background = background
        self._renderer = vtkRenderer()
        self._render_window = vtkRenderWindow()
        self._render_window.AddRenderer(self._renderer)
        self._render_window_interactor = vtkRenderWindowInteractor()
        self._render_window_interactor.SetRenderWindow(self._render_window)
        self._render_window.SetSize(300, 300)
        self._render_window.SetWindowName(window_name)
        self._render_window_interactor.Initialize()
        self._renderer.SetBackground(background)
        self._bcf_actor = vtkActor()
        self._bcf_mapper = vtkPolyDataMapper()
        self._show_mesh = show_mesh
        self._mesh_color = mesh_color
        self._show_normals = show_normals
        if self._show_mesh:
            self._bcf_actor.GetProperty().EdgeVisibilityOn()
            self._bcf_actor.GetProperty().SetEdgeColor(self._mesh_color)
        self._show_axes = show_axes
        if self._show_axes:
            self._axes_actor = vtkAxesActor()
            self._renderer.AddActor(self._axes_actor)
        else:
            self._axes_actor = None
        self._values = values
        self._contours_count = contours_count
        self._use_cell_data = use_cell_data
        self._show_labels = show_labels
        self._scalarbar_title = scalarbar_title

    def render(self, mesh: Mesh):
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
        if self._values:
            scalars = vtkFloatArray()
            for v in self._values:
                scalars.InsertNextValue(v)
            poly_data.GetPointData().SetScalars(scalars)
            bcf = vtkBandedPolyDataContourFilter()
            bcf.SetInputData(poly_data)
            if self._contours_count > 0:
                bcf.SetNumberOfContours(self._contours_count)
                bcf.GenerateValues(self._contours_count, [min(self._values), max(self._values)])
                bcf.SetNumberOfContours(self._contours_count + 1)
                bcf.GenerateContourEdgesOn()
            bcf.Update()
            # self._bcf_mapper.ImmediateModeRenderingOn()
            self._bcf_mapper.SetInputData(bcf.GetOutput())
            self._bcf_mapper.SetScalarRange(min(self._values), max(self._values))
            self._bcf_mapper.SetLookupTable(self._lut)
            self._bcf_mapper.ScalarVisibilityOn()
            if self._use_cell_data:
                self._bcf_mapper.SetScalarModeToUseCellData()
            self._bcf_actor.SetMapper(self._bcf_mapper)
            self._renderer.AddActor(self._bcf_actor)
            edge_mapper = vtkPolyDataMapper()
            edge_mapper.SetInputData(bcf.GetContourEdgesOutput())
            edge_mapper.SetResolveCoincidentTopologyToPolygonOffset()
            edge_actor = vtkActor()
            edge_actor.SetMapper(edge_mapper)
            if self._use_gray:
                edge_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
            else:
                edge_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
            self._renderer.AddActor(edge_actor)
            if self._show_mesh and self._show_normals:
                # show normals
                # calculate normals
                normals = vtkDoubleArray()
                normals.SetName("normals")
                normals.SetNumberOfComponents(3)
                normals.SetNumberOfTuples(len(mesh.nodes))
                for node_number, node in enumerate(mesh.nodes):
                    elements = mesh.get_adjacent(node)
                    n = np.array((0.0, 0.0, 0.0))
                    for element in elements:
                        neighbors = element.neighbors(node)
                        n_ = np.cross(neighbors[0].vec3d - node.vec3d, neighbors[1].vec3d - node.vec3d)
                        n_ = n_ / np.linalg.norm(n_)
                        n += n_
                    n = n / len(elements)
                    n = n / np.linalg.norm(n)
                    normals.SetTuple(node_number, list(n))
                poly_data.GetPointData().AddArray(normals)
                poly_data.GetPointData().SetActiveVectors("normals")
                # normals = vtkPolyDataNormals()
                # normals.SetInputData(poly_data)
                arrow = vtkArrowSource()
                glyph = vtkGlyph3D()
                # glyph.SetInputConnection(normals.GetOutputPort())
                glyph.SetInputData(poly_data)
                glyph.SetSourceConnection(arrow.GetOutputPort())
                # glyph.SetVectorModeToUseNormal()
                glyph.SetScaleModeToScaleByVector()
                glyph.SetScaleFactor(mesh.mean_edge_length() / 2.0)
                glyph.OrientOn()
                glyph.Update()
                mapper2 = vtkPolyDataMapper()
                mapper2.SetInputConnection(glyph.GetOutputPort())
                actor2 = vtkActor()
                actor2.SetMapper(mapper2)
                actor2.GetProperty().SetColor(self._mesh_color)
                self._renderer.AddActor(actor2)
            if self._show_labels:
                # show labels
                mask = vtkMaskPoints()
                mask.SetInputData(bcf.GetOutput())
                mask.SetOnRatio(
                    round(bcf.GetOutput().GetNumberOfPoints() / 20) if bcf.GetOutput().GetNumberOfPoints() > 20 else 1
                )
                # mask.SetMaximumNumberOfPoints(20)
                # Create labels for points - only show visible points
                visible_points = vtkSelectVisiblePoints()
                visible_points.SetInputConnection(mask.GetOutputPort())
                visible_points.SetRenderer(self._renderer)
                ldm = vtkLabeledDataMapper()
                ldm.SetInputConnection(mask.GetOutputPort())
                ldm.SetLabelFormat("%.2f")
                ldm.SetLabelModeToLabelScalars()
                text_property = ldm.GetLabelTextProperty()
                text_property.SetFontFamilyToArial()
                text_property.SetFontSize(8)
                if self._use_gray:
                    text_property.SetColor(0.0, 1.0, 0.0)
                else:
                    text_property.SetColor(0.0, 0.0, 0.0)
                text_property.ShadowOff()
                text_property.BoldOff()
                contour_labels = vtkActor2D()
                contour_labels.SetMapper(ldm)
                self._renderer.AddActor(contour_labels)
            scalar_bar = vtkScalarBarActor()
            scalar_bar.SetOrientationToHorizontal()
            scalar_bar.SetLookupTable(self._lut)
            if self._scalarbar_title:
                scalar_bar.SetTitle(self._scalarbar_title)
            scalar_bar_widget = vtkScalarBarWidget()
            scalar_bar_widget.SetInteractor(self._render_window_interactor)
            scalar_bar_widget.SetScalarBarActor(scalar_bar)
            scalar_bar_widget.On()
        else:
            self._bcf_mapper.SetInputData(poly_data)
            self._bcf_actor.SetMapper(self._bcf_mapper)
        self._renderer.AddActor(self._bcf_actor)
        self._render_window.Render()
        self._render_window_interactor.Start()
