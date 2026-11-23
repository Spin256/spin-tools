# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Non-pickable wind source, direction arrow and translucent stream guide."""
import numpy as np
import math
import vtk
from wind_settings import wind_direction


class WindPreview:
    def __init__(self, renderer):
        self.renderer = renderer
        self.sphere = vtk.vtkSphereSource()
        self.sphere.SetRadius(3)
        self.sphere.SetThetaResolution(16)
        self.sphere.SetPhiResolution(12)
        self.line = vtk.vtkLineSource()
        self.tube = vtk.vtkTubeFilter()
        self.tube.SetInputConnection(self.line.GetOutputPort())
        self.tube.SetNumberOfSides(32)
        self.arrow = vtk.vtkArrowSource()
        self.actors = []
        for source, opacity in ((self.sphere, 1.), (self.arrow, 1.), (self.tube, .1), (self.line, 1.)):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1., .75, .2)
            actor.GetProperty().SetOpacity(opacity)
            actor.GetProperty().SetLineWidth(2)
            actor.PickableOff()
            actor.SetVisibility(False)
            renderer.AddActor(actor)
            self.actors.append(actor)
        self.spread_mesh = vtk.vtkPolyData()
        self.spread_points = vtk.vtkPoints()
        self.spread_mesh.SetPoints(self.spread_points)
        self.spread_directions = vtk.vtkPolyData()
        self.spread_glyph = vtk.vtkGlyph3D()
        self.spread_glyph.SetInputData(self.spread_directions)
        self.spread_glyph.SetSourceConnection(self.arrow.GetOutputPort())
        self.spread_glyph.SetVectorModeToUseVector()
        self.spread_glyph.SetScaleModeToScaleByVector()
        self.spread_glyph.OrientOn()
        for source, opacity in ((self.spread_mesh, .16), (self.spread_glyph, 1.)):
            mapper = vtk.vtkPolyDataMapper()
            if isinstance(source, vtk.vtkPolyData):
                mapper.SetInputData(source)
            else:
                mapper.SetInputConnection(source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(.3, .9, 1.)
            actor.GetProperty().SetOpacity(opacity)
            actor.PickableOff()
            actor.SetVisibility(False)
            renderer.AddActor(actor)
            self.actors.append(actor)

    def update(self, settings):
        visible = bool(settings['enabled'] and settings.get('show_guides', True))
        for actor in self.actors:
            actor.SetVisibility(visible)
        self.actors[2].SetVisibility(visible and settings['radial'])
        source = np.array([settings[key] for key in ('x', 'y', 'z')], dtype=float)
        direction = wind_direction(settings['heading'], settings['elevation']).astype(float)
        length = settings['distance'] if settings['axial'] else 200.
        endpoint = source + direction * length
        self.sphere.SetCenter(*source)
        self.line.SetPoint1(*source)
        self.line.SetPoint2(*endpoint)
        self.tube.SetRadius(settings['radius'])
        reference = np.array([0., 1., 0.]) if abs(direction[1]) < .9 else np.array([1., 0., 0.])
        side = np.cross(direction, reference)
        side /= np.linalg.norm(side)
        up = np.cross(direction, side)
        matrix = vtk.vtkMatrix4x4()
        for column, axis in enumerate((direction, side, up)):
            for row in range(3):
                matrix.SetElement(row, column, float(axis[row] * min(length, 100.)))
        for row in range(3):
            matrix.SetElement(row, 3, float(source[row]))
        self.actors[1].SetUserMatrix(matrix)
        spreading = settings.get('mode', 'parallel') == 'spreading' and settings.get('spread', 0) > 0
        self.actors[1].SetVisibility(visible and not spreading)
        self.actors[4].SetVisibility(visible and spreading)
        self.actors[5].SetVisibility(visible and spreading)
        if spreading:
            angle = math.radians(settings['spread'])
            sine, cosine = math.sin(angle), math.cos(angle)
            if settings['spread'] == 90:
                cosine = 0.
            slant = min(length, 150.)
            if settings['radial'] and sine > 1.e-8:
                slant = min(slant, settings['radius'] / sine)
            points = vtk.vtkPoints()
            points.InsertNextPoint(*source)
            cells = vtk.vtkCellArray()
            arrow_points = vtk.vtkPoints()
            vectors = vtk.vtkFloatArray()
            vectors.SetNumberOfComponents(3)
            for i in range(48):
                azimuth = 2 * math.pi * i / 48
                outward = side * math.cos(azimuth) + up * math.sin(azimuth)
                flow = direction * cosine + outward * sine
                points.InsertNextPoint(*(source + flow * slant))
                cells.InsertNextCell(3)
                for index in (0, i + 1, (i + 1) % 48 + 1):
                    cells.InsertCellPoint(index)
                if i % 8 == 0:
                    arrow_points.InsertNextPoint(*source)
                    vectors.InsertNextTuple(flow * slant)
            self.spread_points = points
            self.spread_mesh.SetPoints(points)
            self.spread_mesh.SetPolys(cells)
            self.spread_directions.SetPoints(arrow_points)
            self.spread_directions.GetPointData().SetVectors(vectors)
            self.spread_directions.Modified()
