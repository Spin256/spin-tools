# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Screen-centered orbit and camera-position-centered look controls."""
import vtk


def center_pivot(renderer, actors):
    origin, size = renderer.GetOrigin(), renderer.GetSize()
    x, y = origin[0] + size[0] * .5, origin[1] + size[1] * .5
    picker = vtk.vtkCellPicker()
    picker.PickFromListOn()
    picker.SetTolerance(.002)
    for actor in actors:
        if actor.GetVisibility():
            picker.AddPickList(actor)
    if picker.Pick(x, y, 0, renderer):
        return picker.GetPickPosition()
    # Empty space: use the screen center at the existing focal depth.
    camera = renderer.GetActiveCamera()
    renderer.SetWorldPoint(*camera.GetFocalPoint(), 1.)
    renderer.WorldToDisplay()
    renderer.SetDisplayPoint(x, y, renderer.GetDisplayPoint()[2])
    renderer.DisplayToWorld()
    point = renderer.GetWorldPoint()
    return tuple(value / point[3] for value in point[:3]) if abs(point[3]) > 1.e-12 else camera.GetFocalPoint()


def turn_camera(camera, mode, dx, dy):
    if mode == 'orbit':
        camera.Azimuth(-dx * .25)
        camera.Elevation(dy * .25)
    else:
        camera.Yaw(-dx * .25)
        camera.Pitch(dy * .25)
    camera.OrthogonalizeViewUp()
