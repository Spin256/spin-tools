# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Still-water height, buoyancy and resistance controls plus a preview-only plane."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QCheckBox, QDoubleSpinBox, QLabel
import numpy as np
import vtk
from water_settings import DEFAULT_WATER


class WaterPanel(QWidget):
    changed = Signal(object)

    def __init__(self, garment_cm, parent=None, water_source=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.enabled_box = QCheckBox('Enable water')
        self.guide_box = QCheckBox('Show waterline surface')
        self.guide_box.setChecked(True)
        self.water_source = water_source
        self.export_plane = QCheckBox('Return water-height plane to DAZ')
        self.export_plane.setChecked(water_source is None)
        self.export_plane.setToolTip('Return a separate plain reference prop at the water height. Disabled when a captured Water prop already supplies the surface.')
        layout.addWidget(self.enabled_box)
        layout.addWidget(self.guide_box)
        layout.addWidget(self.export_plane)
        if water_source is not None:
            source_note = QLabel('Detected water surface: ' + ', '.join(water_source['labels']))
            source_note.setWordWrap(True)
            source_note.setToolTip('No duplicate plane is returned. If you change the water height here, move the existing DAZ water prop to match.')
            layout.addWidget(source_note)
        self.fields = {}
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for key, label, low, high, step, suffix in (
                ('height', 'Water height (Y)', -100000, 100000, 1, ' cm'),
                ('buoyancy', 'Buoyancy', 0, 300, 5, ' %'),
                ('drag', 'Submerged drag', 0, 50, .5, ' /s'),
                ('entry', 'Entry resistance', 0, 100, 1, ' /s'),
                ('transition', 'Surface transition depth', .1, 50, .5, ' cm')):
            field = QDoubleSpinBox()
            field.setRange(low, high)
            field.setDecimals(1)
            field.setSingleStep(step)
            field.setSuffix(suffix)
            field.setValue(DEFAULT_WATER[key])
            self.fields[key] = field
            form.addRow(label, field)
        self.fields['height'].setValue(float((garment_cm[:, 1].min() + garment_cm[:, 1].max()) * .5))
        if water_source is not None:
            self.fields['height'].setValue(water_source['height'])
            self.enabled_box.setChecked(True)
        self.export_plane.setEnabled(water_source is None and self.enabled_box.isChecked())
        self.fields['buoyancy'].setToolTip('100% balances gravity when fully submerged; above 100% tends to rise. Acts opposite the configured gravity. Zero gravity also removes buoyancy.')
        self.fields['entry'].setToolTip('Extra resistance to downward motion near the waterline. This is not a solid collision plane.')
        layout.addLayout(form)
        note = QLabel('Water extends below the horizontal waterline. Hair below it is buoyed and slowed; 100% buoyancy is neutral underwater, more makes hair float up. Entry resistance slows hair moving down through the surface. Pause to change settings, then resume.')
        note.setWordWrap(True)
        layout.addWidget(note)
        for field in self.fields.values():
            field.valueChanged.connect(self.publish)
        self.enabled_box.toggled.connect(self.publish)
        self.guide_box.toggled.connect(self.publish)

    def settings(self):
        return dict({key: field.value() for key, field in self.fields.items()},
                    enabled=self.enabled_box.isChecked(), show_surface=self.guide_box.isChecked())

    def publish(self):
        self.export_plane.setEnabled(self.water_source is None and self.enabled_box.isChecked())
        self.changed.emit(self.settings())


class WaterPreview:
    def __init__(self, renderer, garment_cm):
        self.center = (garment_cm.min(axis=0) + garment_cm.max(axis=0)) * .5
        self.extent = max(150., float(np.ptp(garment_cm[:, (0, 2)], axis=0).max()) * 1.5)
        self.plane = vtk.vtkPlaneSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.plane.GetOutputPort())
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetColor(.15, .65, .9)
        self.actor.GetProperty().SetOpacity(.16)
        self.actor.GetProperty().LightingOff()
        self.actor.PickableOff()
        self.actor.SetVisibility(False)
        renderer.AddActor(self.actor)

    def update(self, settings):
        x, z, e, y = self.center[0], self.center[2], self.extent, settings['height']
        self.plane.SetOrigin(x - e, y, z - e)
        self.plane.SetPoint1(x + e, y, z - e)
        self.plane.SetPoint2(x - e, y, z + e)
        self.actor.SetVisibility(bool(settings['enabled'] and settings.get('show_surface', True)))
