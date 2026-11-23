# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Gravity aiming and strength panel, sharing the wind direction sphere."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QDoubleSpinBox, QPushButton, QLabel
from wind_controls import DirectionSphere


class GravityPanel(QWidget):
    changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.sphere = DirectionSphere()
        layout.addWidget(self.sphere)
        self.fields = {}
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for key, label, low, high, value, suffix, step in (
                ('heading', 'Horizontal direction', 0, 360, 0, '°', 1),
                ('elevation', 'Vertical direction', -90, 90, -90, '°', 1),
                ('strength', 'Gravity strength', 0, 5, 1, ' g', .1)):
            field = QDoubleSpinBox()
            field.setRange(low, high)
            field.setDecimals(2 if key == 'strength' else 1)
            field.setSingleStep(step)
            field.setSuffix(suffix)
            field.setValue(value)
            self.fields[key] = field
            form.addRow(label, field)
            field.valueChanged.connect(self.publish)
        self.fields['heading'].setWrapping(True)
        layout.addLayout(form)
        self.normal = QPushButton('Restore normal gravity')
        layout.addWidget(self.normal)
        self.normal.clicked.connect(self.reset_gravity)
        self.sphere.aimed.connect(self.aim)
        note = QLabel('Default: downward at 1 g. Zero gives weightlessness; 2 g doubles gravity. Drag the sphere or use numeric angles. Gravity acts uniformly everywhere, with no source position or falloff. Pause to adjust, then resume from the current cloth state.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.publish()

    def settings(self):
        return {key: field.value() for key, field in self.fields.items()}

    def publish(self):
        if len(self.fields) == 3:
            self.sphere.set_angles(self.fields['heading'].value(), self.fields['elevation'].value())
            self.changed.emit(self.settings())

    def aim(self, heading, elevation):
        self.fields['heading'].setValue(heading)
        self.fields['elevation'].setValue(elevation)

    def reset_gravity(self):
        for key, value in (('heading', 0), ('elevation', -90), ('strength', 1)):
            self.fields[key].setValue(value)
