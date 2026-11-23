# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Wind source controls and a draggable full-sphere direction widget."""
import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QCheckBox, QDoubleSpinBox, QLabel, QPushButton, QComboBox
from wind_settings import DEFAULT_WIND, wind_direction


class DirectionSphere(QWidget):
    aimed = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.heading, self.elevation, self.drag = 0., 0., None
        self.setMinimumSize(180, 180)
        self.setMaximumHeight(200)
        self.setToolTip('Drag left/right to turn through 360°. Drag up/down to aim upward/downward. Numeric angles below provide fine control.')

    def set_angles(self, heading, elevation):
        self.heading, self.elevation = heading, elevation
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = min(self.width() * .36, (self.height() - 35) * .43)
        center = QPointF(self.width() / 2, (self.height() - 25) / 2)
        gradient = QRadialGradient(center - QPointF(radius * .3, radius * .3), radius * 1.5)
        gradient.setColorAt(0, QColor('#aec9dc'))
        gradient.setColorAt(1, QColor('#34495b'))
        p.setBrush(gradient)
        p.setPen(QPen(QColor('#7e98ad'), 1))
        p.drawEllipse(center, radius, radius)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor('#68859a'), 1))
        p.drawEllipse(center, radius * .42, radius)
        p.drawEllipse(center, radius, radius * .35)
        d = wind_direction(self.heading, self.elevation)
        target = center + QPointF(float(d[0] * .85 + d[2] * .4) * radius,
                                 float(-d[1] * .9 + d[2] * .3) * radius)
        p.setPen(QPen(QColor('#ffbb46'), 3))
        p.drawLine(center, target)
        p.setBrush(QColor('#ffbb46'))
        p.drawEllipse(target, 6, 6)
        p.setPen(self.palette().text().color())
        p.drawText(0, self.height() - 26, self.width(), 22, Qt.AlignmentFlag.AlignCenter,
                   f'{self.heading:.1f}° horizontal · {self.elevation:+.1f}° vertical')

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag = (event.position(), self.heading, self.elevation)
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            start, heading, elevation = self.drag
            delta = event.position() - start
            self.aimed.emit((heading + delta.x() * .8) % 360, max(-90, min(90, elevation - delta.y() * .8)))

    def mouseReleaseEvent(self, event):
        self.drag = None


class WindPanel(QWidget):
    changed = Signal(object)

    def __init__(self, garment_points_cm, parent=None):
        super().__init__(parent)
        self.center = (garment_points_cm.min(axis=0) + garment_points_cm.max(axis=0)) * .5
        self.center_provider = None
        self.fields = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.enabled_box = QCheckBox('Enable wind')
        self.guides_box = QCheckBox('Show wind source and stream')
        self.guides_box.setChecked(True)
        layout.addWidget(self.enabled_box)
        layout.addWidget(self.guides_box)
        self.sphere = DirectionSphere()
        layout.addWidget(self.sphere)
        # Under the ball: while ticked the wind always aims at the center;
        # dragging the ball swings the source around the center at the same distance, moving the
        # source re-aims it, and the direction fields follow (greyed out). Untick to set a free direction.
        self.follow_box = QCheckBox('Keep aimed at center')
        self.follow_box.setChecked(True)
        self.follow_box.setToolTip('On: the wind always blows toward the center. Dragging the ball swings the '
                                   'source around the center; moving the source re-aims it. Off: the ball and '
                                   'direction fields set a free direction (e.g. wind past one side).')
        layout.addWidget(self.follow_box)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        def field(key, label, low, high, step, suffix):
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(1 if key != 'frequency' else 2)
            spin.setSingleStep(step)
            spin.setSuffix(suffix)
            spin.setValue(DEFAULT_WIND[key])
            self.fields[key] = spin
            form.addRow(label, spin)
            return spin
        field('x', 'Source left/right (X)', -100000, 100000, 1, ' cm').setValue(float(self.center[0]))
        field('y', 'Source height (Y)', -100000, 100000, 1, ' cm').setValue(float(self.center[1]))
        field('z', 'Source front/back (Z)', -100000, 100000, 1, ' cm').setValue(float(self.center[2] + 100))
        field('heading', 'Horizontal direction', 0, 360, 1, '°').setWrapping(True)
        field('elevation', 'Vertical direction', -90, 90, 1, '°')
        field('speed', 'Wind speed', 0, 30, .5, ' m/s')
        self.mode = QComboBox()
        self.mode.addItem('Parallel', 'parallel')
        self.mode.addItem('Spreading', 'spreading')
        form.addRow('Wind mode', self.mode)
        field('spread', 'Outward spread angle', 0, 90, 1, '°').setToolTip('Angle away from the centerline. 0° is parallel, 45° is equal axial/outward airflow, and 90° is purely outward.')
        self.axial_box = QCheckBox('Fade with distance from source')
        self.radial_box = QCheckBox('Limit wind to a stream')
        self.radial_box.setChecked(True)
        form.addRow(self.axial_box)
        field('distance', 'Distance to zero wind', 1, 100000, 10, ' cm')
        form.addRow(self.radial_box)
        field('radius', 'Stream radius', 1, 100000, 5, ' cm')
        field('softness', 'Stream edge softness', 0, 100, 5, ' %')
        self.shielding_box = QCheckBox('Avatar and props shelter the hair')
        self.shielding_box.setChecked(True)
        form.addRow(self.shielding_box)
        field('gusts', 'Gust strength', 0, 100, 5, ' %')
        field('frequency', 'Gust frequency', .05, 5, .05, ' Hz')
        field('turbulence', 'Turbulence', 0, 100, 5, ' %')
        layout.addLayout(form)
        aim = QPushButton('Aim at hair center')
        layout.addWidget(aim)
        note = QLabel('Parallel keeps one direction. Spreading tilts airflow outward around the centerline; aim upward under a skirt to lift and spread it. The cyan cone shows flow directions, while the outer stream shows the affected region. Distance fade and radius/softness still apply. In Parallel, disable both for uniform wind. Pause to adjust, then resume.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.auto_aiming = self.batching = False
        self.orbit_distance = None
        for key in ('x', 'y', 'z'):
            self.fields[key].valueChanged.connect(self.source_moved)
        for key in ('heading', 'elevation'):
            self.fields[key].valueChanged.connect(self.direction_set)
        for widget in self.fields.values():
            widget.valueChanged.connect(self.publish)
        self.follow_box.toggled.connect(self.follow_changed)
        self.follow_changed(self.follow_box.isChecked())
        for widget in (self.enabled_box, self.guides_box, self.axial_box, self.radial_box, self.shielding_box):
            widget.toggled.connect(self.publish)
        self.sphere.aimed.connect(self.aim)
        self.mode.currentIndexChanged.connect(self.publish)
        aim.clicked.connect(lambda: self.follow_box.setChecked(True) or self.aim_at_center())
        self.aim_at_center()
        self.publish()

    def settings(self):
        settings = {key: widget.value() for key, widget in self.fields.items()}
        settings.update(enabled=self.enabled_box.isChecked(), show_guides=self.guides_box.isChecked(),
                        axial=self.axial_box.isChecked(), radial=self.radial_box.isChecked(),
                        shielding=self.shielding_box.isChecked(), mode=self.mode.currentData())
        return settings

    def publish(self):
        if self.batching:
            return                    # one update after a compound change (no in-between previews)
        self.fields['spread'].setEnabled(self.mode.currentData() == 'spreading')
        self.fields['distance'].setEnabled(self.axial_box.isChecked())
        for key in ('radius', 'softness'):
            self.fields[key].setEnabled(self.radial_box.isChecked())
        self.sphere.set_angles(self.fields['heading'].value(), self.fields['elevation'].value())
        self.changed.emit(self.settings())

    def batch(self, change):
        """Apply several field changes, then publish once."""
        if self.batching:
            change()
            return
        self.batching = self.auto_aiming = True
        try:
            change()
        finally:
            self.batching = self.auto_aiming = False
        self.publish()

    def set_direction(self, heading, elevation):
        self.fields['heading'].setValue(heading)
        self.fields['elevation'].setValue(elevation)

    def aim(self, heading, elevation):
        """Ball drag. Aimed at the center: the source swings around the center (the box stays ticked).
        Otherwise a free direction; the source stays put."""
        def change():
            self.set_direction(heading, elevation)
            if self.follow_box.isChecked():
                self.orbit_source()
        self.batch(change)

    def source_distance(self):
        source = np.array([self.fields[key].value() for key in ('x', 'y', 'z')])
        return max(float(np.linalg.norm(self.center - source)), 1.)

    def orbit_source(self):
        # Keep the distance from when following started or the source last moved by hand, so
        # rounding of the position fields never makes the source creep in or out while turning.
        if self.orbit_distance is None:
            self.orbit_distance = self.source_distance()
        direction = np.asarray(wind_direction(self.fields['heading'].value(), self.fields['elevation'].value()), float)
        for key, value in zip(('x', 'y', 'z'), self.center - direction * self.orbit_distance):
            self.fields[key].setValue(float(value))

    def source_moved(self, *_):
        if self.auto_aiming:
            return
        self.orbit_distance = None
        if self.follow_box.isChecked():
            self.aim_at_center()

    def direction_set(self, *_):
        """Direction fields are greyed out while aimed at the center; a programmatic change there
        swings the source like the ball does."""
        if not self.auto_aiming and self.follow_box.isChecked():
            self.batch(self.orbit_source)

    def follow_changed(self, on):
        for key in ('heading', 'elevation'):
            self.fields[key].setEnabled(not on)            # they follow the source while aimed
        if on:
            self.orbit_distance = None
            self.aim_at_center()

    def aim_at_center(self):
        if self.center_provider is not None:
            center = np.asarray(self.center_provider(), float)
            if np.linalg.norm(center-self.center) > 1e-6:
                self.orbit_distance = None
            self.center = center
        source = np.array([self.fields[key].value() for key in ('x', 'y', 'z')])
        d = self.center - source
        if np.linalg.norm(d) > 1.e-6:
            self.batch(lambda: self.set_direction(math.degrees(math.atan2(d[0], -d[2])) % 360,
                                                  math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))))

    def display_pose(self, settings, center):
        """Show a moving source without publishing it as a new user edit."""
        self.center = np.asarray(center, float)
        old = self.batching, self.auto_aiming
        self.batching = self.auto_aiming = True
        try:
            for key in ('x','y','z','heading','elevation'):
                self.fields[key].setValue(float(settings[key]))
            self.sphere.set_angles(settings['heading'], settings['elevation'])
        finally:
            self.batching, self.auto_aiming = old
