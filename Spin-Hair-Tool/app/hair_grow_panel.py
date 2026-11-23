# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""The Hair window's Grow hair section (Spin256 · spin256.com/tools).

Named areas painted on the avatar (polygons), each with its edge, scatter and strand settings; Grow hair adds their
strands to the hair mesh, so the simulation and the return treat them as hair. Areas belong to the capture they were
painted on (reopening the tool on it shows them again); a new export from DAZ starts with none."""
import json
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QWidget)

DEFAULT_MATERIALS = {"Hair": (60, 42, 30), "Gray": (165, 160, 155)}


class MaterialDialog(QDialog):
    """New / edit material: a name and a color."""

    def __init__(self, parent, name, color, taken):
        super().__init__(parent)
        self.setWindowTitle("Hair material")
        self.taken = {t.lower() for t in taken}
        self.name = QLineEdit(name)
        self.color = tuple(color)
        self.swatch = QPushButton()
        self.swatch.clicked.connect(self.pick)
        self.show_swatch()
        form = QFormLayout(self)
        form.addRow("Name", self.name)
        form.addRow("Color", self.swatch)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)
        self.name.textChanged.connect(self.check)
        self.check()

    def check(self, *_):
        name = self.name.text().strip()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(name) and name.lower() not in self.taken)

    def show_swatch(self):
        self.swatch.setText("#%02x%02x%02x" % self.color)
        self.swatch.setStyleSheet("QPushButton { background-color: rgb(%d,%d,%d); color: %s; }"
                                  % (*self.color, "white" if sum(self.color) < 380 else "black"))

    def pick(self):
        chosen = QColorDialog.getColor(QColor(*self.color), self, "Hair color")
        if chosen.isValid():
            self.color = (chosen.red(), chosen.green(), chosen.blue())
            self.show_swatch()

    def result_value(self):
        return self.name.text().strip(), self.color

from hair_grow import GrowArea, grow_area, suggested_segments

NO_AREA = "-none-"
# Named presets of every area: grown hair does not follow a new pose, so it is grown again on
# each capture; a preset brings the areas back (loaded by hand). Saved with the tool (user
# data, git-ignored like hair_presets.json).
PRESET_FILE = Path(__file__).resolve().parent.parent / "grow_presets.json"
PAINT_COLOR = (255, 60, 200)        # painted polygons: one bright color, whatever the area's hair color
PAINT_LIFT = .03                    # cm over the skin
EDGE_LABELS = (("polygon", "Polygon edge"), ("smooth", "Smooth line"), ("circle", "Circle"), ("ellipse", "Ellipse"),
               ("rectangle", "Rectangle"), ("triangle", "Triangle"))
SCATTER_LABELS = (("even", "Even (spaced)"), ("random", "Random"), ("grid", "Regular"))


class GrowPanel(QObject):
    changed = Signal()               # painted areas changed (overlay)
    grow_requested = Signal()

    def __init__(self, layout, settings, avatar_vertices, avatar_triangles, parent=None, capture="none",
                 avatar_label="Avatar", preset_file=None):
        super().__init__(parent)
        self.settings = settings
        self.avatar_label = str(avatar_label)
        self.preset_file = Path(preset_file) if preset_file else PRESET_FILE
        self.avatar_vertices = np.asarray(avatar_vertices, float)
        self.avatar_triangles = np.asarray(avatar_triangles, np.int64)
        self.centres = self.avatar_vertices[self.avatar_triangles].mean(axis=1)
        self.key = f"grow/capture/{capture}"
        self.areas = self.load()
        note = QLabel("Paint polygons of the avatar into named areas, choose how each grows, then Grow hair. The "
                      "strands come out straight from the skin and are simulated with the rest of the hair.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.preset = QComboBox()
        self.save_preset_button = QPushButton("Save preset…")
        self.delete_preset_button = QPushButton("Delete preset")
        preset_form = QFormLayout()
        preset_form.addRow("Preset", self.preset)
        # Apply preset: load the chosen preset and grow all its areas in one step.
        self.apply_preset_button = QPushButton("Apply preset")
        self.apply_preset_button.setToolTip("Load this preset's areas and grow them all.")
        row = QHBoxLayout(); row.addWidget(self.apply_preset_button); row.addWidget(self.save_preset_button)
        row.addWidget(self.delete_preset_button)
        preset_form.addRow(row)
        layout.addLayout(preset_form)
        self.area_box = QComboBox()
        self.area_name = QLineEdit()
        self.area_name.setPlaceholderText("e.g. Left armpit, Beard")
        self.new_button = QPushButton("New area")
        self.delete_button = QPushButton("Delete area")
        form = QFormLayout()
        form.addRow("Area", self.area_box)
        row = QHBoxLayout(); row.addWidget(self.area_name, 1); row.addWidget(self.new_button)
        form.addRow("New area", row)
        layout.addLayout(form)
        layout.addWidget(self.delete_button)
        self.paint = QPushButton("Paint polygons (left drag; Alt removes)")
        self.paint.setCheckable(True)
        self.brush = QDoubleSpinBox(); self.brush.setRange(.2, 20); self.brush.setSingleStep(.2); self.brush.setSuffix(" cm")
        self.brush.setValue(self.settings.value("grow/brush", 1.5, type=float))
        self.brush.valueChanged.connect(lambda v: self.settings.setValue("grow/brush", v))
        self.clear_paint = QPushButton("Clear area's polygons")
        layout.addWidget(self.paint)
        form = QFormLayout()
        form.addRow("Brush radius", self.brush)
        self.edge = QComboBox()
        for key, label in EDGE_LABELS:
            self.edge.addItem(label, key)
        self.edge.setToolTip("Polygon edge: roots stay on the painted polygons. Smooth line: a smooth edge through the "
                             "boundary polygons. A shape: fitted to the painted area (for compact areas facing one way).")
        self.flip = QCheckBox("Point the other way")
        self.extend = QCheckBox("Allow past the painting")
        self.extend.setToolTip("Off: the shape only trims the painted polygons. On: it may reach nearby skin facing the "
                               "same way, so the whole shape grows.")
        self.feather = self.number(0, 100, 0, 5, " %", "How far the blend from the edge reaches: this share of the way "
                                   "from the edge to the area's center. 0%: a hard edge.")
        # Edge blend: the density and length at the very edge, blending up to the area's own
        # across the feather.
        self.edge_density = self.number(0, 100, 20, 5, " %", "Density at the edge, as a share of the area's density.")
        self.edge_length = self.number(0, 100, 50, 5, " %", "Length at the edge, as a share of the area's length.")
        self.scatter = QComboBox()
        for key, label in SCATTER_LABELS:
            self.scatter.addItem(label, key)
        self.density = self.number(1, 2000, 60, 10, " /cm²", "Strands per square centimeter.")
        self.length = self.number(.05, 150, 1.5, .1, " cm")       # up to 150 cm
        self.variation = self.number(0, 100, 20, 5, " %", "Each strand is up to this much shorter, at random.")
        self.min_length = self.number(.05, 150, .1, .05, " cm", "No strand is shorter than this, whatever the length "
                                      "variation and edge length.")
        self.segments = QSpinBox(); self.segments.setRange(1, 500); self.segments.setValue(4)
        self.segments.setToolTip("Segments along each strand. Long hair needs about one per 2.5 cm to follow the head.")
        self.width = self.number(.02, 2, .15, .01, " mm", "Width of each hair strip at the root; it narrows to the tip.")
        self.stiff = self.number(0, 100, 0, 5, " %", "Share of the simulation's Stiffness these strands keep. Grown "
                                 "hair feels full gravity and is not held in its grown pose; higher keeps it springy.")
        self.root_lift = self.number(0, 99, 50, 5, " %", "How firmly the strands keep leaving the skin straight out "
                                     "before they fall (like a follicle). Gives volume and covers the parting on top "
                                     "of the head. 0%: they lie flat from the root.")
        # Named materials: areas pick a material by name, so editing one recolors every area
        # that uses it. New… / Edit… open a small popup with the name and a color picker.
        self.material = QComboBox(); self.material.setToolTip("The hair's material (its color in DAZ).")
        self.material2 = QComboBox()
        self.material2.setToolTip("A second material some strands take, e.g. gray hints in dark hair or dark patches "
                                  "in blond.")
        self.edit_material = QPushButton("Edit…"); self.edit_material.setToolTip("Rename or recolor this material.")
        self.edit_material2 = QPushButton("Edit…"); self.edit_material2.setToolTip("Rename or recolor this material.")
        self.new_material = QPushButton("New material…")
        self.color, self.color2 = QWidget(), QWidget()           # material rows (layouts below)
        row = QHBoxLayout(self.color); row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.material, 1); row.addWidget(self.edit_material)
        row = QHBoxLayout(self.color2); row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.material2, 1); row.addWidget(self.edit_material2)
        self.mix = self.number(0, 100, 0, 5, " %", "Share of strands in the secondary color, chosen at random. 0%: one "
                               "color (one material in DAZ); otherwise two materials.")
        for label, widget in (("Edge", self.edge), ("", self.flip), ("", self.extend), ("Feather", self.feather), ("Edge density", self.edge_density),
                              ("Edge length", self.edge_length), ("Scatter", self.scatter),
                              ("Density", self.density), ("Length", self.length), ("Length variation", self.variation),
                              ("Minimum length", self.min_length),
                              ("Segments", self.segments), ("Width", self.width), ("Stiffness", self.stiff),
                              ("Root lift", self.root_lift),
                              ("Material", self.color), ("Secondary", self.color2), ("Secondary mix", self.mix),
                              ("", self.new_material)):
            form.addRow(label, widget) if label else form.addRow(widget)
        layout.addLayout(form)
        layout.addWidget(self.clear_paint)
        self.grow_button = QPushButton("Grow all areas")
        self.grow_button.setToolTip("Grow every area, not only the one shown (replacing hair grown before), and add it "
                                    "to the simulated hair.")
        layout.addWidget(self.grow_button)
        # Long segments warning: below the grow button, with a suggested number.
        self.segment_warning = QLabel(); self.segment_warning.setWordWrap(True)
        self.segment_warning.setStyleSheet("color: #d08000;")
        self.segment_warning.hide()
        layout.addWidget(self.segment_warning)
        self.summary = QLabel(); self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.settings_widgets = [self.edge, self.flip, self.extend, self.feather, self.edge_density, self.edge_length,
                                 self.scatter, self.density, self.length,
                                 self.variation, self.min_length, self.segments, self.width, self.stiff, self.root_lift,
                                 self.material,
                                 self.material2,
                                 self.edit_material, self.edit_material2, self.mix,
                                 self.clear_paint, self.paint,
                                 self.delete_button]
        self.new_button.clicked.connect(self.new_area)
        self.new_button.setEnabled(False)                     # until the new area has a name
        self.area_name.textChanged.connect(lambda text: self.new_button.setEnabled(bool(text.strip())))
        self.area_name.returnPressed.connect(lambda: self.new_button.isEnabled() and self.new_area())
        self.delete_button.clicked.connect(self.delete_area)
        self.area_box.currentIndexChanged.connect(lambda *_: self.recall())
        self.clear_paint.clicked.connect(self.clear_area)
        self.material.activated.connect(lambda *_: self.store())
        self.material2.activated.connect(lambda *_: self.store())
        self.edit_material.clicked.connect(lambda: self.edit_named_material(self.material.currentText()))
        self.edit_material2.clicked.connect(lambda: self.edit_named_material(self.material2.currentText()))
        self.new_material.clicked.connect(lambda: self.edit_named_material(None))
        self.grow_button.clicked.connect(self.confirm_grow)
        self.preset.activated.connect(lambda *_: self.load_preset(self.preset.currentText()))
        self.apply_preset_button.clicked.connect(self.apply_preset)
        self.save_preset_button.clicked.connect(lambda: self.save_preset())
        self.delete_preset_button.clicked.connect(self.delete_preset)
        for widget in (self.edge, self.scatter):
            widget.currentIndexChanged.connect(lambda *_: self.store())
        self.flip.toggled.connect(lambda *_: self.store())
        self.extend.toggled.connect(lambda *_: self.store())
        for widget in (self.feather, self.edge_density, self.edge_length, self.density, self.length, self.variation,
                       self.min_length,
                       self.segments, self.width, self.stiff, self.root_lift,
                       self.mix):
            widget.valueChanged.connect(lambda *_: self.store())
        self.refill_materials()
        self.refill()
        # Presets are loaded or applied by hand.
        self.refill_presets()

    # ----- presets -----------------------------------------------------------------------
    def presets(self):
        try:
            data = json.loads(self.preset_file.read_text(encoding="utf-8"))
            return {str(k): v for k, v in data.items() if isinstance(v, dict) and isinstance(v.get("areas"), list)}
        except (OSError, ValueError, AttributeError):
            return {}

    def write_presets(self, presets):
        temporary = self.preset_file.with_name(self.preset_file.name + ".tmp")
        temporary.write_text(json.dumps(presets, indent=1), encoding="utf-8")
        temporary.replace(self.preset_file)

    def refill_presets(self, select=None):
        self.preset.blockSignals(True)
        self.preset.clear()
        self.preset.addItem(NO_AREA)
        for name in sorted(self.presets(), key=str.lower):
            self.preset.addItem(name)
        self.preset.setCurrentIndex(max(0, self.preset.findText(select)) if select else 0)
        self.preset.blockSignals(False)
        self.update_preset_buttons()

    def update_preset_buttons(self):
        name = self.preset.currentText()
        preset = self.presets().get(name) if name != NO_AREA else None
        self.delete_preset_button.setEnabled(preset is not None)
        self.apply_preset_button.setEnabled(preset is not None)

    def save_preset(self, name=None, ask=True):
        """Save every area (painted polygons and settings) under a name; returns the name or None."""
        if not any(area.faces for area in self.areas):
            self.summary.setText("Paint an area before saving a preset.")
            return None
        if name is None:
            current = self.preset.currentText()
            name, ok = QInputDialog.getText(self.save_preset_button, "Save grow preset", "Preset name:",
                                            text=current if current != NO_AREA else self.avatar_label)
            if not ok:
                return None
        name = str(name).strip()
        if not name or name == NO_AREA:
            self.summary.setText("Give the preset a name.")
            return None
        presets = self.presets()
        # Saving over an existing preset asks first.
        if ask and name in presets and QMessageBox.question(
                self.save_preset_button, "Save grow preset", f"A preset named “{name}” already exists. Replace it?")                 != QMessageBox.StandardButton.Yes:
            self.summary.setText(f"Preset “{name}” was not replaced.")
            return None
        presets[name] = dict(vertex_count=len(self.avatar_vertices),
                             areas=[area.to_json() for area in self.areas],
                             materials={k: list(v) for k, v in self.materials.items()})
        self.write_presets(presets)
        self.refill_presets(name)
        self.summary.setText(f"Saved preset “{name}” ({len(self.areas)} area(s)).")
        return name

    def load_preset(self, name, ask=True):
        """Replace this capture's areas with a preset's (only on a figure with the same mesh)."""
        preset = self.presets().get(name)
        if preset is None:
            self.update_preset_buttons()
            return False
        if preset.get("vertex_count") != len(self.avatar_vertices):
            self.summary.setText(f"“{name}” was painted on a figure with a different mesh "
                                 f"({preset.get('vertex_count')} vertices; this one has {len(self.avatar_vertices)}).")
            self.refill_presets()
            return False
        if ask and any(area.faces for area in self.areas) and QMessageBox.question(
                self.preset, "Load grow preset", f"Replace this capture's areas with “{name}”?") \
                != QMessageBox.StandardButton.Yes:
            self.refill_presets()
            return False
        # The preset's materials join (and update) this capture's.
        for material, color in (preset.get("materials") or {}).items():
            self.materials[str(material)] = tuple(int(c) for c in color)
        self.areas = [GrowArea.from_json(area) for area in preset["areas"]]
        self.refill_materials()
        self.save()
        self.refill()
        self.update_preset_buttons()
        self.summary.setText(f"Loaded preset “{name}”. Grow hair to grow it.")
        return True

    def apply_preset(self):
        """Load the chosen preset (without asking) and grow all its areas."""
        name = self.preset.currentText()
        if self.load_preset(name, ask=False):
            self.grow_requested.emit()
            return True
        return False

    def delete_preset(self):
        presets = self.presets()
        presets.pop(self.preset.currentText(), None)
        self.write_presets(presets)
        self.refill_presets()

    # ----- areas -------------------------------------------------------------------------
    @staticmethod
    def number(low, high, value, step, suffix, tip=""):
        field = QDoubleSpinBox(); field.setRange(low, high); field.setSingleStep(step); field.setValue(value)
        field.setSuffix(suffix); field.setDecimals(2); field.setToolTip(tip)
        return field

    def load(self):
        """This capture's areas (and its named materials into self.materials)."""
        self.materials = dict(DEFAULT_MATERIALS)
        try:
            data = json.loads(self.settings.value(self.key, "[]"))
            if not isinstance(data, dict):
                return []
            self.materials = {str(k): tuple(int(c) for c in v) for k, v in data.get("materials", {}).items()} \
                or dict(DEFAULT_MATERIALS)
            data = data.get("areas", [])
            areas = [GrowArea.from_json(item) for item in data]
        except (TypeError, ValueError, KeyError, AttributeError):
            return []
        return areas

    def save(self):
        self.settings.setValue(self.key, json.dumps(dict(areas=[area.to_json() for area in self.areas],
                                                         materials={k: list(v) for k, v in self.materials.items()})))

    # ----- named materials -----------------------------------------------------------------
    def swatch(self, color):
        pixmap = QPixmap(14, 14)
        pixmap.fill(QColor(*color))
        return QIcon(pixmap)

    def refill_materials(self):
        for box in (self.material, self.material2):
            box.blockSignals(True)
            box.clear()
            for name, color in self.materials.items():
                box.addItem(self.swatch(color), name)
            box.blockSignals(False)

    def material_dialog(self, name, color, taken):
        """(name, color) from the popup, or None if canceled."""
        dialog = MaterialDialog(self.material, name, color, taken)
        return dialog.result_value() if dialog.exec() == QDialog.DialogCode.Accepted else None

    def edit_named_material(self, name):
        """Edit a material (name None: a new one). A rename follows into every area that uses it."""
        new = name is None
        if new:
            base, k = "Material", 1
            while f"{base} {k}" in self.materials:
                k += 1
            name, color = f"{base} {k}", (90, 60, 40)
        else:
            if name not in self.materials:
                return False
            color = self.materials[name]
        taken = [n for n in self.materials if n != name or new]
        chosen = self.material_dialog(name, color, taken)
        if chosen is None:
            return False
        new_name, new_color = chosen
        if not new_name or new_name.lower() in {t.lower() for t in taken}:
            return False
        if new:
            self.materials[new_name] = tuple(new_color)
        else:
            self.materials = {(new_name if n == name else n): (tuple(new_color) if n == name else c)
                              for n, c in self.materials.items()}
            for area in self.areas:
                if area.material == name:
                    area.material = new_name
                if area.material2 == name:
                    area.material2 = new_name
        self.refill_materials()
        area = self.current()
        if new and area is not None:
            area.material = new_name
        self.save()
        self.recall()
        return True

    def colors_of(self, area):
        """The area's primary and secondary colors from its named materials."""
        missing = DEFAULT_MATERIALS["Hair"]
        return (tuple(self.materials.get(area.material, area.color or missing)),
                tuple(self.materials.get(area.material2, area.color2 or missing)))

    def current(self):
        index = self.area_box.currentIndex() - 1
        return self.areas[index] if 0 <= index < len(self.areas) else None

    def refill(self, select=None):
        self.area_box.blockSignals(True)
        self.area_box.clear()
        self.area_box.addItem(NO_AREA)
        for area in self.areas:
            self.area_box.addItem(area.name)
        index = self.area_box.findText(select) if select else 0
        self.area_box.setCurrentIndex(max(0, index))
        self.area_box.blockSignals(False)
        self.recall()

    def new_area(self):
        name = self.area_name.text().strip()
        if not name or name == NO_AREA or any(a.name.lower() == name.lower() for a in self.areas):
            self.summary.setText("Give the new area a name no other area has.")
            return
        # A new area starts with the default settings, not the shown area's.
        area = GrowArea(name, seed=len(self.areas) + 1)
        self.areas.append(area)
        self.area_name.clear()
        self.save()
        self.refill(select=name)
        self.paint.setChecked(True)

    def delete_area(self):
        area = self.current()
        if area is not None:
            self.areas.remove(area)
            self.save()
            self.refill()
            self.changed.emit()

    def clear_area(self):
        area = self.current()
        if area is not None:
            area.faces = set()
            self.save()
            self.update_summary()
            self.changed.emit()

    def recall(self):
        area = self.current()
        for widget in self.settings_widgets:
            widget.setEnabled(area is not None)
        if area is None:
            self.paint.setChecked(False)
            self.update_summary()
            self.changed.emit()
            return
        blocked = [w.blockSignals(True) for w in (self.edge, self.scatter, self.flip, self.extend, self.feather,
                                                  self.edge_density, self.edge_length, self.density,
                                                  self.length, self.variation, self.min_length, self.segments,
                                                  self.width, self.stiff, self.root_lift)]
        self.edge.setCurrentIndex(max(0, self.edge.findData(area.edge)))
        self.scatter.setCurrentIndex(max(0, self.scatter.findData(area.scatter)))
        self.flip.setChecked(area.flip)
        self.extend.setChecked(area.extend)
        self.feather.setValue(area.feather * 100)
        self.edge_density.setValue(area.edge_density * 100)
        self.edge_length.setValue(area.edge_length * 100)
        self.density.setValue(area.density)
        self.length.setValue(area.length)
        self.variation.setValue(area.length_variation * 100)
        self.min_length.setValue(area.min_length)
        self.segments.setValue(area.segments)
        self.width.setValue(area.width)
        self.stiff.setValue(area.stiffness * 100)
        self.root_lift.setValue(area.root_lift * 100)
        for w, state in zip((self.edge, self.scatter, self.flip, self.extend, self.feather, self.edge_density,
                             self.edge_length, self.density, self.length,
                             self.variation, self.min_length, self.segments, self.width, self.stiff,
                             self.root_lift), blocked):
            w.blockSignals(state)
        for box, name in ((self.material, area.material), (self.material2, area.material2)):
            if box.findText(name) < 0 and name:
                self.materials[name] = DEFAULT_MATERIALS["Hair"]
                self.refill_materials()
            box.blockSignals(True); box.setCurrentIndex(max(0, box.findText(name))); box.blockSignals(False)
        self.mix.blockSignals(True); self.mix.setValue(area.mix * 100); self.mix.blockSignals(False)
        self.material2.setEnabled(area.mix > 0); self.edit_material2.setEnabled(area.mix > 0)
        self.flip.setVisible(area.edge == "triangle")
        self.extend.setVisible(area.edge not in ("polygon", "smooth"))
        self.update_summary()
        self.changed.emit()

    def store(self):
        area = self.current()
        if area is None:
            return
        area.edge, area.scatter = self.edge.currentData(), self.scatter.currentData()
        area.flip, area.feather = self.flip.isChecked(), self.feather.value() / 100
        area.edge_density, area.edge_length = self.edge_density.value() / 100, self.edge_length.value() / 100
        area.extend = self.extend.isChecked()
        area.density, area.length = self.density.value(), self.length.value()
        area.length_variation, area.segments = self.variation.value() / 100, int(self.segments.value())
        area.min_length = self.min_length.value()
        area.width = self.width.value()
        area.stiffness = self.stiff.value() / 100
        area.root_lift = self.root_lift.value() / 100
        area.mix = self.mix.value() / 100
        area.material, area.material2 = self.material.currentText(), self.material2.currentText()
        self.material2.setEnabled(area.mix > 0); self.edit_material2.setEnabled(area.mix > 0)
        self.flip.setVisible(area.edge == "triangle")
        self.extend.setVisible(area.edge not in ("polygon", "smooth"))
        self.save()
        self.update_summary()

    def update_grow_button(self):
        count = sum(1 for area in self.areas if area.faces)
        self.grow_button.setText(f"Grow all areas ({count})" if count else "Grow all areas")
        self.update_segment_warning()

    # Heavy hair (no hard limit on Segments; ask before growing very long or finely divided
    # hair instead). Past these, the hair can be slow to simulate and to return to DAZ.
    HEAVY_LENGTH, HEAVY_SEGMENTS = 60., 30

    def heavy_areas(self):
        return [area for area in self.areas if area.faces
                and (area.length > self.HEAVY_LENGTH or area.segments > self.HEAVY_SEGMENTS)]

    def confirm_grow(self):
        heavy = self.heavy_areas()
        if heavy:
            names = ", ".join(f"“{a.name}” ({a.length:g} cm, {a.segments} segments)" for a in heavy)
            if QMessageBox.warning(
                    self.grow_button, "Grow hair",
                    f"{names}: hair longer than {self.HEAVY_LENGTH:g} cm or with more than {self.HEAVY_SEGMENTS} "
                    "segments makes many polygons, and can be slow to simulate and to return to DAZ." + chr(10) * 2 + "Grow anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                self.summary.setText("Not grown.")
                return
        self.grow_requested.emit()

    def update_segment_warning(self):
        """Areas whose segments are longer than 2.5 cm may cut through the head: say so and suggest a number."""
        lines = []
        for area in self.areas:
            wanted = suggested_segments(area)
            if area.faces and wanted:
                lines.append(f"“{area.name}”: {area.length / max(area.segments, 1):.1f} cm per segment may cut through "
                             f"the head. Try {wanted} segments.")
        self.segment_warning.setText(chr(10).join(lines))
        self.segment_warning.setVisible(bool(lines))

    def update_summary(self):
        self.update_grow_button()
        area = self.current()
        if area is None:
            self.summary.setText(f"{len(self.areas)} area(s). Choose or name an area to paint it."
                                 if self.areas else "No areas yet: name one and press New area.")
            return
        painted = np.array(sorted(area.faces), np.int64)
        size = 0.
        if len(painted):
            a, b, c = (self.avatar_vertices[self.avatar_triangles[painted, k]] for k in range(3))
            size = float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() * .5)
        self.summary.setText(f"{area.name}: {len(painted):,} polygons, {size:.1f} cm², about "
                             f"{int(size * area.density):,} strands.")

    # ----- painting ------------------------------------------------------------------------
    def paint_at(self, point, removing=False):
        """Add (or remove) the avatar triangles within the brush of ``point`` to the current area."""
        area = self.current()
        if area is None:
            return False
        near = np.flatnonzero(np.linalg.norm(self.centres - point, axis=1) <= self.brush.value())
        before = len(area.faces)
        if removing:
            area.faces.difference_update(near.tolist())
        else:
            area.faces.update(near.tolist())
        if len(area.faces) != before or (not removing and len(near)):
            self.update_summary()
            self.changed.emit()
            return True
        return False

    def finish_paint(self):
        self.save()

    def overlay(self):
        """Polydata of the selected area's painted triangles (none for -none-), in one bright paint color, lifted a
        little off the skin so it is never hidden in it."""
        current = self.current()
        cells, colors = [], []
        for area in ([current] if current is not None else []):
            faces = sorted(area.faces)
            if not faces:
                continue
            cells.append(self.avatar_triangles[faces])
            colors.append(np.tile(np.array(PAINT_COLOR, np.uint8), (len(faces), 1)))
        data = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        if not hasattr(self, "lifted"):
            from hair_grow import vertex_normals
            self.lifted = self.avatar_vertices + vertex_normals(self.avatar_vertices, self.avatar_triangles) * PAINT_LIFT
        points.SetData(numpy_to_vtk(self.lifted, deep=True))
        data.SetPoints(points)
        if cells:
            tris = np.concatenate(cells)
            ids = np.column_stack([np.full(len(tris), 3), tris]).astype(np.int64).ravel()
            polys = vtk.vtkCellArray()
            polys.SetCells(len(tris), numpy_to_vtkIdTypeArray(ids, deep=True))
            data.SetPolys(polys)
            data.GetCellData().SetScalars(numpy_to_vtk(np.concatenate(colors), deep=True,
                                                       array_type=vtk.VTK_UNSIGNED_CHAR))
        return data

    # ----- growing ---------------------------------------------------------------------------
    def grow(self):
        """Every painted area's strips: list of dict(area, vertices, faces, uvs, roots, strands)."""
        grown = []
        cache = getattr(self, "grow_cache", {})
        self.grow_cache, self.regrown = {}, []
        for area in self.areas:
            area.color, area.color2 = self.colors_of(area)          # from its named materials
            if area.faces:
                # An area is grown again only when its painting or settings changed.
                settings = area.to_json(); settings.pop("name")
                key = json.dumps(settings, sort_keys=True)
                result = cache.get(key)
                if result is None:
                    result = grow_area(self.avatar_vertices, self.avatar_triangles, area)
                    self.regrown.append(area.name)
                self.grow_cache[key] = result
                result = dict(result, key=key)
                if result["strands"]:
                    grown.append(dict(result, area=area))
        return grown


def with_grown(inputs, grown):
    """Session inputs with the grown strips appended (captured hair unchanged and first). ``inputs["grown"]`` lists each
    area's vertex and face range, name and color; the strips' surfaces are named "Grown <n>" (never a static name)."""
    extended = dict(inputs)
    vertices, faces = [inputs["vertices"]], [inputs["faces"]]
    surfaces, names = [inputs["face_surface"]], list(inputs["surface_names"])
    uv_parts = [inputs["vertex_uvs"]] if inputs.get("vertex_uvs") is not None else None
    offset, face_offset, parts, roots = len(inputs["vertices"]), len(inputs["faces"]), [], []
    for index, item in enumerate(grown):
        names.append(f"Grown {index + 1}")
        vertices.append(item["vertices"])
        faces.append(np.where(item["faces"] >= 0, item["faces"] + offset, -1))
        surface = np.full(len(item["faces"]), len(names) - 1, dtype=np.asarray(inputs["face_surface"]).dtype)
        second = np.asarray(item.get("face_secondary", np.zeros(len(item["faces"]), bool)), bool)
        if second.any():                         # the area's secondary color is a second surface
            names.append(f"Grown {index + 1} secondary")
            surface[second] = len(names) - 1
        surfaces.append(surface)
        if uv_parts is not None:
            uv_parts.append(item["uvs"])
        parts.append(dict(name=item["area"].name, color=tuple(item["area"].color), color2=tuple(item["area"].color2),
                          face_secondary=second, material=item["area"].material, material2=item["area"].material2,
                          key=(index, len(names), item.get("key")),   # its analysis is kept while this stays the same
                          start=offset,
                          end=offset + len(item["vertices"]), face_start=face_offset,
                          face_end=face_offset + len(item["faces"]), strands=item["strands"],
                          stiffness=float(item["area"].stiffness),
                          root_lift=float(getattr(item["area"], "root_lift", 0.))))
        roots.append(item["roots"])
        offset += len(item["vertices"])
        face_offset += len(item["faces"])
    extended["vertices"] = np.concatenate(vertices)
    extended["faces"] = np.concatenate(faces)
    extended["face_surface"] = np.concatenate(surfaces)
    extended["surface_names"] = names
    if uv_parts is not None:
        extended["vertex_uvs"] = np.concatenate(uv_parts)
    extended["grown"] = parts
    extended["grown_roots"] = np.concatenate(roots) if roots else np.zeros((0, 3))
    return extended
