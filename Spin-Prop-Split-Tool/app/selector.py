# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Spin Prop Split Tool polygon selector and DAZ split bridge.

Split off any number of pieces, give each its own pivot, then send them all
at once. Faces not in a piece form the Remainder.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path
import sys

from PySide6.QtCore import Qt, QSettings, QTimer, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap, QPainter, QPainterPath, QPen, QColor
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

from mesh import demo_mesh, load_obj, save_selections
from multi_split import split_pieces
from ngons import ngon_report, split_ngons
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
import surface_view
from surface_view import FLAT, COLORS, TEXTURES, Surfaces
from branding import APP_NAME, VERSION
from project_settings import ROOT, session_directory
from live_return import publish, release_unsent
import sys as _sys
from pathlib import Path as _Path
_SUITE_APP = str(_Path(__file__).resolve().parents[2] / "app")
if _SUITE_APP not in _sys.path:
    _sys.path.append(_SUITE_APP)          # suite app/: shared interface (spin_ui) and keybindings
import spin_ui


def visibility_icon(visible):
    pixmap=QPixmap(20,20);pixmap.fill(Qt.GlobalColor.transparent)
    painter=QPainter(pixmap);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor('#202630' if visible else '#666666'),1.6))
    path=QPainterPath();path.moveTo(1,10);path.quadTo(10,0,19,10);path.quadTo(10,20,1,10)
    painter.drawPath(path);painter.drawEllipse(7,7,6,6)
    if not visible:painter.drawLine(2,18,18,2)
    painter.end()
    return QIcon(pixmap)


class PieceList(QListWidget):
    visibilityRequested=Signal(object,bool)

    def mousePressEvent(self,event):
        item=self.itemAt(event.position().toPoint())
        self.eye_pressed=bool(item and event.button()==Qt.MouseButton.LeftButton and
                              event.position().x()<self.visualItemRect(item).left()+26)
        if self.eye_pressed:
            self.visibilityRequested.emit(item,bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier))
            event.accept();return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self,event):
        if getattr(self,'eye_pressed',False):
            self.eye_pressed=False;event.accept();return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self,event):
        item=self.itemAt(event.position().toPoint())
        if item and event.position().x()<self.visualItemRect(item).left()+26:
            event.accept();return
        super().mouseDoubleClickEvent(event)

PIECE_COLORS = [(92, 196, 120), (196, 112, 214), (232, 206, 72), (80, 200, 220), (230, 120, 120),
                (150, 150, 240), (240, 160, 210), (170, 220, 90)]
SELECTION_COLOR, REMAINDER_COLOR = (255, 165, 55), (139, 175, 211)
REMAINDER = "Remainder"


def closest_edge_point(point, a, b):
    direction = [end - start for start, end in zip(a, b)]
    length_squared = sum(value * value for value in direction)
    if not length_squared:
        return a
    fraction = sum((p - start) * delta for p, start, delta in zip(point, a, direction))
    fraction = min(1, max(0, fraction / length_squared))
    return tuple(start + fraction * delta for start, delta in zip(a, direction))


class Selector(QMainWindow):
    def __init__(self, bridge_source=None, bridge_result=None, bridge_session=None, allow_ngons=False, config=None):
        super().__init__()
        # Standalone / custom-bridge options: output, result, after_export, callback, overwrite, close_after.
        self.config = dict(config or {})
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 720)
        self.settings = QSettings('Spin256', APP_NAME)
        spin_ui.remember_window(self, self.settings)
        self.bridge = bridge_result is not None
        self.bridge_source, self.bridge_result, self.bridge_session = bridge_source, bridge_result, bridge_session
        self.allow_ngons = allow_ngons
        self.ngon_count = 0
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        # Controls on the left, as in the other Spin Tools (shared sections, button row, status bar).
        panel = QWidget()
        controls = QVBoxLayout(panel)
        sections = spin_ui.Sections(controls, self.settings)
        self.sections = sections.items
        self.viewport = QVTKRenderWindowInteractor(central)
        self.mode = None
        sidebar = sections.add("Mesh and selection", True)
        spin_ui.note("Click to select · Ctrl+click to toggle. Left-drag orbit · middle-drag pan · wheel zoom. "
                     "F1 lists the keys.", sidebar)

        def button(label, action):
            widget = QPushButton(label)
            widget.clicked.connect(action)
            sidebar.addWidget(widget)
            return widget

        if not self.bridge:
            button("Open OBJ…", self.open_obj)
            button("Load demo", self.load_demo)
        self.ngon_button = button("Split n-gons", self.split_ngons)
        self.ngon_button.setToolTip("Split polygons with more than four sides into triangles/quads. "
                                    "From DAZ this uses DAZ's own split, so the pieces verify on return.")
        button("Select connected", self.select_connected)
        button("Clear selection", self.clear_selection)

        sidebar = sections.add("Pieces", True)
        spin_ui.note("Click a piece to select it · click its eye to show/hide · Ctrl-click the eye to isolate. "
                     "Visibility affects only the preview.", sidebar)
        button("Split off selection", self.split_off)
        self.piece_list = PieceList()
        self.piece_list.setIconSize(QSize(20,20))
        self.piece_list.visibilityRequested.connect(self.toggle_piece_visibility)
        self.piece_list.itemClicked.connect(self.choose_piece)
        self.piece_list.itemDoubleClicked.connect(self.rename_piece)
        sidebar.addWidget(self.piece_list, 1)
        button("Show all pieces", self.show_all_pieces)
        button("Rename piece…", lambda: self.rename_piece(self.piece_list.currentItem()))
        button("Remove piece", self.remove_piece)
        self.pivot_vertex = button("Place pivot on vertex", lambda: self.set_mode("vertex"))
        self.pivot_edge = button("Place pivot on edge", lambda: self.set_mode("edge"))
        for widget in (self.pivot_vertex, self.pivot_edge):
            widget.setToolTip("Sets the pivot of the piece chosen in the list (or of the current selection).")
        # UV map: the UV lines of the selection (or the whole prop) on a transparent PNG.
        uv_box = sections.add("UV map", False)
        spin_ui.note("Saves the UV lines of the selected polygons (all polygons when nothing is selected) as a PNG "
                     "with a transparent background.", uv_box)
        self.uv_size = QComboBox()
        for size in surface_view.UV_MAP_SIZES:
            self.uv_size.addItem(f"{size} × {size}", size)
        self.uv_size.setCurrentIndex(max(0, self.uv_size.findData(self.settings.value("uv/size", 2048, type=int))))
        self.uv_size.currentIndexChanged.connect(lambda *_: self.settings.setValue("uv/size", self.uv_size.currentData()))
        uv_form = QFormLayout()
        uv_form.addRow("Image size", self.uv_size)
        uv_box.addLayout(uv_form)
        self.uv_button = QPushButton("Save UV map…")
        self.uv_button.clicked.connect(lambda: self.save_uv_map())
        uv_box.addWidget(self.uv_button)
        view_box = sections.add("View", False)
        self.view_box = view_box
        self.view = spin_ui.ViewControls(view_box, self.settings, "prop", self.update_view, avatar_name="pivot marker", tool="prop_split",
                                         avatar_opacity=100)
        self.view.reset_view_button.setToolTip("Fit the mesh in the view again" + spin_ui.key_hint("prop_split", "reset_view") + ".")   # "Reset view", as in every tool
        self.view.reset_view_button.clicked.connect(self.frame)
        # Display: flat piece colors, each DAZ surface's base color, or its full
        # base color texture. Pieces keep a tint of their color and the selection stays orange. Preview only.
        self.display = QComboBox()
        self.display.addItem("Flat (piece colors)", FLAT)
        self.display.addItem("Surface colors", COLORS)
        self.display.addItem("Textures", TEXTURES)
        self.display.setToolTip("Surface colors and textures come from the DAZ surfaces of the captured prop "
                                "(their base color and base color map).")
        display_form = QFormLayout()
        display_form.addRow("Display", self.display)
        view_box.addLayout(display_form)
        self.display.currentIndexChanged.connect(self.display_changed)
        self.surfaces = None
        self.texture_actors = []          # [(actor, face ids)] in Textures display
        self.texture_cache = {}

        self.custom_bridge = bool(self.config.get('after_export') or self.config.get('callback'))
        if self.bridge:
            context, text, default = 'daz', "Close after sending", False
            tip = "Off: minimize after each send so you can inspect the split in DAZ and send again."
            send_text = "Send to DAZ"
        else:
            if not self.custom_bridge:
                button("Export selection JSON…", self.export)
            if self.custom_bridge:
                context, text, default = 'bridge', "Close after send", True
                tip = "On: exit after the export and its bridge callback start. Off: minimize so you can send again."
            else:
                context, text, default = 'export', "Close after export", False
                tip = "On: close once the pieces are written."
            send_text = "Send…" if self.custom_bridge else "Export pieces…"
        # Shared bottom row; there is no simulation here, so Run/Pause is hidden.
        self.actions = spin_ui.ActionRow(self.settings, reset_text="Reset", send_text=send_text, close_context=context, tool="prop_split",
                                         close_text=text, close_default=default, close_tip=tip,
                                         close_override=self.config.get('close_after'))
        self.actions.run_button.hide()
        self.reset_button, self.send_button = self.actions.reset_button, self.actions.send_button
        self.close_after = self.actions.close_after
        self.close_after_key = self.actions.close_after_key
        self.reset_button.setToolTip("Remove every piece, pivot and the selection (the mesh is unchanged).")
        self.reset_button.clicked.connect(self.reset_pieces)
        if self.bridge:
            self.send_button.clicked.connect(self.send_to_daz)
            self.send_button.setToolTip("Send every piece (and the Remainder) to DAZ. The original prop is hidden, "
                                        "not deleted. Send again to replace the previous split"
                                        + spin_ui.key_hint("prop_split", "send_to_daz") + ".")
        else:
            self.send_button.clicked.connect(self.export_pieces)
            self.send_button.setToolTip("Write each piece as an OBJ beside the chosen file, with a result manifest.")
        controls.addStretch()
        column, self.copyright_footer = spin_ui.control_column(panel, self.actions, 300, version=VERSION, tool_folder=Path(__file__).resolve().parent.parent)
        column.setMaximumWidth(360)
        layout.addWidget(column)
        holder, self.status = spin_ui.viewport_with_status(self.viewport)
        self.status.set_state("ready")
        layout.addWidget(holder, 1)

        self.renderer = spin_ui.style_renderer(vtk.vtkRenderer())
        self.window = self.viewport.GetRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.window.SetMultiSamples(0)
        self.interactor = self.window.GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        style.AddObserver("LeftButtonPressEvent", self.mouse_down)
        style.AddObserver("LeftButtonReleaseEvent", self.mouse_up)
        style.AddObserver("MouseMoveEvent", self.mouse_move)
        style.AddObserver("LeaveEvent", lambda *_: self.clear_hover())
        self.style = style
        self.press = None

        self.actor = vtk.vtkActor()
        self.mapper = vtk.vtkPolyDataMapper()
        self.actor.SetMapper(self.mapper)
        prop = self.actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(.13, .18, .24)
        self.renderer.AddActor(self.actor)

        self.marker_source = vtk.vtkSphereSource()
        self.marker_mapper = vtk.vtkPolyDataMapper()
        self.marker_mapper.SetInputConnection(self.marker_source.GetOutputPort())
        self.marker = vtk.vtkActor()
        self.marker.SetMapper(self.marker_mapper)
        self.marker.GetProperty().SetColor(1, .28, .12)
        self.marker.PickableOff()
        self.renderer.AddActor(self.marker)
        self.hover_source=vtk.vtkSphereSource()
        self.hover_edge_source=vtk.vtkLineSource()
        self.hover_actors=[]
        # Hover highlights are drawn in an overlay layer with the same camera,
        # so the hovered vertex/edge always shows on top of the surface.
        self.hover_renderer=vtk.vtkRenderer();self.hover_renderer.SetLayer(1);self.hover_renderer.InteractiveOff()
        self.hover_renderer.SetActiveCamera(self.renderer.GetActiveCamera())
        self.window.SetNumberOfLayers(2);self.window.AddRenderer(self.hover_renderer)
        for source in (self.hover_source,self.hover_edge_source):
            mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(source.GetOutputPort())
            actor=vtk.vtkActor();actor.SetMapper(mapper);actor.PickableOff();actor.VisibilityOff()
            actor.GetProperty().SetColor(1.,.88,.12);actor.GetProperty().SetLineWidth(5)
            actor.GetProperty().LightingOff();self.hover_renderer.AddActor(actor);self.hover_actors.append(actor)
        self.hover_target=None
        self.picker = vtk.vtkCellPicker()
        self.picker.PickFromListOn()
        self.picker.AddPickList(self.actor)
        self.picked_actor_faces = {}
        self.camera = spin_ui.ViewportCamera(self.renderer, self.render_view, lambda: [self.actor])
        self.keys = spin_ui.Shortcuts(self, 'prop_split', {
            'reset_view': self.frame, 'send_to_daz': lambda: self.send_button.click(),
            'clear_selection': self.clear_selection, 'cancel_interaction': self.cancel_mode,
            'grow_selection': self.select_connected}, camera=self.camera)
        self.view_box.addWidget(self.keys.help_button())
        self.view_bar = spin_ui.add_view_toolbar(self.keys, self.status)   # orbit/pan/zoom/reset buttons
        self.view.lighting.attach(self.renderer, self.render_view)
        if bridge_source is None:
            self.load_demo()
        else:
            self.set_mesh(load_obj(bridge_source), bridge_source)

    # ----- mesh loading and n-gons -------------------------------------------------
    def set_mesh(self, mesh, source):
        self.mesh, self.source = mesh, source
        self.surfaces = None
        if source is not None and self.bridge_session is not None and str(source) == str(self.bridge_source):
            try:
                session = json.loads(Path(self.bridge_session).read_text(encoding="utf-8-sig"))
                self.surfaces = Surfaces.from_session(source, session)
                if self.surfaces is not None and len(self.surfaces.face_surface) != len(mesh.faces):
                    self.surfaces = None
            except (OSError, ValueError):
                self.surfaces = None
        self.update_display_choices()
        self.uv_data = None                        # UVs of the OBJ, for Save UV map (with or without DAZ looks)
        if source is not None:
            try:
                self.uv_data = self.surfaces or Surfaces(source, [])
                if len(self.uv_data.face_uvs) != len(mesh.faces):
                    self.uv_data = None
            except (OSError, ValueError, IndexError):
                self.uv_data = None
        if hasattr(self, "uv_button"):
            has_uvs = self.uv_data is not None and any(uv is not None for uv in self.uv_data.face_uvs)
            self.uv_button.setEnabled(has_uvs)
            self.uv_button.setToolTip("" if has_uvs else "This mesh has no UVs.")
        self.selection = set()
        self.pieces = []              # [{"name", "faces", "pivot"}]
        self.remainder_visible = True
        self.display_faces = None
        self.clear_hover()
        self.remainder_pivot = None
        self.pivot = self.mode = None
        self.marker.VisibilityOff()
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        for point in mesh.vertices:
            points.InsertNextPoint(point)
        polygons = vtk.vtkCellArray()
        for face in mesh.faces:
            polygons.InsertNextCell(len(face))
            for index in face:
                polygons.InsertCellPoint(index)
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetPolys(polygons)
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetNumberOfTuples(len(mesh.faces))
        data.GetCellData().SetScalars(self.colors)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(data)
        normals.SplittingOff()
        normals.ConsistencyOff()
        normals.Update()
        self.full_data=normals.GetOutput()
        self.mapper.SetInputData(normals.GetOutput())
        self.mapper.SetScalarModeToUseCellData()
        self.mapper.SetColorModeToDirectScalars()
        self.marker_source.SetRadius(max(data.GetLength() * .012, .000001))
        self.hover_source.SetRadius(max(data.GetLength() * .006, .000001))
        self.ngon_count = sum(len(face) > 4 for face in mesh.faces)
        self.ngon_button.setEnabled(self.ngon_count > 0 and source is not None)
        self.refresh_pieces()
        self.paint_selection()
        self.frame()
        if self.ngon_count and source is not None and not self.allow_ngons:
            QTimer.singleShot(0, self.warn_ngons)

    def warn_ngons(self):
        try:
            count, surfaces = ngon_report(self.source)
        except OSError:
            return
        if not count:
            return
        where = f" in surface{'s' if len(surfaces) > 1 else ''} {', '.join(s or '(none)' for s in surfaces)}" if surfaces else ""
        QMessageBox.warning(
            self, "Polygons with more than four sides",
            f"This mesh has {count} polygon{'s' if count > 1 else ''} with more than four sides{where}.\n\n"
            "The DAZ split needs triangles and quads. Click Split n-gons before splitting; "
            "from DAZ it uses DAZ's own split of those polygons.")

    def daz_facets(self):
        if not self.bridge or self.bridge_session is None:
            return None
        try:
            return json.loads(Path(self.bridge_session).read_text(encoding="utf-8-sig")).get("source_facets")
        except (OSError, ValueError):
            return None

    def split_ngons(self):
        if not self.ngon_count or self.source is None:
            return
        if self.pieces or self.selection:
            answer = QMessageBox.question(self, "Split n-gons", "Splitting n-gons renumbers polygons and clears "
                                          "the current pieces and selection. Continue?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            if self.bridge:
                # The capture is ours: rewrite it in place, using DAZ's own facets.
                destination = Path(self.source)
            else:
                # Never modify the user's file; work on a copy in the session folder.
                destination = session_directory() / (Path(self.source).stem + "-ngons-split.obj")
            count, exact = split_ngons(self.source, destination, self.daz_facets())
            self.set_mesh(load_obj(destination), destination)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot split n-gons", str(error))
            return
        note = " using DAZ's split" if exact else (" (fan triangulation)" if count else "")
        self.status.setText(f"Split {count} n-gon{'s' if count != 1 else ''}{note}.\n" + self.status.text())

    def open_obj(self):
        name, _ = QFileDialog.getOpenFileName(self, "Open mesh", filter="Wavefront OBJ (*.obj)")
        if name:
            try:
                self.set_mesh(load_obj(name), Path(name))
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "Cannot open mesh", str(error))

    def load_demo(self):
        example = ROOT/'examples/Cube and Sphere/Cube and Sphere.obj'
        if example.exists(): self.set_mesh(load_obj(example), example)
        else: self.set_mesh(demo_mesh(), None)

    # ----- pieces -----------------------------------------------------------------
    def assigned(self):
        return {face for piece in self.pieces for face in piece["faces"]}

    def remainder(self):
        return set(range(len(self.mesh.faces))) - self.assigned()

    def split_off(self):
        if not self.selection:
            self.status.setText("Select polygons first, then Split off selection.")
            return
        faces = set(self.selection)
        # Polygons move from any earlier piece into the new one.
        for piece in self.pieces:
            piece["faces"] -= faces
        self.pieces = [piece for piece in self.pieces if piece["faces"]]
        names = {piece["name"].lower() for piece in self.pieces}
        number = len(self.pieces) + 1
        while f"piece {number}" in names:
            number += 1
        self.pieces.append({"name": f"Piece {number}", "faces": faces, "pivot": self.pivot})
        self.selection, self.pivot = set(), None
        self.refresh_pieces(select=len(self.pieces) - 1)
        self.paint_selection()

    def current_piece(self):
        """Index into self.pieces, REMAINDER, or None."""
        item = self.piece_list.currentItem()
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def choose_piece(self, item):
        key = item.data(Qt.ItemDataRole.UserRole)
        faces = self.remainder() if key == REMAINDER else self.pieces[key]["faces"]
        pivot = self.remainder_pivot if key == REMAINDER else self.pieces[key]["pivot"]
        self.selection = set(faces)
        self.show_marker(pivot)
        self.paint_selection()

    def rename_piece(self, item):
        key = item.data(Qt.ItemDataRole.UserRole)
        if key == REMAINDER:
            return
        name, ok = QInputDialog.getText(self, "Rename piece", "Piece name", text=self.pieces[key]["name"])
        name = name.strip()
        if not ok or not name:
            return
        taken = {p["name"].lower() for i, p in enumerate(self.pieces) if i != key} | {REMAINDER.lower()}
        if name.lower() in taken:
            QMessageBox.information(self, "Name already used", "Choose a unique piece name.")
            return
        self.pieces[key]["name"] = name
        self.refresh_pieces(select=key)

    def remove_piece(self):
        key = self.current_piece()
        if key is None or key == REMAINDER:
            return
        del self.pieces[key]
        self.selection = set()
        self.refresh_pieces()
        self.paint_selection()

    def refresh_pieces(self, select=None):
        self.piece_list.clear()
        for index, piece in enumerate(self.pieces):
            pivot = " · pivot set" if piece["pivot"] is not None else ""
            item = QListWidgetItem(f"{piece['name']} — {len(piece['faces']):,} polygons{pivot}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setIcon(visibility_icon(piece.get('visible',True)))
            item.setToolTip('Click eye to show/hide; Ctrl-click eye to isolate. Preview only; still exported.')
            item.setForeground(Qt.GlobalColor.black)
            item.setBackground(_qcolor(PIECE_COLORS[index % len(PIECE_COLORS)]))
            self.piece_list.addItem(item)
        if self.pieces:
            remaining = len(self.remainder())
            pivot = " · pivot set" if self.remainder_pivot is not None else ""
            item = QListWidgetItem(f"{REMAINDER} — {remaining:,} polygons{pivot}")
            item.setData(Qt.ItemDataRole.UserRole, REMAINDER)
            item.setIcon(visibility_icon(self.remainder_visible))
            item.setToolTip('Click eye to show/hide; Ctrl-click eye to isolate. Preview only; still exported.')
            self.piece_list.addItem(item)
        if select is not None and select < self.piece_list.count():
            self.piece_list.setCurrentRow(select)

    def visible_faces(self):
        hidden={face for piece in self.pieces if not piece.get('visible',True) for face in piece['faces']}
        if not self.remainder_visible:hidden.update(self.remainder())
        return [face for face in range(len(self.mesh.faces)) if face not in hidden]

    def toggle_piece_visibility(self,item,isolate=False):
        key=item.data(Qt.ItemDataRole.UserRole)
        row=self.piece_list.currentRow()
        if isolate:
            for index,piece in enumerate(self.pieces):piece['visible']=index==key
            self.remainder_visible=key==REMAINDER
        elif key==REMAINDER:self.remainder_visible=not self.remainder_visible
        else:self.pieces[key]['visible']=not self.pieces[key].get('visible',True)
        self.selection.intersection_update(self.visible_faces())
        self.clear_hover();self.refresh_pieces(select=row);self.paint_selection()

    def show_all_pieces(self):
        for piece in self.pieces:piece['visible']=True
        self.remainder_visible=True
        self.refresh_pieces(select=self.piece_list.currentRow());self.paint_selection()

    # ----- send / export ----------------------------------------------------------
    def split_request(self):
        """Pieces to send: the split-off pieces, or the current selection as a single piece."""
        if self.pieces:
            pieces = [(p["name"], p["faces"], p["pivot"]) for p in self.pieces]
            if self.remainder() and self.remainder_pivot is not None:
                pieces.append((REMAINDER, self.remainder(), self.remainder_pivot))
            return pieces
        return [("Selected", self.selection, self.pivot)]

    def after_send(self, message):
        """Like the other Spin Tools: close if asked, otherwise hand control back by minimizing."""
        self.status.set_state("sent" if self.bridge else "exported")
        self.status.setText(message)
        if self.close_after.isChecked():
            self.close()
        elif self.bridge or self.custom_bridge:
            self.showMinimized()

    def send_to_daz(self):
        if self.source != self.bridge_source:
            QMessageBox.warning(self, "Wrong mesh", "Open the original mesh from this DAZ session.")
            return
        if self.bridge_session is None:
            QMessageBox.warning(self, "Cannot send", "This selector was not opened from a Spin Tools DAZ capture.")
            return
        session = Path(self.bridge_session)
        try:
            # Every Send is a new revision; the receive window imports the latest.
            revision = Path(tempfile.mkdtemp(prefix=session.stem + "-return-", dir=session.parent))
            try:
                split_pieces(self.source, revision / "result.json", self.split_request())
            except BaseException:
                shutil.rmtree(revision, ignore_errors=True)
                raise
            publish(session, revision / "result.json")
        except (OSError, ValueError) as error:
            self.status.set_state("failed")
            self.status.setText("Send failed; the pieces are still here to retry.")
            QMessageBox.warning(self, "Cannot split prop", str(error))
            return
        self.after_send("The Spin Tools receive window imports it; adjust and send again to replace it.")

    def export_pieces(self):
        output = self.config.get('output')
        overwrite = bool(self.config.get('overwrite'))
        if not output:
            output, _ = QFileDialog.getSaveFileName(self, "Export pieces", filter="Wavefront OBJ (*.obj)")
            if not output:
                return
            if not output.lower().endswith('.obj'):
                output += '.obj'
        from split_export import export_split
        try:
            try:
                payload, result = export_split(self.source, output, self.split_request(), self.config,
                                               overwrite, self.allow_ngons)
            except ValueError as error:
                if overwrite or 'already exists' not in str(error):
                    raise
                answer = QMessageBox.question(self, "Replace exported files?",
                                              "Some exported files already exist. Replace them?")
                if answer != QMessageBox.StandardButton.Yes:
                    return
                payload, result = export_split(self.source, output, self.split_request(), self.config,
                                               True, self.allow_ngons)
        except (OSError, ValueError) as error:
            self.status.set_state("failed")
            self.status.setText("Export failed; the pieces are still here to retry.")
            QMessageBox.warning(self, "Cannot export pieces", str(error))
            return
        self.after_send(f"Exported {len(payload['outputs'])} pieces: {result}")

    def export(self):
        name, _ = QFileDialog.getSaveFileName(self, "Export selections", filter="JSON (*.json)")
        if name:
            groups = {piece["name"]: piece["faces"] for piece in self.pieces}
            try:
                save_selections(name, self.source, self.selection, groups, self.pivot)
            except OSError as error:
                QMessageBox.warning(self, "Cannot export selections", str(error))

    def closeEvent(self, event):
        spin_ui.save_window(self, self.settings)
        super().closeEvent(event)
        if self.bridge_session is not None:
            try:
                release_unsent(self.bridge_session, self.bridge_source)
            except OSError:
                pass

    # ----- viewport interaction ---------------------------------------------------
    def frame(self):
        self.renderer.ResetCamera()
        self.render_view()

    def render_view(self):
        if self.isVisible():
            self.window.Render()

    def update_view(self):
        self.view.apply([self.actor] + [actor for actor, _ in self.texture_actors], [self.marker])
        if not self.actor.GetVisibility():self.clear_hover()
        if not self.view.show_avatar.isChecked() or getattr(self, "marker_hidden", True):
            self.marker.SetVisibility(False)
        self.render_view()

    def cancel_mode(self):
        self.mode = None
        self.clear_hover()
        self.status.setText("Pivot placement canceled.")

    def reset_pieces(self):
        if self.pieces and QMessageBox.question(self, "Reset", "Remove every piece and pivot?") != QMessageBox.StandardButton.Yes:
            return
        self.pieces, self.selection, self.pivot, self.remainder_pivot, self.mode = [], set(), None, None, None
        self.remainder_visible=True
        self.clear_hover()
        self.show_marker(None)
        self.refresh_pieces()
        self.paint_selection()
        self.status.set_state("ready")

    def mouse_down(self, style, _event):
        self.press = self.interactor.GetEventPosition()
        if not self.mode:
            style.OnLeftButtonDown()

    def mouse_up(self, style, _event):
        position = self.interactor.GetEventPosition()
        if not self.mode:
            style.OnLeftButtonUp()
        press, self.press = self.press, None
        if press is None or sum((a - b) ** 2 for a, b in zip(press, position)) > 16:
            return
        if not self.picker.Pick(*position, 0, self.renderer):
            return
        face = self.picked_face()
        if face is None:
            return
        if self.mode:
            self.place_pivot(face, self.picker.GetPickPosition())
        elif QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier:
            self.selection.symmetric_difference_update({face})
        else:
            self.selection = {face}
            self.piece_list.clearSelection()
            self.piece_list.setCurrentItem(None)
        self.paint_selection()

    def set_mode(self, mode):
        self.clear_hover()
        self.mode = mode
        target = self.current_piece()
        name = REMAINDER if target == REMAINDER else (self.pieces[target]["name"] if target is not None else "the selection")
        self.status.setText(f"Click a polygon to snap the pivot of {name} to its nearest {mode}.")

    def pivot_candidate(self,face,point):
        vertices = [self.mesh.vertices[index] for index in self.mesh.faces[face]]
        candidates = vertices
        edges=list(zip(vertices,vertices[1:]+vertices[:1]))
        if self.mode == "edge":
            candidates = [closest_edge_point(point, a, b)
                          for a, b in edges]
        index=min(range(len(candidates)),key=lambda i:sum((a-b)**2 for a,b in zip(candidates[i],point)))
        return candidates[index],edges[index] if self.mode=='edge' else None

    def clear_hover(self):
        self.hover_target=None
        changed=any(actor.GetVisibility() for actor in getattr(self,'hover_actors',[]))
        for actor in getattr(self,'hover_actors',[]):actor.VisibilityOff()
        if changed:self.render_view()

    def mouse_move(self,style,_event):
        style.OnMouseMove()
        if self.mode not in ('vertex','edge') or not self.view.show_object.isChecked():
            self.clear_hover();return
        position=self.interactor.GetEventPosition()
        if not self.picker.Pick(*position,0,self.renderer):
            self.clear_hover();return
        face=self.picked_face()
        if face is None:
            self.clear_hover();return
        point,edge=self.pivot_candidate(face,self.picker.GetPickPosition())
        self.hover_target=point
        self.hover_actors[0].SetVisibility(edge is None)
        self.hover_actors[1].SetVisibility(edge is not None)
        if edge is None:self.hover_actors[0].SetPosition(*point)
        else:
            self.hover_edge_source.SetPoint1(*edge[0]);self.hover_edge_source.SetPoint2(*edge[1])
        self.render_view()

    def place_pivot(self, face, point):
        pivot,_=self.pivot_candidate(face,point)
        target = self.current_piece()
        if target == REMAINDER:
            self.remainder_pivot = pivot
        elif target is not None:
            self.pieces[target]["pivot"] = pivot
        else:
            self.pivot = pivot
        self.show_marker(pivot)
        self.mode = None
        self.clear_hover()
        if target is not None:
            self.refresh_pieces(select=self.piece_list.count() - 1 if target == REMAINDER else target)

    def show_marker(self, pivot):
        self.marker_hidden = pivot is None
        if pivot is None:
            self.marker.VisibilityOff()
        else:
            self.marker.SetPosition(*pivot)
            self.marker.SetVisibility(self.view.show_avatar.isChecked())

    # ----- display: flat, surface colors, textures -----------------------------------------------
    def display_mode(self):
        mode = self.display.currentData()
        if mode == TEXTURES and (self.surfaces is None or not self.surfaces.has_textures()):
            return COLORS if self.surfaces is not None else FLAT
        if mode == COLORS and self.surfaces is None:
            return FLAT
        return mode

    def update_display_choices(self):
        model = self.display.model()
        model.item(1).setEnabled(self.surfaces is not None)
        model.item(2).setEnabled(self.surfaces is not None and self.surfaces.has_textures())
        wanted = self.settings.value("view/display", FLAT)
        index = self.display.findData(wanted)
        with_signals = self.display.blockSignals(True)
        self.display.setCurrentIndex(index if index >= 0 and model.item(index).isEnabled() else 0)
        self.display.blockSignals(with_signals)

    def display_changed(self, *_):
        self.settings.setValue("view/display", self.display.currentData())
        self.display_faces = None
        if hasattr(self, "mesh"):
            self.paint_selection()

    def save_uv_map(self, path=None):
        """Write the UV map PNG; returns the path written, or None."""
        if self.uv_data is None:
            return None
        faces = sorted(self.selection) if self.selection else range(len(self.mesh.faces))
        if path is None:
            name = (Path(self.source).stem if self.source else "prop") + "-uv.png"
            folder = self.settings.value("uv/folder", str(Path.home()))
            path, _ = QFileDialog.getSaveFileName(self, "Save UV map", str(Path(folder) / name), "PNG image (*.png)")
            if not path:
                return None
            if not path.lower().endswith(".png"):
                path += ".png"
            self.settings.setValue("uv/folder", str(Path(path).parent))
        image, drawn = surface_view.uv_map_image(self.uv_data, faces, int(self.uv_size.currentData()))
        if not drawn:
            self.status.setText("The selected polygons have no UVs.")
            return None
        if not image.save(str(path), "PNG"):
            QMessageBox.warning(self, "UV map", f"Could not save {path}.")
            return None
        self.status.setText(f"Saved the UV map of {drawn:,} polygon{'s' if drawn != 1 else ''} ({image.width()} × {image.height()}) to {path}.")
        return path

    def picked_face(self):
        """The mesh face under the last pick (a cell of the main actor or of a textured surface actor)."""
        picked = self.picker.GetCellId()
        actor = self.picker.GetActor() if hasattr(self.picker, "GetActor") else self.actor
        faces = self.display_faces if actor is self.actor else self.picked_actor_faces.get(id(actor))
        if faces is None or picked < 0 or picked >= len(faces):
            return None
        return faces[picked]

    def texture(self, path):
        """A VTK texture for an image file, or None if it cannot be read (cached)."""
        if path not in self.texture_cache:
            texture = None
            reader = vtk.vtkImageReader2Factory.CreateImageReader2(str(path))
            if reader is not None:
                reader.SetFileName(str(path))
                reader.Update()
                if reader.GetOutput().GetNumberOfPoints() > 0:
                    texture = vtk.vtkTexture()
                    texture.SetInputConnection(reader.GetOutputPort())
                    texture.InterpolateOn()
                    texture.RepeatOn()
                    texture.MipmapOn()
            self.texture_cache[path] = texture
        return self.texture_cache[path]

    def clear_texture_actors(self):
        for actor, _ in self.texture_actors:
            self.renderer.RemoveActor(actor)
            self.picker.DeletePickList(actor)
        self.texture_actors = []
        self.picked_actor_faces = {}

    def build_texture_actors(self, faces):
        """Textured actors for the faces whose surface has a readable texture; returns the faces left plain."""
        self.clear_texture_actors()
        readable = {path for path in set(self.surfaces.textures) if path and self.texture(path) is not None}
        meshes = [entry for entry in surface_view.textured_meshes(self.mesh, self.surfaces, faces) if entry[0] in readable]
        drawn = set()
        for path, points, tcoords, polygons, ids in meshes:
            data = vtk.vtkPolyData()
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(numpy_to_vtk(points, deep=True))
            data.SetPoints(vtk_points)
            cells = vtk.vtkCellArray()
            for polygon in polygons:
                cells.InsertNextCell(len(polygon))
                for index in polygon:
                    cells.InsertCellPoint(index)
            data.SetPolys(cells)
            data.GetPointData().SetTCoords(numpy_to_vtk(tcoords, deep=True))
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetNumberOfTuples(len(ids))
            data.GetCellData().SetScalars(colors)
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(data)
            normals.SplittingOff()
            normals.ConsistencyOff()
            normals.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(normals.GetOutput())
            mapper.SetScalarModeToUseCellData()
            mapper.SetColorModeToDirectScalars()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetTexture(self.texture(path))
            actor.GetProperty().DeepCopy(self.actor.GetProperty())
            self.renderer.AddActor(actor)
            self.picker.AddPickList(actor)
            self.texture_actors.append((actor, list(ids)))
            self.picked_actor_faces[id(actor)] = list(ids)
            drawn.update(ids)
        return [face for face in faces if face not in drawn]

    def face_color_rows(self, faces, owner, textured):
        """Colors (n x 3) for faces: flat piece colors, or surface color / white texture tint with the piece tinted in."""
        mode = self.display_mode()
        if mode == FLAT:
            base = np.array([owner.get(face, REMAINDER_COLOR) for face in faces], dtype=np.float64).reshape(-1, 3)
        else:
            base = (np.full((len(faces), 3), 255.) if textured else self.surfaces.face_colors(faces))
            owned = np.array([face in owner for face in faces], dtype=bool)
            if owned.any():
                tint = np.array([owner[face] for face in faces if face in owner], dtype=np.float64)
                base[owned] = surface_view.tinted(base[owned], tint)
        selected = np.array([face in self.selection for face in faces], dtype=bool)
        if selected.any():
            base[selected] = SELECTION_COLOR
        return base.astype(np.uint8)

    def paint_selection(self):
        visible=self.visible_faces()
        mode=self.display_mode()
        if self.display_faces!=visible or getattr(self,'display_key',None)!=mode:
            self.display_key=mode
            if mode==TEXTURES:
                visible=self.build_texture_actors(visible)
            else:
                self.clear_texture_actors()
            self.display_faces=visible
            data=vtk.vtkPolyData();data.SetPoints(self.full_data.GetPoints())
            polygons=vtk.vtkCellArray()
            for face in visible:
                vertices=self.mesh.faces[face];polygons.InsertNextCell(len(vertices))
                for vertex in vertices:polygons.InsertCellPoint(vertex)
            data.SetPolys(polygons);data.GetPointData().ShallowCopy(self.full_data.GetPointData())
            self.colors=vtk.vtkUnsignedCharArray();self.colors.SetNumberOfComponents(3)
            self.colors.SetNumberOfTuples(len(visible));data.GetCellData().SetScalars(self.colors)
            self.mapper.SetInputData(data)
        owner = {}
        for index, piece in enumerate(self.pieces):
            for face in piece["faces"]:
                owner[face] = PIECE_COLORS[index % len(PIECE_COLORS)]
        if self.display_faces:
            rows = self.face_color_rows(self.display_faces, owner, textured=False)
            for cell, color in enumerate(rows):
                self.colors.SetTypedTuple(cell, tuple(int(c) for c in color))
        self.colors.Modified()
        for actor, ids in self.texture_actors:
            colors = actor.GetMapper().GetInput().GetCellData().GetScalars()
            for cell, color in enumerate(self.face_color_rows(ids, owner, textured=True)):
                colors.SetTypedTuple(cell, tuple(int(c) for c in color))
            colors.Modified()
        if self.texture_actors:
            self.update_view()
        self.update_status()
        if self.isVisible():
            self.window.Render()

    def update_status(self):
        label = self.source.name if self.source else "Demo mesh"
        details = f"{label}\n{len(self.selection):,} of {len(self.mesh.faces):,} polygons selected"
        if self.pieces:
            details += f"\n{len(self.pieces)} piece{'s' if len(self.pieces) > 1 else ''} + Remainder"
        if self.pivot is not None and not self.pieces:
            details += "\nPivot  " + ", ".join(f"{value:.5g}" for value in self.pivot)
        if self.ngon_count:
            details += f"\n{self.ngon_count} n-gon{'s' if self.ngon_count > 1 else ''} — use Split n-gons"
        self.status.setText(details)

    def clear_selection(self):
        self.selection.clear()
        self.mode = None
        self.clear_hover()
        self.paint_selection()

    def select_connected(self):
        expanded = set()
        for seed in sorted(self.selection):
            if seed not in expanded:
                expanded.update(self.mesh.connected(seed))
        self.selection = expanded.intersection(self.visible_faces())
        self.paint_selection()


def _qcolor(rgb):
    from PySide6.QtGui import QColor
    return QColor(*rgb)


def main(argv=None):
    from standalone_runtime import ensure_gui_streams
    ensure_gui_streams()
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-source", type=Path)
    parser.add_argument("--bridge-result", type=Path)
    parser.add_argument("--bridge-session", type=Path)
    parser.add_argument("--obj", type=Path, help="Open a standalone OBJ")
    parser.add_argument("--allow-ngons", action="store_true",
                        help="Accept polygons with more than four sides (no warning; exported as-is) "
                             "for bridges that support them. The DAZ bridge always needs triangles/quads.")
    parser.add_argument("--output", type=Path, help="Standalone/custom bridge: export pieces beside this .obj")
    parser.add_argument("--result", type=Path, help="Result manifest path (same folder as --output)")
    parser.add_argument("--after-export", type=Path,
                        help="Python script started with the completed result manifest path")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing earlier exports")
    parser.add_argument("--close-after", choices=("on", "off"),
                        help="Close after export/send (overrides the remembered checkbox for this run)")
    args = parser.parse_args(argv)
    if (args.bridge_source is None) != (args.bridge_result is None):
        parser.error("Both bridge paths are required.")
    if args.obj and args.bridge_source: parser.error('Use --obj or a DAZ bridge session, not both.')
    if args.bridge_session is not None and args.bridge_result is None:
        parser.error("--bridge-session needs the bridge paths.")
    config = {key: str(value.resolve()) for key, value in
              (("output", args.output), ("result", args.result), ("after_export", args.after_export)) if value}
    if args.overwrite: config["overwrite"] = True
    if args.close_after: config["close_after"] = args.close_after == "on"
    if args.bridge_source and config:
        parser.error("Export options are for standalone or custom bridges, not a DAZ session.")
    app = QApplication([sys.argv[0]])
    suite_app = str(Path(__file__).resolve().parents[2] / 'app')   # suite app/: taskbar icon
    if suite_app not in sys.path:
        sys.path.append(suite_app)
    import spin_icon
    spin_icon.set_app_icon(app, 'prop_split')
    selector = Selector(args.bridge_source, args.bridge_result, args.bridge_session, args.allow_ngons, config)
    if args.obj: selector.set_mesh(load_obj(args.obj), args.obj.resolve())
    selector.show()
    selector.viewport.Initialize()
    selector.viewport.Start()
    selector.frame()
    return app.exec()


if __name__ == '__main__': sys.exit(main())
