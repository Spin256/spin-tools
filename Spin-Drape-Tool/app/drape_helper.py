# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Interactive preview and material controls for meshes captured from DAZ Studio."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal, Slot, QEvent
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget, QSplitter,
                               QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
import vtk

import warp_drape
from warp_drape import PreparationCancelled, simulate
from drape_session import load_inputs, load_session
from drape_timing import DrapeSchedule
from drape_return import write_static_result, write_animation_result
from live_return import is_live, publish_revision, release_capture
from drape_animation import AnimationFrames
from garment_structure import GarmentStructure
from fabric_layers import layer_pair
from live_drape import LiveDrape, grab_patch
from drape_proxy import DEFAULT_SIZE as PROXY_SIZE, ProxyMesh, proxy_live
import drape_performance
from ground_scene import MODES as GROUND_MODES, ground_planes, lowest_point
from camera_navigation import turn_camera
from fabric_material import FABRICS
from fabric_selection import FabricSelection
from wind_settings import WindControl
from wind_controls import WindPanel
from wind_preview import WindPreview
from gravity_settings import GravityControl
from gravity_controls import GravityPanel
from water_settings import WaterControl
from water_controls import WaterPanel, WaterPreview
from branding import APP_NAME, VERSION, window_settings
import sys as _sys
from pathlib import Path as _Path
_SUITE_APP = str(_Path(__file__).resolve().parents[2] / "app")
if _SUITE_APP not in _sys.path:
    _sys.path.append(_SUITE_APP)          # suite app/: shared interface (spin_ui) and keybindings
import spin_ui


def polydata(vertices, faces):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3)
        for index in face:
            cells.InsertCellPoint(int(index))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh


def mesh_actor(renderer, mesh, color, opacity=1.0):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.SplittingOff()
    # DAZ meshes are consistently wound, so the consistency pass (slow on large garments) is skipped.
    normals.ConsistencyOff()
    normals.ComputeCellNormalsOff()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)
    return actor


class DrapeWorker(QThread):
    frame = Signal(object)
    failed = Signal(str)
    paused = Signal()
    preparing = Signal(str, float)          # preparation step and fraction, before the first frame

    def __init__(self, garment, colliders, settings, parent):
        super().__init__(parent)
        self.garment = garment
        self.colliders = colliders
        self.settings = dict(settings)
        # Proxy simulation: a lighter copy is simulated and its motion carried to the full mesh.
        self.proxy_size = self.settings.pop("proxy_size", None)
        self.rigid_motion = self.settings.pop("rigid_motion", True)
        ground = self.settings.get("ground")
        self.ground_lowest = ground.pop("lowest", None) if ground else None   # avatar colliders to measure
        self.ground_height_cm = ground["height_cm"] if ground else None
        self.proxy = self.settings.pop("proxy", None)        # reused from an earlier run when it matches
        self.animation = None
        self.frame_number = 0
        # True while the window is still drawing the last preview: newer frames replace the
        # pending one instead of making the solver wait (the pause/final frame always goes).
        self.preview_busy = False

    def use_proxy(self):
        """Build (or reuse) the proxy and map the run's settings onto it; returns the garment to simulate."""
        points, faces = self.garment
        key = (len(points), float(np.asarray(points[::997], np.float64).sum()))   # built around this shape
        if self.proxy is None or getattr(self.proxy, "key", None) != key or self.proxy.size != self.proxy_size:
            self.preparing.emit(f"building the proxy mesh (about {self.proxy_size:,} vertices)", .01)
            self.proxy = ProxyMesh(points, faces, self.proxy_size, self.settings.get("garment_ranges"),
                                   progress=lambda text, fraction: self.preparing.emit(text, .01))
            self.proxy.size = self.proxy_size
            self.proxy.key = key
        proxy, s = self.proxy, self.settings
        if s.get("garment_ranges") is not None:
            s["garment_ranges"] = proxy.ranges
        s["button_vertices"] = proxy.to_proxy(s.get("button_vertices") or []).tolist()
        s["pinned_vertices"] = proxy.to_proxy(s.get("pinned_vertices") or []).tolist()
        pairs = proxy.owner[np.asarray(s.get("fasteners") or [], dtype=np.int64).reshape(-1, 2)]
        s["fasteners"] = [tuple(pair) for pair in pairs.tolist() if pair[0] != pair[1]]
        cut = proxy.owner[np.asarray(s.get("cut_pairs") or [], dtype=np.int64).reshape(-1, 2)]
        s["cut_pairs"] = sorted({tuple(pair) for pair in cut.tolist() if pair[0] != pair[1]})
        live = s.get("interactive")
        if live is not None and hasattr(live, "proxy"):
            live.proxy = proxy
        return proxy.rest, proxy.faces

    def full(self, vertices_cm, colliders_cm=None):
        """Full-mesh vertices (cm) from simulated ones (cm); unchanged without a proxy. With a proxy
        the full mesh is also kept outside the colliders (fine fabric between proxy points could sit
        inside a leg), using the moving colliders' pose for this frame (``colliders_cm``)."""
        if self.proxy is None or len(vertices_cm) == self.proxy.full_count:
            return vertices_cm
        full = self.proxy.expand(np.asarray(vertices_cm) * .01) * 100.
        if not self.colliders:
            return full
        import threading
        from collider_guard import FullMeshGuard
        guards = self.__dict__.setdefault("full_guards", {})
        key = threading.get_ident()                   # one per thread: the worker records, the window shows
        if key not in guards:
            ground = self.settings.get("ground")
            guards[key] = FullMeshGuard(self.colliders, self.settings.get("thickness_mm", 4.0) * .0005,
                                        ground=None if not ground else dict(height=ground["height_cm"] * .01))
        return guards[key].project(full, colliders_cm)

    def run(self):
        pending = None
        last_preview = None
        def flush_preview(force=False):
            # Proxy frames go out as proxy vertices; the window carries the ones it shows to the
            # full mesh, so skipped frames cost nothing here.
            nonlocal pending, last_preview
            if pending is not None and (force or not self.preview_busy):
                self.preview_busy = True
                self.frame.emit(pending)
                pending = None
                last_preview = time.monotonic()

        def paused():
            flush_preview(force=True)
            self.paused.emit()

        try:
            motion = self.settings.get("motion")
            samples = getattr(motion, "samples", None)
            if motion is not None and samples is not None and hasattr(motion, "positions"):
                from rigid_motion import RigidPaths
                # Rigid limbs between DAZ frames (or straight lines when turned off).
                motion.rigid = RigidPaths(motion, {index: np.asarray(self.colliders[index][1])
                                                   for index in samples
                                                   if index < len(self.colliders)}) if self.rigid_motion else None
            if self.ground_lowest is not None and motion is not None:
                # Lowest point of the avatar over every timeline frame, not just the first.
                self.preparing.emit("finding the avatar's lowest point", .01)
                lowest = min(self.ground_height_cm, min(
                    float(motion.positions(i, frame)[:, 1].min()) * 100
                    for i in self.ground_lowest if i in motion.samples for frame in range(motion.intervals + 1))
                    if any(i in motion.samples for i in self.ground_lowest) else self.ground_height_cm)
                self.ground_height_cm = self.settings["ground"]["height_cm"] = lowest
            if motion is not None and self.settings.get("max_move_cm"):
                self.preparing.emit("measuring the avatar movement" +
                                    (" (finding rigid limbs)" if self.rigid_motion else ""), .01)
            if self.settings.get("loop") and motion is not None:
                self.settings["duration"] = 0.        # a loop has no final settling (warp_drape does the same)
            schedule = DrapeSchedule(
                motion, self.settings["duration"],
                self.settings.get("initial_duration", 0), self.settings.get("frame_settle", 0),
                self.settings.get("max_move_cm", 0),
            )
            self.duration = schedule.duration          # with split frames; the window shows it
            steps = iter(schedule)
            frame_number = 0
            live = self.settings.get("interactive")
            current_step = None
            def set_step(step):
                nonlocal current_step
                current_step = step
            if motion is not None and live is None:
                self.animation = AnimationFrames(motion, self.garment[0] * 100.0)

            def publish(vertices):
                nonlocal frame_number, pending
                frame_number += 1
                self.frame_number = frame_number
                # The loop's second pass sends steps beyond the schedule, so the step the solver reported is used.
                step = current_step if current_step is not None else next(steps)
                if motion is None and live is None:
                    pending = vertices
                else:
                    colliders = {}
                    if motion is not None:
                        rigid = getattr(motion, "rigid", None)
                        if step.parts > 1 and not step.hold and rigid is not None and step.interval >= 0:
                            fraction = (step.part + 1) / step.parts
                            colliders = {index: rigid.positions(index, step.interval, fraction) * 100.
                                         for index in motion.samples}
                        else:
                            colliders = motion.preview(step.sample)
                    # Split timeline frames: record only the last piece, when the DAZ frame is reached.
                    if (self.animation is not None and step.part == step.parts - 1
                            and not step.phase.startswith("Carrying")):
                        self.animation.put(step.sample, self.full(vertices, colliders))
                    pending = dict(
                        vertices=vertices, colliders=colliders,
                        time=frame_number / schedule.fps,
                        timeline_frame=motion.start_frame + step.sample if motion is not None else None, phase=step.phase,
                    )
                # Preview rendering blocks this worker. Keep every solver and
                # animation sample, but redraw at most 30 times per real second.
                if last_preview is None or time.monotonic() - last_preview >= 1./30:
                    flush_preview()

            garment = self.use_proxy() if self.proxy_size else self.garment
            if self.isInterruptionRequested():
                raise PreparationCancelled()
            def loop_finished(status):
                self.loop_status = status
            simulate(
                garment, self.colliders, publish,
                self.isInterruptionRequested, on_step=set_step, on_pause=paused, on_loop_status=loop_finished,
                on_prepare=self.preparing.emit, **self.settings,
            )
            flush_preview(force=True)
        except PreparationCancelled:
            pass                                 # cancelled by the user before the first frame
        except Exception as error:
            flush_preview(force=True)
            self.failed.emit(str(error))


class ReturnWorker(QThread):
    def __init__(self, spec, vertices, ranges, animation, parent, live_session=None):
        super().__init__(parent)
        self.spec, self.vertices, self.ranges, self.animation = spec, vertices, ranges, animation
        self.live_session = live_session
        self.error = None

    def write(self, spec):
        if self.animation is not None:
            write_animation_result(spec, self.animation, self.ranges)
        else:
            write_static_result(spec, self.vertices, self.ranges)

    def run(self):
        try:
            if self.live_session is None:
                raise ValueError("This capture has no Spin Tools receive session. Recapture it from DAZ.")
            # The helper stays open: publish a complete revision for the
            # Spin Tools receive window.
            publish_revision(self.live_session, self.spec, self.write)
        except Exception as error:
            self.error = str(error)


class DrapeWindow(QWidget):
    open_windows = []      # windows reopened after a cut (kept alive here)

    def __init__(
        self, garment, colliders, garment_label, collision_labels,
        collision_roles=None, garment_ranges=None, capture_frame=None, output_mode="static", motion=None,
        return_spec=None,
        button_vertices=None,
        water_source=None,
        live_session=None,
        garment_paths=None,
        cut_pairs=(),
        cut_release=None,
        uncut=None,
    ):
        super().__init__()
        # The window's own arguments: a garment cut reopens the window with the cut mesh.
        self.window_inputs = dict(
            garment=garment, colliders=colliders, garment_label=garment_label, collision_labels=collision_labels,
            collision_roles=collision_roles, garment_ranges=garment_ranges, capture_frame=capture_frame,
            output_mode=output_mode, motion=motion, return_spec=return_spec, button_vertices=button_vertices,
            water_source=water_source, live_session=live_session, garment_paths=garment_paths,
            cut_pairs=list(cut_pairs), cut_release=cut_release, uncut=uncut)
        self.cut_pairs = list(cut_pairs)
        self.cut_release = cut_release
        self.cut_picks = []
        self.live_session = live_session
        self.garment = garment
        self.colliders = colliders
        self.collision_labels = list(collision_labels)
        self.motion = motion
        self.return_spec = return_spec
        self.result_ready = False
        self.animation = None
        self.start_vertices = garment[0] * 100.0
        self.vertices = self.start_vertices.copy()
        self.garment_ranges = garment_ranges or [(0, len(garment[0]))]
        self.button_vertices = button_vertices or []
        self.structure = GarmentStructure(garment[0], garment[1], self.garment_ranges, self.button_vertices)
        self.fasteners = []
        self.pending_fastener = None
        self.bar_probe = None
        self.pinned = np.zeros(len(garment[0]), dtype=bool)      # no pins until the user adds them (Pin top 3 cm)
        self.selection = np.zeros(len(garment[0]), dtype=bool)
        self.frozen = self.selection.copy()
        self.freeze_targets = self.vertices.copy()
        self.selection_surface = FabricSelection(len(garment[0]), garment[1])
        self.paint_drag = False
        self.wind_control = WindControl()
        self.gravity_control = GravityControl()
        self.water_control = WaterControl()
        self.worker = None
        self.return_worker = None
        self.reset_pending = False
        self.close_pending = False
        self.simulation_failed = False
        self.frame_number = 0
        self.active_duration = 5.0
        self.live = None
        self.drag = None
        self.camera_drag = None
        self.live_ready = False
        self.live_at_end = False
        self.setWindowTitle(APP_NAME)
        self.resize(1200, 800)

        layout = QHBoxLayout(self)
        self.settings = window_settings('Helper')
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)
        panel = QWidget()
        controls = QVBoxLayout(panel)
        heading = QLabel(f"Garment: {garment_label}\nCollision meshes: {', '.join(collision_labels)}")
        heading.setTextFormat(Qt.TextFormat.PlainText)
        heading.setWordWrap(True)
        controls.addWidget(heading)
        if motion is not None:
            controls.addWidget(QLabel(
                f"Timeline {motion.start_frame} to {motion.end_frame} at {motion.fps:g} fps"
            ))
        elif capture_frame is not None:
            controls.addWidget(QLabel(f"Current-frame capture at frame {capture_frame}"))
        legend = QLabel(
            '<span style="color:#aaaaaa">Fixed colliders</span> · '
            '<span style="color:#b46ee6">Moving colliders</span> · '
            '<span style="color:#46a0ff">Garments</span>'
        )
        legend.setWordWrap(True)
        controls.addWidget(legend)

        shared_sections = spin_ui.Sections(controls, self.settings)
        self.sections = shared_sections.items
        section = shared_sections.add

        view_controls = section("View")
        # Shared View controls: garment and avatar show/hide + opacity; props have their own box.
        self.view = spin_ui.ViewControls(view_controls, self.settings, "garment", self.update_collider_visibility, tool="drape",
                                         avatar_name="avatar / moving objects")
        self.show_moving = self.view.show_avatar
        self.show_fixed = QCheckBox("Show props / fixed objects")
        self.show_fixed.setChecked(True)
        self.show_fixed.setToolTip("Preview visibility only. Hidden objects still participate in collisions.")
        view_controls.insertWidget(2, self.show_fixed)
        self.reset_view_button = self.view.reset_view_button
        self.reset_view_button.setToolTip("Restore the starting viewing angle and fit the garment and visible collision objects. The simulation is unchanged.")
        self.reset_view_button.clicked.connect(self.reset_view)
        camera_help = QLabel("Left-drag: orbit around the surface at screen center. Right-drag: turn in place. Middle-drag: pan. Wheel: zoom at pointer. Left-drag edits fabric when an editing mode is enabled.")
        camera_help.setWordWrap(True)
        view_controls.addWidget(camera_help)
        # One Simulation section: run and return settings, then solver and collision.
        simulation_controls = section("Simulation", True)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.output_mode = QComboBox()
        self.output_mode.addItem("Final static mesh", "static")
        self.output_mode.addItem("Animation sequence", "animation")
        self.output_mode.setCurrentIndex(1 if output_mode == "animation" else 0)
        if motion is None:
            form.addRow("Return preference", self.output_mode)
        self.interactive_enabled = QCheckBox("Interactive settling for static output")
        self.interactive_enabled.setChecked(True)
        self.interactive_enabled.setToolTip("Pause after the configured final settling time, keeping the live cloth state available to resume or tug. Uncheck for a timed run.")
        simulation_controls.addWidget(self.interactive_enabled)
        self.duration = self.number(0.1, 120, 5, 1, " s")
        self.initial_duration = self.number(0, 120, 2, 0.5, " s")
        self.initial_duration.setToolTip("Let the garment settle while the avatar remains in the starting pose.")
        self.frame_settle = self.number(0, 2, 1, 0.05, " s")
        self.frame_settle.setToolTip(
            "Hold each new avatar pose while the cloth settles. Zero follows the source timeline timing. "
            "Positive values add simulation time at each frame, rounded up to a full frame."
        )
        if motion is not None:
            self.duration.setMinimum(0)
            self.duration.setToolTip("Extra time to let the garment settle after the avatar reaches the final pose.")
        self.stretch = self.number(0, 20, 0, 0.5, " %")
        self.stretch.setToolTip("Extra edge length allowed before the cloth resists stretching. Zero gives the firmest result.")
        self.bending = self.number(0, 100, 20, 5, " %")
        self.bending.setToolTip("Resist tiny mesh-cell folds while retaining the starting shape and pleats. Zero restores the fully flexible behavior; higher values give stiffer fabric.")
        self.thickness = self.number(0.1, 20, 4, 0.1, " mm")
        self.thickness.setToolTip("Body collision thickness. Half this value separates garment vertices from collision objects. Cloth self-collision has its own gap below.")
        self.friction = self.number(0, 2, 0.6, 0.1)
        self.friction.setToolTip("Higher values resist sliding against collision meshes.")
        self.damping = self.number(0, 20, 2.4, 0.2, " /s")
        self.damping.setToolTip("Higher values make cloth motion settle faster.")
        self.iterations = QSpinBox()
        self.iterations.setRange(0, 100)
        self.iterations.setSpecialValueText("Auto")
        self.iterations.setValue(self.settings.value("solver/passes", 0, type=int))
        self.iterations.setToolTip(
            "Fabric passes per step. Higher keeps fabric closer to its real length (less sag) but is slower. "
            "Auto chooses from the garment's density (denser meshes need more). For very dense garments, "
            "Proxy simulation keeps length better and is faster.")
        self.iterations.valueChanged.connect(lambda value: self.settings.setValue("solver/passes", value))
        # 0.75 cm keeps fast-moving limbs from passing through the garment; larger moves let them
        # through, and 0 uses DAZ frames as they are.
        self.max_move = self.number(0, 20, .75, .25, " cm")
        self.max_move.setValue(self.settings.value("motion/max_move_cm", .75, type=float))
        self.max_move.setToolTip("Fast DAZ frames are split so the avatar moves at most this far per simulated frame, "
                                 "letting the cloth follow instead of the body passing through it (the simulation "
                                 "takes longer). 0 uses the DAZ frames as they are.")
        self.max_move.valueChanged.connect(lambda value: self.settings.setValue("motion/max_move_cm", value))
        self.rigid_motion = QCheckBox("Rigid limbs between frames")
        self.rigid_motion.setChecked(self.settings.value("motion/rigid", True, type=bool))
        self.rigid_motion.setToolTip("Between DAZ frames, limbs rotate about their joints and keep their length "
                                     "(found automatically from the frames, no rig needed), so a pose-to-pose "
                                     "timeline of just two frames works. Off: every point moves in a straight line.")
        self.rigid_motion.toggled.connect(lambda on: self.settings.setValue("motion/rigid", on))
        # Seamless loop.
        self.loop_animation = QCheckBox("Loop animation (carry motion)")
        self.loop_animation.setChecked(self.settings.value("motion/loop", False, type=bool))
        self.loop_animation.setToolTip("Keep simulating past the last frame into frames 1, 2, ... until the cloth's motion "
                                       "meets the first run, so the animation repeats without restarting from rest. The last "
                                       "frame leads into frame 0: end the avatar animation one pose before its first. "
                                       "Final settling is skipped.")
        self.loop_animation.toggled.connect(lambda on: self.settings.setValue("motion/loop", on))
        self.loop_frames = QSpinBox()
        self.loop_frames.setRange(2, 60)
        self.loop_frames.setValue(8)
        self.loop_frames.setToolTip("If the cloth has not met the first run by the middle of the animation, it is steered "
                                    "onto the closest frame over this many frames.")
        # DAZ return format for an animation: DUF by default, FBX selectable.
        self.return_format = spin_ui.return_format_box(self.settings)
        if motion is not None:
            form.addRow("Initial drape time", self.initial_duration)
            form.addRow("Avatar speed limit", self.max_move)
            form.addRow(self.rigid_motion)
            form.addRow("Extra settling per frame", self.frame_settle)
            # Playback of a finished animation, with the animation settings; shown in the
            # Animation section at the bottom.
            self.playback = spin_ui.PlaybackBar(self.show_recorded_frame)
        form.addRow("Final settling time" if motion is not None else "Settling time", self.duration)
        simulation_controls.addLayout(form)
        # Proxy simulation for heavy garments: off unless the user turns it on (remembered).
        self.use_proxy = QCheckBox("Proxy simulation (faster, for heavy garments)")
        self.use_proxy.setChecked(self.settings.value("proxy/enabled", False, type=bool))
        self.use_proxy.setToolTip("Simulate a lighter copy of the garment and carry its motion onto the full mesh. "
                                  "Much faster; new wrinkles are only as fine as the proxy, captured detail is kept. "
                                  "The result sent to DAZ is still the full original mesh.")
        self.proxy_size = QSpinBox()
        self.proxy_size.setRange(1000, 150000)
        self.proxy_size.setSingleStep(5000)
        self.proxy_size.setValue(self.settings.value("proxy/size", PROXY_SIZE, type=int))
        self.proxy_size.setSuffix(" vertices")
        self.proxy_size.setToolTip("About how many vertices the simulated copy has. More: finer wrinkles, slower.")
        simulation_controls.addWidget(self.use_proxy)
        proxy_form = QFormLayout()
        proxy_form.addRow("Proxy size", self.proxy_size)
        simulation_controls.addLayout(proxy_form)
        # The full size advice stays here (wrapped); the status bar has a short version.
        self.proxy_advice = spin_ui.note("")
        simulation_controls.addWidget(self.proxy_advice)
        self.use_proxy.toggled.connect(lambda on: self.settings.setValue("proxy/enabled", on))
        self.use_proxy.toggled.connect(lambda *_: self.switch_proxy_while_paused())
        self.continue_from = None
        self.use_proxy.toggled.connect(self.proxy_size.setEnabled)
        self.use_proxy.toggled.connect(lambda *_: hasattr(self, "status") and self.show_size_advice())
        self.proxy_size.valueChanged.connect(lambda value: self.settings.setValue("proxy/size", value))
        self.proxy_size.valueChanged.connect(lambda *_: hasattr(self, "status") and self.show_size_advice())
        self.proxy_size.setEnabled(self.use_proxy.isChecked())
        self.proxy = None
        material_controls = section("Fabric", True)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.fabric_preset = QComboBox()
        self.fabric_preset.addItem("Current / custom")
        self.fabric_preset.addItems(list(FABRICS))
        self.fabric_preset.setToolTip("Approximate starting points. Adjust any value to customize the fabric.")
        self.fabric_weight = self.number(10, 2000, 250, 25, " g/m²")
        self.edge_width = self.number(0, 100, 0, 1, " mm")
        self.edge_width.setToolTip("Reinforced band measured inward along the cloth from all open mesh boundaries. Zero disables it.")
        self.edge_stiffness = self.number(0, 100, 75, 5, " %")
        self.edge_stiffness.setToolTip("Extra bend resistance throughout the edge band, above the main fabric setting. Applies equally to all open boundaries.")
        form.addRow("Fabric preset", self.fabric_preset)
        form.addRow("Stretch allowance", self.stretch)
        form.addRow("Bend resistance", self.bending)
        form.addRow("Fabric weight", self.fabric_weight)
        form.addRow("Edge reinforcement width", self.edge_width)
        form.addRow("Edge stiffness", self.edge_stiffness)
        self.crease_enabled = QCheckBox("Reinforce sharp folds")
        self.crease_angle = self.number(1, 179, 30, 5, "°")
        self.crease_stiffness = self.number(0, 100, 50, 5, " %")
        self.crease_angle.setToolTip("Only internal edges whose captured fold angle exceeds this threshold are reinforced. Flat is 0°; a fully folded-back edge is 180°. Evaluated at the start, not from wrinkles forming during simulation.")
        self.crease_stiffness.setToolTip("Extra bend resistance at detected creases, retaining their captured shape. The stronger of crease and open-edge reinforcement applies where they overlap.")
        form.addRow(self.crease_enabled)
        form.addRow("Minimum fold angle", self.crease_angle)
        form.addRow("Crease stiffness", self.crease_stiffness)
        form.addRow("Damping", self.damping)
        material_controls.addLayout(form)
        fabric_note = QLabel("Presets are adjustable starting points. Higher bend resistance gives firmer fabric. Edge width 0 disables reinforcement; all open edges share the same settings.")
        fabric_note.setWordWrap(True)
        material_controls.addWidget(fabric_note)
        solver_controls = simulation_controls            # same section, own sub-heading
        heading = QLabel("Solver and collision")
        heading.setStyleSheet("font-weight: bold; margin-top: 6px;")
        solver_controls.addWidget(heading)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow("Solver passes", self.iterations)
        form.addRow("Collision thickness", self.thickness)
        form.addRow("Friction", self.friction)
        self.self_collision = QCheckBox("Self-collision (experimental)")
        self.self_collision_gap = self.number(.1, 10, 1, .1, " mm")
        self.self_collision_passes = QSpinBox()
        self.self_collision_passes.setRange(1, 8)
        self.self_collision_passes.setValue(2)
        self.self_collision.setToolTip("Keep non-neighboring fabric vertices away from triangle surfaces. Slower; an initial approximation, not full continuous collision detection.")
        form.addRow(self.self_collision)
        form.addRow("Cloth separation", self.self_collision_gap)
        form.addRow("Self-collision passes", self.self_collision_passes)
        form.addRow(spin_ui.note("Self-collision is experimental: fast crossings and edge-only intersections "
                                 "can still pass through."))
        solver_controls.addLayout(form)
        # Ground: an infinite floor (Y up), optional, with its own grip; can be sent to DAZ like water.
        ground_heading = QLabel("Ground")
        ground_heading.setStyleSheet("font-weight: bold; margin-top: 6px;")
        solver_controls.addWidget(ground_heading)
        self.ground_prop_heights = ground_planes(colliders, collision_labels)
        self.ground_mode = QComboBox()
        for key, label in GROUND_MODES:
            self.ground_mode.addItem(label, key)
        remembered = self.settings.value("ground/mode", "off")
        start_mode = "plane" if self.ground_prop_heights else (remembered if remembered != "plane" else "off")
        self.ground_mode.setCurrentIndex(max(0, self.ground_mode.findData(start_mode)))
        self.ground_mode.setToolTip("An infinite floor the fabric rests on. Lowest point of the avatar uses her lowest "
                                    "point over every timeline frame; a flat captured prop named Ground… sets the "
                                    "height when one was captured.")
        self.ground_custom = self.number(-10000, 10000, self.settings.value("ground/height", 0., type=float), 1, " cm")
        self.ground_grip = self.number(0, 100, self.settings.value("ground/grip", 50., type=float), 5, " %")
        self.ground_grip.setToolTip("How firmly fabric holds where it touches the ground: 0% slides freely, 100% stays put.")
        self.ground_send = QCheckBox("Send ground plane to DAZ")
        self.ground_send.setChecked(self.settings.value("ground/send", False, type=bool))
        self.ground_send.setToolTip("Also return a flat ground plane at this height, like the water-height plane. "
                                    "Not offered when a Ground prop was captured (it is already in the scene).")
        self.ground_note = spin_ui.note("")
        ground_form = QFormLayout()
        ground_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        ground_form.addRow("Ground", self.ground_mode)
        ground_form.addRow("Custom height", self.ground_custom)
        ground_form.addRow("Ground grip", self.ground_grip)
        ground_form.addRow(self.ground_send)
        ground_form.addRow(self.ground_note)
        solver_controls.addLayout(ground_form)
        self.applying_fabric = False
        self.fabric_preset.currentTextChanged.connect(self.apply_fabric_preset)
        for widget in (self.stretch, self.bending, self.fabric_weight, self.damping, self.friction):
            widget.valueChanged.connect(self.mark_custom_fabric)
        self.output_note = QLabel()
        self.output_note.setWordWrap(True)
        self.output_mode.currentIndexChanged.connect(self.update_output_note)
        self.update_output_note()
        simulation_controls.addWidget(self.output_note)

        fabric_controls = section("Fabric attachments and fastenings")
        self.keep_details = QCheckBox("Keep pockets and buttons attached")
        self.keep_details.setChecked(True)
        self.keep_details.setToolTip("Small separate mesh pieces follow the fabric underneath. Large separate panels remain cloth.")
        fabric_controls.addWidget(self.keep_details)
        self.fasten_buttons = QCheckBox("Fasten fabric at detected buttons")
        self.fasten_buttons.setChecked(True)
        self.fasten_buttons.setToolTip("Connect nearby front edges at button locations using the fastening mode below. Both sides can move. Turn off to let the garment open.")
        fabric_controls.addWidget(self.fasten_buttons)
        self.detail_note = QLabel(
            f"{len(self.structure.detail_groups)} separate details; {len(self.structure.fasteners)} button fastenings detected."
        )
        self.detail_note.setWordWrap(True)
        fabric_controls.addWidget(self.detail_note)
        self.fastening_mode = QComboBox()
        self.fastening_mode.addItem("Close gap", "close")
        self.fastening_mode.addItem("Keep starting distance", "keep")
        self.fastening_gap = self.number(0, 50, 1, .5, " mm")
        self.fastening_gap.setToolTip("Maximum target gap for closed fastenings. Applies to detected button fastenings and custom pairs.")
        fastening_form = QFormLayout()
        fastening_form.addRow("Fastening mode", self.fastening_mode)
        fastening_form.addRow("Closed gap", self.fastening_gap)
        fabric_controls.addLayout(fastening_form)
        self.fastening_mode.currentIndexChanged.connect(
            lambda: self.fastening_gap.setEnabled(self.fastening_mode.currentData() == "close"))

        self.fastener_pick_mode = QComboBox()
        self.fastener_pick_mode.addItem("Two clicks", "pair")
        self.fastener_pick_mode.addItem("Through layers (normal bar)", "bar")
        self.fastener_pick_mode.setCurrentIndex(1)
        self.bar_reach = self.number(1, 100, 20, 1, " mm")
        self.bar_reach.setToolTip("Search this far in either direction along the clicked face normal. The closest separate fabric layer is joined.")
        fastening_form.addRow("Picking method", self.fastener_pick_mode)
        fastening_form.addRow("Bar reach each side", self.bar_reach)
        fastening_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.fastener_pick_mode.currentIndexChanged.connect(lambda: setattr(self, "pending_fastener", None))
        self.fabric_edit = QCheckBox("Edit fabric fastenings by clicking")
        fabric_controls.addWidget(self.fabric_edit)
        self.fastener_count = QLabel("No custom fabric fastenings")
        self.fastener_count.setWordWrap(True)
        fabric_controls.addWidget(self.fastener_count)
        self.clear_fasteners = QPushButton("Clear custom fabric fastenings")
        fabric_controls.addWidget(self.clear_fasteners)
        fastener_help = QLabel("Reset first. Through layers joins the clicked face to the closest fabric layer along its normal, within the bar reach. Two clicks: click one fabric point on each side of the opening, then run the drape. Cyan points and lines show all enabled fastenings. Close gap draws the points together; Keep starting distance retains their original spacing. Shift-click edits avatar pins instead.")
        fastener_help.setWordWrap(True)
        fabric_controls.addWidget(fastener_help)

        cut_form = QFormLayout()
        self.cut_edit = QCheckBox("Pick cut points by clicking")
        self.cut_edit.setToolTip("Click points along the line to cut, in order. The cut follows the mesh edges "
                                 "between them; run it to a garment edge or around a loop so the sides come apart.")
        fabric_controls.addWidget(self.cut_edit)
        self.cut_release_frame = QSpinBox()
        self.cut_release_frame.setToolTip("The two sides stay joined until this frame, then let go "
                                          "(a clasp opening). The first frame cuts from the start.")
        if motion is not None:
            self.cut_release_frame.setRange(motion.start_frame, motion.end_frame)
            self.cut_release_frame.setValue(cut_release if cut_release is not None else motion.start_frame)
            cut_form.addRow("Cut opens at frame", self.cut_release_frame)
        fabric_controls.addLayout(cut_form)
        cut_buttons = QHBoxLayout()
        self.cut_apply = QPushButton("Apply cut")
        self.cut_clear = QPushButton("Clear points")
        self.cut_undo = QPushButton("Remove all cuts")
        for widget in (self.cut_apply, self.cut_clear, self.cut_undo):
            cut_buttons.addWidget(widget)
        fabric_controls.addLayout(cut_buttons)
        self.cut_note = QLabel()
        self.cut_note.setWordWrap(True)
        fabric_controls.addWidget(self.cut_note)
        self.cut_apply.clicked.connect(self.apply_cut)
        self.cut_clear.clicked.connect(self.clear_cut_points)
        self.cut_undo.clicked.connect(self.remove_cuts)

        pin_controls = section("Avatar pins")
        self.pin_edit = QCheckBox("Edit pins by clicking the dress")
        self.fabric_edit.toggled.connect(lambda checked: self.pin_edit.setChecked(False) if checked else None)
        self.pin_edit.toggled.connect(lambda checked: self.fabric_edit.setChecked(False) if checked else None)
        pin_controls.addWidget(self.pin_edit)
        instructions = QLabel(
            "Shift-click also edits pins. Orange points are pinned. In timeline mode they follow their "
            "closest collision surface. Click a pinned area again to unpin it."
        )
        instructions.setWordWrap(True)
        pin_controls.addWidget(instructions)
        self.brush = self.number(0, 100, 5, 1, " mm")
        self.brush.setToolTip("Radius around the clicked vertex. Zero edits one vertex.")
        brush_form = QFormLayout()
        brush_form.addRow("Pin brush radius", self.brush)
        pin_controls.addLayout(brush_form)
        self.pin_top = QPushButton("Pin top 3 cm")
        self.pin_clear = QPushButton("Clear all pins")
        self.pin_count = QLabel()
        pin_controls.addWidget(self.pin_top)
        pin_controls.addWidget(self.pin_clear)
        pin_controls.addWidget(self.pin_count)

        live_controls = section("Live settling and grabbing", True)
        self.hold_frame = QPushButton("Hold frame")
        self.hold_frame.setCheckable(True)
        self.pause_simulation = QPushButton("Pause simulation")
        self.pause_simulation.setCheckable(True)
        self.grab_edit = QCheckBox("Grab fabric by dragging")
        self.grab_radius = self.number(0, 100, 15, 1, " mm")
        self.grab_strength = self.number(1, 100, 80, 5, " %")
        live_controls.addWidget(self.hold_frame)
        live_controls.addWidget(self.pause_simulation)
        # Keep the checkable action for shortcuts/state, with one visible control
        # in the always-visible main action area below the sections.
        self.pause_simulation.hide()
        live_controls.addWidget(self.grab_edit)
        live_form = QFormLayout()
        live_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        live_form.addRow("Grab radius", self.grab_radius)
        live_form.addRow("Grab strength", self.grab_strength)
        live_controls.addLayout(live_form)
        live_help = QLabel("Static output only. Final settling time defaults to 5 seconds, then pauses automatically. Pause freezes the current cloth state; Resume continues without resetting. While it runs, drag unpinned fabric to tug it; drag anywhere else, or while paused, to move the view. Hold frame stops only the avatar. You can send any paused result to DAZ. Reset is separate.")
        live_help.setWordWrap(True)
        live_controls.addWidget(live_help)
        self.hold_frame.toggled.connect(self.set_frame_hold)
        self.pause_simulation.toggled.connect(self.set_simulation_pause)
        self.grab_edit.toggled.connect(lambda checked: self.release_grab() if not checked else None)
        self.interactive_enabled.toggled.connect(self.update_live_controls)

        gravity_controls = section("Gravity")
        self.gravity_panel = GravityPanel()
        self.gravity_panel.changed.connect(self.gravity_control.set)
        gravity_controls.addWidget(self.gravity_panel)

        wind_controls = section("Wind")
        self.wind_panel = WindPanel(self.start_vertices)
        wind_controls.addWidget(self.wind_panel)

        water_controls = section("Water")
        self.water_panel = WaterPanel(self.start_vertices, water_source=water_source)
        water_controls.addWidget(self.water_panel)

        paint_controls = section("Paint and freeze fabric")
        self.paint_edit = QCheckBox("Paint fabric selection")
        self.paint_mode = QComboBox()
        self.paint_mode.addItems(["Add", "Remove"])
        self.paint_radius = self.number(0, 100, 15, 1, " mm")
        self.selection_count = QLabel()
        self.selection_count.setWordWrap(True)
        paint_controls.addWidget(self.paint_edit)
        paint_form = QFormLayout()
        paint_form.addRow("Paint mode", self.paint_mode)
        paint_form.addRow("Brush radius", self.paint_radius)
        paint_controls.addLayout(paint_form)
        self.grow_selection = QPushButton("Grow selection")
        self.shrink_selection = QPushButton("Shrink selection")
        self.clear_selection = QPushButton("Clear selection")
        self.freeze_selected = QPushButton("Freeze selected")
        self.freeze_unselected = QPushButton("Freeze unselected")
        self.unfreeze_selected = QPushButton("Unfreeze selected")
        self.unfreeze_all = QPushButton("Unfreeze all")
        for widget in (self.grow_selection, self.shrink_selection, self.clear_selection,
                       self.freeze_selected, self.freeze_unselected, self.unfreeze_selected, self.unfreeze_all):
            paint_controls.addWidget(widget)
        paint_controls.addWidget(self.selection_count)
        self.keep_freeze_timeline = QCheckBox("Keep frozen fabric when continuing timeline")
        self.keep_freeze_timeline.setChecked(False)
        self.keep_freeze_timeline.setToolTip("Off: release world-frozen cloth before avatar movement continues. On: keep it fixed in world space throughout the movement. Resuming cloth at a held/final frame always preserves freezing.")
        paint_controls.addWidget(self.keep_freeze_timeline)
        paint_help = QLabel("Static output only. Painting holds the avatar frame and pauses the solver. Green: selected; purple: frozen; yellow: selected and frozen. Grow/shrink changes the selection by one vertex ring. Freeze selected or unselected, then resume to settle the rest. Frozen cloth stays at its world position and does not follow a moving avatar.")
        paint_help.setWordWrap(True)
        paint_controls.addWidget(paint_help)
        self.shortcut_note = QLabel()
        self.shortcut_note.setWordWrap(True)
        paint_controls.addWidget(self.shortcut_note)
        self.paint_edit.toggled.connect(self.set_paint_mode)
        for mode in (self.grab_edit, self.pin_edit, self.fabric_edit):
            mode.toggled.connect(lambda checked: self.paint_edit.setChecked(False) if checked else None)
        self.grow_selection.clicked.connect(lambda: self.resize_selection(True))
        self.shrink_selection.clicked.connect(lambda: self.resize_selection(False))
        self.clear_selection.clicked.connect(self.clear_painted_selection)
        self.freeze_selected.clicked.connect(lambda: self.apply_freeze('selected'))
        self.freeze_unselected.clicked.connect(lambda: self.apply_freeze('unselected'))
        self.unfreeze_selected.clicked.connect(lambda: self.apply_freeze('remove'))
        self.unfreeze_all.clicked.connect(lambda: self.apply_freeze('clear'))
        if motion is not None:
            # Animation: return preference, loop, playback and DAZ return format together,
            # at the bottom.
            animation_controls = section("Animation", True)
            animation_form = QFormLayout()
            animation_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            animation_form.addRow("Return preference", self.output_mode)
            animation_form.addRow(self.loop_animation)
            animation_form.addRow("Loop landing frames", self.loop_frames)
            animation_form.addRow("Playback", self.playback)
            animation_form.addRow("DAZ return format", self.return_format)
            animation_controls.addLayout(animation_form)

        # Shared bottom row (Run/Pause, Reset, Send, Close after sending). Close after is shown
        # only where a Send can leave the tool open (live DAZ, standalone).
        self.actions = spin_ui.ActionRow(self.settings, send_text="Send static result to DAZ", tool="drape")
        self.run_button, self.reset_button = self.actions.run_button, self.actions.reset_button
        self.send_button, self.close_after = self.actions.send_button, self.actions.close_after
        self.reset_button.setToolTip("Reset the garment to its captured position. Pins and material settings are kept.")
        self.send_button.setEnabled(False)
        self.send_button.setToolTip("Send the paused static or completed animation result to the Spin Tools receive window in DAZ"
                                    + spin_ui.key_hint("drape", "send_to_daz") + ".")
        self.stop_button = QPushButton("Stop")          # timed (non-interactive) runs only
        self.stop_button.setEnabled(False)
        self.stop_button.setVisible(False)              # shown only while a timed run is going
        self.stop_button.setToolTip("Stop this timed run (with interactive settling off there is no pause).")
        self.run_button.parentWidget().layout().itemAt(0).layout().insertWidget(1, self.stop_button)
        self.close_after.setVisible(False)
        self.close_after_key = None
        self.actions.close_after_key = None
        self.actions.remember = self.remember_close_after
        controls.addStretch()
        column, self.copyright_footer = spin_ui.control_column(panel, self.actions, 380, version=VERSION, tool_folder=Path(__file__).resolve().parent.parent)
        self.splitter.addWidget(column)

        self.viewport = QVTKRenderWindowInteractor(self)
        self.viewport.installEventFilter(self)
        holder, self.status = spin_ui.viewport_with_status(self.viewport)
        self.progress = self.status
        self.preparing = False
        self.run_started = self.first_frame_time = None
        self.show_size_advice()
        self.splitter.addWidget(holder)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([420, 900])
        spin_ui.remember_window(self, self.settings, self.splitter)
        self.renderer = spin_ui.style_renderer(vtk.vtkRenderer())
        self.viewport.GetRenderWindow().AddRenderer(self.renderer)
        self.wind_preview = WindPreview(self.renderer)
        self.wind_panel.changed.connect(self.update_wind)
        self.wind_control.set(self.wind_panel.settings())
        self.wind_preview.update(self.wind_panel.settings())
        self.water_preview = WaterPreview(self.renderer, self.start_vertices)
        self.water_panel.changed.connect(self.update_water)
        self.water_control.set(self.water_panel.settings())
        self.water_preview.update(self.water_panel.settings())
        collider_roles = collision_roles or ["fixed"] * len(colliders)
        self.collider_meshes = []
        self.collider_actors = []
        self.collider_roles = collider_roles
        for (points, faces), role in zip(colliders, collider_roles):
            color = spin_ui.MOVING_COLOR if role == "moving" else spin_ui.AVATAR_COLOR
            mesh = polydata(points * 100, faces)
            self.collider_meshes.append(mesh)
            self.collider_actors.append(mesh_actor(self.renderer, mesh, color, 0.55))
        self.show_fixed.toggled.connect(self.update_collider_visibility)
        self.mesh = polydata(self.vertices, garment[1])
        self.garment_actor = mesh_actor(self.renderer, self.mesh, spin_ui.OBJECT_COLOR)
        # Ground preview: a translucent square at the ground height (View: Show ground).
        self.ground_source = vtk.vtkPlaneSource()
        ground_mapper = vtk.vtkPolyDataMapper()
        ground_mapper.SetInputConnection(self.ground_source.GetOutputPort())
        self.ground_actor = vtk.vtkActor()
        self.ground_actor.SetMapper(ground_mapper)
        self.ground_actor.GetProperty().SetColor(.45, .5, .42)
        self.ground_actor.GetProperty().SetOpacity(.35)
        self.ground_actor.PickableOff()
        self.renderer.AddActor(self.ground_actor)
        self.show_ground = QCheckBox("Show ground")
        self.show_ground.setChecked(self.settings.value("view/ground", True, type=bool))
        self.show_ground.toggled.connect(lambda on: self.settings.setValue("view/ground", on))
        self.show_ground.toggled.connect(lambda *_: self.update_ground())
        self.view.reset_view_button.parentWidget().layout().addWidget(self.show_ground)
        for widget in (self.ground_mode, self.ground_send):
            (widget.currentIndexChanged if widget is self.ground_mode else widget.toggled).connect(
                lambda *_: self.update_ground())
        self.ground_custom.valueChanged.connect(lambda *_: self.update_ground())
        self.ground_grip.valueChanged.connect(lambda value: self.settings.setValue("ground/grip", value))
        self.ground_frame_height = None
        self.update_ground(render=False)

        self.fastener_mesh = vtk.vtkPolyData()
        self.fastener_points = vtk.vtkPoints()
        self.fastener_mesh.SetPoints(self.fastener_points)
        fastener_mapper = vtk.vtkPolyDataMapper()
        fastener_mapper.SetInputData(self.fastener_mesh)
        fastener_actor = vtk.vtkActor()
        fastener_actor.SetMapper(fastener_mapper)
        fastener_actor.GetProperty().SetColor(0.1, 1.0, 1.0)
        fastener_actor.GetProperty().SetLineWidth(3)
        fastener_actor.GetProperty().SetPointSize(8)
        fastener_actor.GetProperty().LightingOff()
        fastener_actor.PickableOff()
        self.renderer.AddActor(fastener_actor)

        self.bar_source = vtk.vtkLineSource()
        bar_mapper = vtk.vtkPolyDataMapper()
        bar_mapper.SetInputConnection(self.bar_source.GetOutputPort())
        self.bar_actor = vtk.vtkActor()
        self.bar_actor.SetMapper(bar_mapper)
        self.bar_actor.GetProperty().SetColor(1.0, .9, .2)
        self.bar_actor.GetProperty().SetLineWidth(3)
        self.bar_actor.GetProperty().LightingOff()
        self.bar_actor.PickableOff()
        self.bar_actor.SetVisibility(False)
        self.renderer.AddActor(self.bar_actor)
        self.bar_reach.valueChanged.connect(self.refresh_bar)

        self.pin_mesh = vtk.vtkPolyData()
        self.pin_points = vtk.vtkPoints()
        self.pin_mesh.SetPoints(self.pin_points)
        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(self.pin_mesh)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.45, 0.05)
        actor.GetProperty().SetPointSize(6)
        actor.GetProperty().LightingOff()
        actor.PickableOff()
        self.renderer.AddActor(actor)

        self.grab_mesh = vtk.vtkPolyData()
        self.grab_points = vtk.vtkPoints()
        self.grab_mesh.SetPoints(self.grab_points)
        grab_glyph = vtk.vtkVertexGlyphFilter()
        grab_glyph.SetInputData(self.grab_mesh)
        grab_mapper = vtk.vtkPolyDataMapper()
        grab_mapper.SetInputConnection(grab_glyph.GetOutputPort())
        self.grab_actor = vtk.vtkActor()
        self.grab_actor.SetMapper(grab_mapper)
        self.grab_actor.GetProperty().SetColor(1., .2, .8)
        self.grab_actor.GetProperty().SetPointSize(7)
        self.grab_actor.GetProperty().LightingOff()
        self.grab_actor.PickableOff()
        self.renderer.AddActor(self.grab_actor)

        self.paint_overlays = []
        for color in ((.2, 1., .35), (.7, .35, 1.), (1., .9, .2)):
            mesh = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            mesh.SetPoints(points)
            glyph = vtk.vtkVertexGlyphFilter()
            glyph.SetInputData(mesh)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetPointSize(5)
            actor.GetProperty().LightingOff()
            actor.PickableOff()
            self.renderer.AddActor(actor)
            self.paint_overlays.append(points)

        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.viewport.SetInteractorStyle(self.style)
        self.style.AddObserver("LeftButtonPressEvent", self.pick_pin)
        self.style.AddObserver("LeftButtonReleaseEvent", self.end_mouse_drag)
        self.style.AddObserver("RightButtonPressEvent", lambda caller, event: self.begin_camera_drag('look'))
        self.style.AddObserver("RightButtonReleaseEvent", self.end_camera_drag)
        self.style.AddObserver("MouseMoveEvent", self.move_mouse_drag)
        self.style.AddObserver("MouseWheelForwardEvent", lambda caller, event: self.zoom_at_pointer(1.2))
        self.style.AddObserver("MouseWheelBackwardEvent", lambda caller, event: self.zoom_at_pointer(1 / 1.2))
        self.picker = vtk.vtkCellPicker()
        self.update_cut_note()
        self.picker.PickFromListOn()
        self.picker.AddPickList(self.garment_actor)
        self.picker.SetTolerance(0.003)

        self.edit_controls = [
            self.cut_edit, self.cut_release_frame,
            self.duration, self.initial_duration, self.frame_settle, self.loop_animation, self.loop_frames,
            self.stretch, self.bending, self.thickness, self.friction, self.damping,
            self.self_collision, self.self_collision_gap, self.self_collision_passes,
            self.iterations, self.pin_edit, self.brush, self.pin_top, self.pin_clear, self.max_move, self.rigid_motion,
            self.keep_details, self.fasten_buttons, self.fabric_edit, self.clear_fasteners,
            self.fastening_mode, self.fastening_gap, self.fastener_pick_mode, self.bar_reach,
            self.fabric_preset, self.fabric_weight, self.edge_width, self.edge_stiffness,
            self.crease_enabled, self.crease_angle, self.crease_stiffness,
            self.use_proxy, self.proxy_size,
        ]
        self.run_button.clicked.connect(self.start_drape)
        self.stop_button.clicked.connect(self.stop_drape)
        self.reset_button.clicked.connect(self.reset_dress)
        self.send_button.clicked.connect(self.send_static_result)
        self.pin_clear.clicked.connect(self.clear_pins)
        self.pin_top.clicked.connect(self.select_top)
        self.clear_fasteners.clicked.connect(self.clear_fabric_fasteners)
        self.fasten_buttons.toggled.connect(self.refresh_fasteners)
        self.renderer.ResetCamera()
        self.start_camera = vtk.vtkCamera()
        self.start_camera.DeepCopy(self.renderer.GetActiveCamera())
        self.camera = spin_ui.ViewportCamera(self.renderer, self.viewport.GetRenderWindow().Render,
                                             lambda: [self.garment_actor] + self.collider_actors)
        self.update_collider_visibility(render=False)        # no drawing before the window is shown
        self.view.lighting.attach(self.renderer, self.viewport.GetRenderWindow().Render)
        self.refresh_pins()
        self.update_output_note()
        self.update_live_controls()
        self.refresh_selection()
        self.setup_shortcuts()
        if self.live_session is not None:
            self.configure_close_after('daz', "Close after sending", False,
                                       "Off: minimize after each send so you can inspect the result in DAZ and send again. "
                                       "On: exit once the result is written.")

    def configure_close_after(self, context, label, default, tooltip, override=None):
        """Remembered per context; an explicit override (command line/project) is not saved."""
        self.actions.configure_close_after(context, label, default, tooltip, override)
        self.close_after_key = self.actions.close_after_key
        self.close_after.setVisible(True)

    def remember_close_after(self, checked):
        if self.close_after_key:
            self.settings.setValue(self.close_after_key, checked)

    def ground_avatar(self):
        """Indices of the avatar colliders (moving ones, or all), Ground/Water planes excluded."""
        chosen = [i for i, role in enumerate(self.collider_roles) if role == "moving"] or list(range(len(self.colliders)))
        return [i for i in chosen
                if not str(self.collision_labels[i]).strip().casefold().startswith(("ground", "water"))]

    def ground_height(self):
        """Ground height (cm) for the current setting, or None for Off."""
        mode = self.ground_mode.currentData()
        if mode == "off":
            return None
        if mode == "floor":
            return 0.
        if mode == "plane":
            return self.ground_prop_heights[0] if self.ground_prop_heights else None
        if mode == "custom":
            return self.ground_custom.value()
        if self.ground_frame_height is not None:          # measured over every frame by the last run
            return self.ground_frame_height
        return lowest_point(self.colliders, self.collider_roles, self.collision_labels)

    def update_ground(self, render=True):
        mode = self.ground_mode.currentData()
        self.settings.setValue("ground/mode", mode)
        self.settings.setValue("ground/height", self.ground_custom.value())
        self.settings.setValue("ground/send", self.ground_send.isChecked())
        self.ground_custom.setEnabled(mode == "custom" and self.worker is None)
        self.ground_grip.setEnabled(mode != "off")
        self.ground_send.setEnabled(mode not in ("off", "plane") and self.return_spec is not None)
        height = self.ground_height()
        note = ""
        if mode == "plane":
            note = (f"From the captured Ground prop: {height:.1f} cm." if height is not None else
                    "No flat prop named Ground… was captured.")
        elif mode == "lowest":
            note = ("No avatar captured." if height is None else f"Avatar's lowest point: {height:.1f} cm" +
                    (" (over every timeline frame)." if self.ground_frame_height is not None or self.motion is None
                     else " in the first frame; measured over every frame when the run starts."))
        self.ground_note.setText(note)
        self.ground_note.setVisible(bool(note))
        if height is None:
            self.ground_actor.SetVisibility(False)
        else:
            points = np.concatenate([self.start_vertices] + [np.asarray(points) * 100 for points, _ in self.colliders])
            centre, size = points.mean(0), float(np.ptp(points[:, [0, 2]], axis=0).max()) * .75 + 50
            self.ground_source.SetOrigin(centre[0] - size, height, centre[2] - size)
            self.ground_source.SetPoint1(centre[0] + size, height, centre[2] - size)
            self.ground_source.SetPoint2(centre[0] - size, height, centre[2] + size)
            self.ground_actor.SetVisibility(self.show_ground.isChecked())
            self.ground_extent = (float(centre[0]), float(centre[2]), size)
        if render:
            self.viewport.GetRenderWindow().Render()

    def update_collider_visibility(self, render=True):
        if not hasattr(self, "collider_actors"):
            return
        moving = [a for a, role in zip(self.collider_actors, self.collider_roles) if role == "moving"]
        fixed = [a for a, role in zip(self.collider_actors, self.collider_roles) if role != "moving"]
        self.view.apply([self.garment_actor], moving)
        for actor in fixed:
            actor.SetVisibility(self.show_fixed.isChecked())
            actor.GetProperty().SetOpacity(self.view.avatar_opacity.value() / 100)
        if render:
            self.viewport.GetRenderWindow().Render()

    def update_wind(self, settings):
        self.wind_control.set(settings)
        self.wind_preview.update(settings)
        self.renderer.ResetCameraClippingRange()
        self.viewport.GetRenderWindow().Render()

    def update_water(self, settings):
        self.water_control.set(settings)
        self.water_preview.update(settings)
        self.renderer.ResetCameraClippingRange()
        self.viewport.GetRenderWindow().Render()

    def update_live_controls(self):
        idle = self.worker is None and self.return_worker is None
        paused = self.live is not None and self.live.paused and self.live.paused_ack
        if hasattr(self, 'use_proxy'):
            # Switching while a live run is paused continues from the current shape (no reset).
            self.use_proxy.setEnabled(self.return_worker is None and not self.close_pending
                                      and (idle or (paused and self.continue_from is None)))
            self.proxy_size.setEnabled(idle and self.use_proxy.isChecked())
        # Gravity, wind and water are read live by the solver: adjustable while it runs (as in Hair).
        for name in ('wind_panel', 'gravity_panel', 'water_panel'):
            if hasattr(self, name):
                getattr(self, name).setEnabled(self.return_worker is None and not self.close_pending)
        static = self.output_mode.currentData() == "static"
        self.interactive_enabled.setEnabled(static and self.worker is None and self.return_worker is None)
        active = self.live is not None and not self.live.finished and self.worker is not None and self.live_ready and self.return_worker is None
        for widget in (self.pause_simulation, self.grab_radius, self.grab_strength):
            widget.setEnabled(active)
        self.hold_frame.setEnabled(active and self.motion is not None and not self.live_at_end and self.drag is None)
        self.hold_frame.setText("Final frame held" if self.live_at_end else
                               "Continue timeline" if self.hold_frame.isChecked() else "Hold frame")
        self.grab_edit.setEnabled(active)
        if hasattr(self, 'paint_edit'):
            paintable = static and self.return_worker is None and (idle or (active and paused))
            self.keep_freeze_timeline.setEnabled(paintable and self.motion is not None)
            for widget in (self.paint_edit, self.paint_mode, self.paint_radius, self.grow_selection,
                           self.shrink_selection, self.clear_selection):
                widget.setEnabled(paintable)
            for widget in (self.freeze_selected, self.freeze_unselected, self.unfreeze_selected, self.unfreeze_all):
                widget.setEnabled(active and self.live.paused_ack)
        if self.live is not None and self.worker is not None:
            self.actions.show_run_state(True, self.live.paused)
            self.run_button.setEnabled(active)
            self.stop_button.setText("Pause simulation")
        if hasattr(self, 'reset_button'):
            self.reset_button.setEnabled((idle or paused) and self.return_worker is None)
        if getattr(self, 'preparing', False) and self.worker is not None and not self.worker.isInterruptionRequested():
            self.run_button.setText("Cancel preparation")
            self.run_button.setEnabled(True)

    def set_frame_hold(self, holding):
        self.hold_frame.setText("Continue timeline" if holding else "Hold frame")
        if self.live is not None:
            if not holding:
                self.release_freeze_for_timeline()
            self.live.set_hold(holding)

    def release_freeze_for_timeline(self):
        if (self.live is not None and self.motion is not None and not self.live_at_end
                and not self.keep_freeze_timeline.isChecked() and np.any(self.frozen)):
            self.frozen[:] = False
            self.live.set_frozen([], np.empty((0, 3), dtype=np.float32))
            self.refresh_selection()
            self.viewport.GetRenderWindow().Render()

    def set_simulation_pause(self, paused):
        self.pause_simulation.setText("Resume simulation" if paused else "Pause simulation")
        if self.live is not None:
            if not paused and not self.live.holding:
                self.release_freeze_for_timeline()
            if not paused and hasattr(self, 'paint_edit'):
                self.paint_edit.setChecked(False)
            self.release_grab()
            self.live.set_pause(paused)
            self.result_ready = False
            if paused:
                self.status.set_state("paused")
                self.progress.setText("Resume to settle or tug the fabric")
            else:
                self.status.set_state("running")
        self.update_live_controls()
        if hasattr(self, "send_button"):
            self.update_send_button()

    @Slot()
    def live_paused(self):
        if self.live is None or not self.live.paused or not self.live.paused_ack:
            return
        self.live_at_end = self.live_at_end or self.live.at_end
        self.end_preparation()
        self.pause_simulation.blockSignals(True)
        self.pause_simulation.setChecked(True)
        self.pause_simulation.setText("Resume simulation")
        self.pause_simulation.blockSignals(False)
        self.result_ready = not self.simulation_failed and self.live_ready
        self.release_grab()
        self.record_speed()
        self.status.set_state("paused")
        self.status.set_progress(None)
        self.progress.setText("Final settling complete — resume, tug, or send to DAZ" if self.live_at_end else
                              "Paused at current frame — resume, tug, or send to DAZ")
        self.update_live_controls()
        self.update_send_button()

    def reset_view(self):
        self.release_grab()
        self.camera_drag = None
        meshes = [self.mesh] + [mesh for mesh, actor in zip(self.collider_meshes, self.collider_actors)
                                if actor.GetVisibility()]
        bounds = np.asarray([mesh.GetBounds() for mesh in meshes])
        fitted = tuple(value for axis in range(3)
                       for value in (bounds[:, axis * 2].min(), bounds[:, axis * 2 + 1].max()))
        self.renderer.GetActiveCamera().DeepCopy(self.start_camera)
        self.renderer.ResetCamera(fitted)
        self.renderer.ResetCameraClippingRange(fitted)
        self.viewport.GetRenderWindow().Render()

    def selection_allowed(self):
        allowed = np.ones(len(self.vertices), dtype=bool)
        if self.keep_details.isChecked():
            allowed &= ~self.structure.attached
        return allowed

    def refresh_selection(self):
        for points, mask in zip(self.paint_overlays, (self.selection & ~self.frozen, self.frozen & ~self.selection,
                                                     self.selection & self.frozen)):
            points.SetData(numpy_to_vtk(self.vertices[mask], deep=True))
        self.selection_count.setText(f"{np.count_nonzero(self.selection)} selected; {np.count_nonzero(self.frozen)} frozen")

    def set_paint_mode(self, enabled):
        self.paint_drag = False
        if enabled:
            self.release_grab()
            self.camera_drag = None
            self.grab_edit.setChecked(False)
            self.pin_edit.setChecked(False)
            self.fabric_edit.setChecked(False)
            if self.live is not None and self.motion is not None and not self.live_at_end:
                self.hold_frame.setChecked(True)
            if self.live is not None and not self.live.paused:
                self.pause_simulation.setChecked(True)

    def paint_at_pointer(self):
        if self.worker is not None and (self.live is None or not self.live.paused_ack or not self.live.paused):
            return
        x, y = self.viewport.GetEventPosition()
        if self.picker.Pick(x, y, 0, self.renderer):
            location = np.asarray(self.picker.GetPickPosition())
            face = self.garment[1][self.picker.GetCellId()]
            allowed = self.selection_allowed()
            seed = int(face[np.argmin(np.linalg.norm(self.vertices[face] - location, axis=1))])
            if not allowed[seed]:
                return
            selected = self.selection_surface.brush(self.vertices, seed, self.paint_radius.value() * .1, allowed)
            self.selection[selected] = self.paint_mode.currentIndex() == 0
            self.refresh_selection()
            self.viewport.GetRenderWindow().Render()

    def resize_selection(self, grow):
        if not self.grow_selection.isEnabled():
            return
        self.selection = self.selection_surface.resize(self.selection, grow, self.selection_allowed())
        self.refresh_selection()
        self.viewport.GetRenderWindow().Render()

    def clear_painted_selection(self):
        self.selection[:] = False
        self.refresh_selection()
        self.viewport.GetRenderWindow().Render()

    def apply_freeze(self, action):
        if self.live is None or not self.live.paused_ack or not self.live.paused or self.return_worker is not None:
            return
        self.release_grab()
        if action in ('selected', 'unselected'):
            chosen = self.selection if action == 'selected' else ~self.selection
            chosen = chosen & self.selection_allowed() & ~self.pinned
            new = chosen & ~self.frozen
            self.freeze_targets[new] = self.vertices[new]
            self.frozen |= chosen
        elif action == 'remove':
            self.frozen[self.selection] = False
        else:
            self.frozen[:] = False
        self.live.set_frozen(np.flatnonzero(self.frozen), self.freeze_targets[self.frozen] * .01)
        self.refresh_selection()
        self.viewport.GetRenderWindow().Render()

    def setup_shortcuts(self):
        self.camera_shortcuts = {
            'orbit_up': ('orbit', 0, 20), 'orbit_down': ('orbit', 0, -20),
            'orbit_left': ('orbit', 20, 0), 'orbit_right': ('orbit', -20, 0),
            'pan_up': ('pan', 0, 1), 'pan_down': ('pan', 0, -1),
            'pan_left': ('pan', -1, 0), 'pan_right': ('pan', 1, 0),
            'zoom_in': ('zoom', 1.2, 0), 'zoom_out': ('zoom', 1/1.2, 0),
        }
        actions = {
            'pause_resume': self.pause_simulation, 'hold_frame': self.hold_frame,
            'grab_fabric': self.grab_edit, 'paint_selection': self.paint_edit,
            'grow_selection': self.grow_selection, 'shrink_selection': self.shrink_selection,
            'freeze_selected': self.freeze_selected, 'freeze_unselected': self.freeze_unselected,
            'unfreeze_selected': self.unfreeze_selected, 'unfreeze_all': self.unfreeze_all,
            'clear_selection': self.clear_selection, 'reset_view': self.reset_view_button,
            'send_to_daz': self.send_button,
            'show_props': self.show_fixed, 'show_avatar': self.show_moving,
            'edit_pins': self.pin_edit, 'edit_fasteners': self.fabric_edit,
            'pin_top': self.pin_top, 'clear_pins': self.pin_clear,
            'clear_fasteners': self.clear_fasteners, 'reset_simulation': self.reset_button,
            'run_simulation': self.run_button,
            'toggle_wind': self.wind_panel.enabled_box,
            'toggle_water': self.water_panel.enabled_box,
        }
        # One suite-wide file (Spin-Tools/keybindings.json): "common" plus this tool's "drape" section,
        # bound through the shared spin_ui.Shortcuts (F1 lists them).
        import keybindings
        self.shortcut_buttons = actions
        handlers = {action: (lambda action=action: self.invoke_shortcut(action, actions))
                    for action in list(actions) + list(self.camera_shortcuts) +
                    ['paint_add', 'paint_remove', 'cancel_interaction']}
        self.keys = spin_ui.Shortcuts(self, 'drape', handlers)
        errors = self.keys.errors
        self.shortcuts = self.keys.shortcuts
        loaded = [f'{action}: {key}' for action, keys in self.keys.bindings.items() for key in keys]
        self.shortcut_note.setText('Keyboard shortcuts are loaded from keybindings.json in the Spin Tools folder (shared by all tools). Press F1 for the list.'
                                   if not errors else 'Some shortcuts could not load. Check keybindings.json.')
        self.shortcut_note.setToolTip(str(keybindings.PATH) + '\n' + '\n'.join(loaded + errors))
        self.view.reset_view_button.parentWidget().layout().addWidget(self.keys.help_button())
        self.view_bar = spin_ui.add_view_toolbar(self.keys, self.status)   # orbit/pan/zoom/reset buttons

    def invoke_shortcut(self, action, actions):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit, QComboBox)):
            return
        if action in self.camera_shortcuts:
            self.keyboard_view(*self.camera_shortcuts[action])
        elif action == 'cancel_interaction' and self.preparing:
            self.cancel_preparation()
        elif action == 'cancel_interaction':
            self.release_grab()
            self.paint_edit.setChecked(False)
            self.grab_edit.setChecked(False)
            self.pin_edit.setChecked(False)
            self.fabric_edit.setChecked(False)
            self.cut_edit.setChecked(False)
            self.pending_fastener = None
            self.refresh_fasteners()
            self.camera_drag = None
        elif action in ('paint_add', 'paint_remove'):
            if self.paint_edit.isEnabled():
                self.paint_mode.setCurrentIndex(0 if action == 'paint_add' else 1)
                self.paint_edit.setChecked(True)
        elif actions[action].isEnabled():
            actions[action].click()

    def keyboard_view(self, mode, horizontal, vertical):
        self.release_grab()
        self.paint_drag = False
        self.camera_drag = None
        self.camera.move(mode, horizontal, vertical)          # shared suite camera

    def zoom_at_pointer(self, factor):
        self.release_grab()
        vtk.vtkInteractorStyleTrackballCamera.DollyToPosition(
            factor, self.viewport.GetEventPosition(), self.renderer)
        self.renderer.ResetCameraClippingRange()
        self.viewport.GetRenderWindow().Render()

    @staticmethod
    def number(minimum, maximum, value, step, suffix=""):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(2)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def apply_fabric_preset(self, name):
        if name not in FABRICS:
            return
        self.applying_fabric = True
        preset = FABRICS[name]
        for key, widget in (('stretch', self.stretch), ('bending', self.bending),
                            ('weight', self.fabric_weight), ('damping', self.damping), ('friction', self.friction)):
            widget.setValue(preset[key])
        self.applying_fabric = False

    def mark_custom_fabric(self):
        if not self.applying_fabric:
            self.fabric_preset.setCurrentIndex(0)

    @Slot()
    def update_output_note(self):
        if self.output_mode.currentData() == "animation":
            self.output_note.setText(
                "Return the settled garment at each DAZ frame as FBX animation. Original garments will be hidden."
                if self.motion is not None else "Enable timeline movement in DAZ to return an animation."
            )
        else:
            self.output_note.setText(
                "Send the completed static result to DAZ as a new prop. The original garment will be hidden."
                if self.return_spec else "Open the helper using Drape Tool.dsa to enable static return to DAZ."
            )
        if hasattr(self, "send_button"):
            self.update_send_button()
        if hasattr(self, "pause_simulation"):
            self.update_live_controls()

    def update_send_button(self):
        animated = self.output_mode.currentData() == "animation"
        self.send_button.setText("Send animation to DAZ" if animated else "Send static result to DAZ")
        self.send_button.setEnabled(
            self.return_spec is not None and self.result_ready and self.return_worker is None
            and (self.worker is None or (self.live is not None and self.live.paused_ack))
            and (not animated or self.animation is not None)
        )

    @Slot()
    def send_static_result(self):
        if not self.send_button.isEnabled():
            return
        animation = self.animation if self.output_mode.currentData() == "animation" else None
        spec = self.return_spec.copy()
        if animation is not None:
            spec["animation_format"] = self.return_format.currentData()
        if (self.water_panel.enabled_box.isChecked() and self.water_panel.export_plane.isChecked()
                and self.water_panel.water_source is None):
            spec['water_plane'] = dict(height=self.water_panel.fields['height'].value(),
                                      x=float(self.water_preview.center[0]), z=float(self.water_preview.center[2]),
                                      extent=self.water_preview.extent)
        height = self.ground_height()
        if self.ground_send.isEnabled() and self.ground_send.isChecked() and height is not None:
            x, z, extent = self.ground_extent
            spec['ground_plane'] = dict(height=float(height), x=x, z=z, extent=extent)
        self.return_worker = ReturnWorker(spec, self.vertices.copy(), self.garment_ranges, animation, self,
                                          self.live_session)
        self.update_live_controls()
        for widget in self.edit_controls + [self.run_button, self.reset_button, self.output_mode, self.send_button]:
            widget.setEnabled(False)
        self.status.set_state("sending")
        self.progress.setText("Sending to DAZ…")
        self.return_worker.finished.connect(self.finish_return)
        self.return_worker.finished.connect(self.return_worker.deleteLater)
        self.return_worker.start()

    @Slot()
    def finish_return(self):
        error = self.return_worker.error
        self.return_worker = None
        if error is None and (self.close_pending or self.close_after.isChecked()):
            self.close()
            return
        if self.close_pending:
            self.close_pending = False
            self.show()
        if error is not None:
            QMessageBox.critical(self, "Return to DAZ", error)
        for widget in self.edit_controls + [self.output_mode]:
            widget.setEnabled(self.worker is None)
        self.run_button.setEnabled(self.worker is None)
        self.reset_button.setEnabled(True)
        self.update_live_controls()
        self.status.set_state("failed" if error is not None else "sent")
        self.progress.setText("Send failed. The completed drape is available to retry." if error is not None else
                              "The Spin Tools receive window imports it; adjust and send again to replace it.")
        self.update_send_button()
        if error is None:
            # Control passes back to DAZ; restore from the taskbar to refine and resend.
            self.showMinimized()

    def refresh_pins(self):
        self.pin_points.SetData(numpy_to_vtk(self.vertices[self.pinned], deep=True))
        self.pin_count.setText(f"{np.count_nonzero(self.pinned)} pinned vertices")
        self.refresh_fasteners()

    def refresh_fasteners(self):
        pairs = list(self.fasteners)
        if self.fasten_buttons.isChecked():
            pairs = list(self.structure.fasteners) + pairs
        indices = np.asarray(pairs, dtype=int).reshape(-1)
        if self.pending_fastener is not None:
            indices = np.append(indices, self.pending_fastener)
        self.fastener_points.SetData(numpy_to_vtk(self.vertices[indices], deep=True))
        lines = vtk.vtkCellArray()
        for pair in range(len(pairs)):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(pair * 2)
            lines.InsertCellPoint(pair * 2 + 1)
        self.fastener_mesh.SetLines(lines)
        markers = vtk.vtkCellArray()
        for index in range(len(indices)):
            markers.InsertNextCell(1)
            markers.InsertCellPoint(index)
        self.fastener_mesh.SetVerts(markers)
        self.fastener_count.setText(
            "First point selected — click the other side" if self.pending_fastener is not None else
            f"{len(self.fasteners)} custom fabric fastenings"
        )

    def refresh_bar(self):
        self.bar_actor.SetVisibility(self.bar_probe is not None)
        if self.bar_probe is not None:
            vertex, normal = self.bar_probe
            center = self.vertices[vertex]
            delta = normal * self.bar_reach.value() * .1
            self.bar_source.SetPoint1(*(center - delta))
            self.bar_source.SetPoint2(*(center + delta))
        self.viewport.GetRenderWindow().Render()

    @Slot()
    def clear_fabric_fasteners(self):
        self.fasteners.clear()
        self.bar_probe = None
        self.refresh_bar()
        self.pending_fastener = None
        self.refresh_fasteners()
        self.viewport.GetRenderWindow().Render()

    def render_vertices(self, vertices):
        self.vertices = vertices
        self.mesh.GetPoints().SetData(numpy_to_vtk(vertices, deep=True))
        self.refresh_pins()
        self.refresh_selection()
        if self.drag is not None:
            self.grab_points.SetData(numpy_to_vtk(self.vertices[self.drag["indices"]], deep=True))
        self.viewport.GetRenderWindow().Render()

    def release_grab(self):
        self.drag = None
        if self.live is not None:
            self.live.set_grab()
        if hasattr(self, "grab_points"):
            self.grab_points.SetNumberOfPoints(0)
            self.grab_points.Modified()

    def eventFilter(self, watched, event):
        if watched is getattr(self, "viewport", None) and event.type() in (QEvent.Type.Leave, QEvent.Type.FocusOut):
            self.release_grab()
            self.camera_drag = None
            self.paint_drag = False
        return super().eventFilter(watched, event)

    def begin_camera_drag(self, mode):
        self.release_grab()
        if mode == 'orbit':
            self.renderer.GetActiveCamera().SetFocalPoint(*self.orbit_pivot())
        self.camera_drag = dict(mode=mode, position=self.viewport.GetEventPosition())

    def orbit_pivot(self):
        """The surface at the screen center, or, when that misses, the middle of the visible
        garment and avatar (not a point in empty space)."""
        actors = [actor for actor in [self.garment_actor] + self.collider_actors if actor.GetVisibility()]
        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.SetTolerance(.002)
        for actor in actors:
            picker.AddPickList(actor)
        origin, size = self.renderer.GetOrigin(), self.renderer.GetSize()
        if picker.Pick(origin[0] + size[0] * .5, origin[1] + size[1] * .5, 0, self.renderer):
            return picker.GetPickPosition()
        if not actors:
            return self.renderer.GetActiveCamera().GetFocalPoint()
        bounds = np.array([actor.GetBounds() for actor in actors])
        return tuple((bounds[:, axis * 2].min() + bounds[:, axis * 2 + 1].max()) / 2 for axis in range(3))

    def end_camera_drag(self, caller=None, event=None):
        self.camera_drag = None

    def begin_grab(self):
        """Start tugging the fabric under the pointer; False when there is nothing to grab there."""
        x, y = self.viewport.GetEventPosition()
        if not self.picker.Pick(x, y, 0, self.renderer):
            return False
        location = np.asarray(self.picker.GetPickPosition())
        face = self.garment[1][self.picker.GetCellId()]
        seed = int(face[np.argmin(np.linalg.norm(self.vertices[face] - location, axis=1))])
        if self.keep_details.isChecked() and self.structure.attached[seed]:
            body = np.unique(self.structure.body_faces)
            seed = int(body[np.argmin(np.linalg.norm(self.vertices[body] - location, axis=1))])
        allowed = ~self.pinned
        allowed &= ~self.frozen
        if self.keep_details.isChecked():
            allowed = allowed & ~self.structure.attached
        indices, weights = grab_patch(self.vertices, self.garment[1], seed,
                                      self.grab_radius.value() * .1, allowed)
        if not len(indices):
            self.progress.setText("That fabric is pinned. Grab an unpinned area.")
            return False
        if self.motion is not None and not self.live_at_end:
            self.hold_frame.setChecked(True)
        self.renderer.SetWorldPoint(*location, 1.)
        self.renderer.WorldToDisplay()
        self.drag = dict(indices=indices, weights=weights, origin=location,
                         positions=self.vertices[indices].copy(), depth=self.renderer.GetDisplayPoint()[2])
        self.move_mouse_drag(None, None)
        return True

    def move_mouse_drag(self, caller, event):
        if self.paint_drag:
            self.paint_at_pointer()
            return
        if self.camera_drag is not None:
            position = self.viewport.GetEventPosition()
            previous = self.camera_drag['position']
            turn_camera(self.renderer.GetActiveCamera(), self.camera_drag['mode'],
                        position[0] - previous[0], position[1] - previous[1])
            self.camera_drag['position'] = position
            self.renderer.ResetCameraClippingRange()
            self.viewport.GetRenderWindow().Render()
            return
        if self.drag is None:
            self.style.OnMouseMove()
            return
        x, y = self.viewport.GetEventPosition()
        self.renderer.SetDisplayPoint(x, y, self.drag["depth"])
        self.renderer.DisplayToWorld()
        world = np.array(self.renderer.GetWorldPoint())
        if abs(world[3]) < 1e-12:
            return
        delta = world[:3] / world[3] - self.drag["origin"]
        targets = (self.drag["positions"] + delta) * .01
        if self.live is not None and np.isfinite(targets).all():
            self.live.set_grab(self.drag["indices"], targets, self.drag["weights"], self.grab_strength.value() / 100)

    def end_mouse_drag(self, caller, event):
        if self.paint_drag:
            self.paint_drag = False
            return
        if self.camera_drag is not None and self.camera_drag['mode'] == 'orbit':
            self.end_camera_drag()
            return
        if self.drag is not None:
            self.release_grab()
            self.viewport.GetRenderWindow().Render()
        else:
            self.style.OnLeftButtonUp()

    def template_paths(self):
        paths = self.window_inputs.get("garment_paths")
        if paths:
            return list(paths)
        if self.return_spec is not None:
            return [entry["path"] for entry in self.return_spec["garments"]]
        return None

    def update_cut_note(self):
        if self.cut_picks:
            text = f"{len(self.cut_picks)} cut point(s); Apply cut splits the mesh along them."
        elif self.cut_pairs:
            text = f"Cut: {len(self.cut_pairs)} vertices split."
        else:
            text = "Reset first. Cutting reopens the window with the cut mesh; paint and pins start again."
        self.cut_note.setText(text)
        editable = self.worker is None and self.return_worker is None
        self.cut_apply.setEnabled(editable and len(self.cut_picks) >= 2 and self.template_paths() is not None)
        self.cut_clear.setEnabled(bool(self.cut_picks))
        self.cut_undo.setEnabled(editable and self.window_inputs.get("uncut") is not None)
        self.refresh_cut_line()

    def refresh_cut_line(self):
        if not hasattr(self, "renderer"):
            return
        if not hasattr(self, "cut_mesh"):
            self.cut_mesh = vtk.vtkPolyData()
            self.cut_points_vtk = vtk.vtkPoints()
            self.cut_mesh.SetPoints(self.cut_points_vtk)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.cut_mesh)
            self.cut_actor = vtk.vtkActor()
            self.cut_actor.SetMapper(mapper)
            self.cut_actor.GetProperty().SetColor(1., .35, .2)
            self.cut_actor.GetProperty().SetLineWidth(3)
            self.cut_actor.GetProperty().SetPointSize(8)
            self.cut_actor.GetProperty().SetRenderPointsAsSpheres(True)
            self.renderer.AddActor(self.cut_actor)
        picks = np.asarray(self.cut_picks, dtype=int)
        self.cut_points_vtk.SetData(numpy_to_vtk(self.vertices[picks].reshape(-1, 3).astype(np.float64), deep=True))
        lines, markers = vtk.vtkCellArray(), vtk.vtkCellArray()
        if len(picks) > 1:
            lines.InsertNextCell(len(picks))
            for index in range(len(picks)):
                lines.InsertCellPoint(index)
        for index in range(len(picks)):
            markers.InsertNextCell(1)
            markers.InsertCellPoint(index)
        self.cut_mesh.SetLines(lines)
        self.cut_mesh.SetVerts(markers)
        self.cut_mesh.Modified()
        self.viewport.GetRenderWindow().Render()

    def clear_cut_points(self):
        self.cut_picks.clear()
        self.update_cut_note()

    def apply_cut(self):
        from garment_cut import cut_window_inputs
        release = self.cut_release_frame.value() if self.motion is not None else None
        if release is not None and release <= self.motion.start_frame:
            release = None
        paths = self.template_paths()
        try:
            inputs, count = cut_window_inputs(self.window_inputs, paths, self.cut_picks, Path(paths[0]).parent, release)
        except (OSError, ValueError) as error:
            self.cut_note.setText(str(error))
            return
        self.reopen(inputs, f"Cut applied: {count} vertices split.")

    def remove_cuts(self):
        original = self.window_inputs.get("uncut")
        if original is not None:
            self.reopen(dict(original), "Cuts removed; the garment is back as captured.")

    def reopen(self, inputs, message):
        window = DrapeWindow(**inputs)
        DrapeWindow.open_windows.append(window)
        window.show()
        window.viewport.Initialize()
        window.cut_note.setText(message)
        self.close()

    def pick_pin(self, caller, event):
        if self.cut_edit.isChecked() and self.worker is None and self.return_worker is None:
            x, y = self.viewport.GetEventPosition()
            if self.picker.Pick(x, y, 0, self.renderer):
                location = np.asarray(self.picker.GetPickPosition())
                cell = self.mesh.GetCell(self.picker.GetCellId())
                candidates = np.array([cell.GetPointId(index) for index in range(cell.GetNumberOfPoints())])
                nearest = int(candidates[np.argmin(np.linalg.norm(self.vertices[candidates] - location, axis=1))])
                if not self.cut_picks or self.cut_picks[-1] != nearest:
                    self.cut_picks.append(nearest)
                self.update_cut_note()
            return
        if self.paint_edit.isChecked() and self.paint_edit.isEnabled():
            self.paint_drag = True
            self.paint_at_pointer()
            return
        # Grab only while the simulation runs and the drag starts on unpinned fabric; otherwise
        # (paused, not started, empty space, avatar, pinned fabric) left-drag moves the view.
        if (self.live is not None and self.live_ready and self.grab_edit.isChecked() and self.return_worker is None
                and not self.live.paused and self.begin_grab()):
            return
        editing = self.pin_edit.isChecked() or self.viewport.GetShiftKey()
        fastening = self.fabric_edit.isChecked() and not self.viewport.GetShiftKey()
        if self.worker is not None or self.return_worker is not None or not (editing or fastening):
            self.begin_camera_drag('orbit')
            return
        x, y = self.viewport.GetEventPosition()
        if self.picker.Pick(x, y, 0, self.renderer):
            location = np.asarray(self.picker.GetPickPosition())
            cell = self.mesh.GetCell(self.picker.GetCellId())
            candidates = np.array([cell.GetPointId(index) for index in range(cell.GetNumberOfPoints())])
            nearest = candidates[np.argmin(np.linalg.norm(self.vertices[candidates] - location, axis=1))]
            if fastening and self.fastener_pick_mode.currentData() == "bar":
                triangle = self.vertices[self.garment[1][self.picker.GetCellId()]]
                normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                normal /= max(float(np.linalg.norm(normal)), 1e-10)
                self.bar_probe = (int(nearest), normal)
                self.refresh_bar()
                pair = layer_pair(self.vertices, self.garment[1], self.picker.GetCellId(), location,
                                  self.bar_reach.value() * .1,
                                  self.structure.attached if self.keep_details.isChecked() else None)
                self.pending_fastener = None
                if pair is not None and not (self.pinned[pair[0]] and self.pinned[pair[1]]):
                    if pair in self.fasteners:
                        self.fasteners.remove(pair)
                    else:
                        self.fasteners.append(pair)
                    self.refresh_fasteners()
                else:
                    self.refresh_fasteners()
                    self.fastener_count.setText("No free fabric layer within bar reach. Try another spot or increase reach.")
                self.viewport.GetRenderWindow().Render()
                return
            if fastening:
                if self.keep_details.isChecked() and self.structure.attached[nearest]:
                    body = np.unique(self.structure.body_faces)
                    nearest = body[np.argmin(np.linalg.norm(self.vertices[body] - location, axis=1))]
                if self.pending_fastener is None:
                    self.pending_fastener = int(nearest)
                elif self.pending_fastener == nearest:
                    self.pending_fastener = None
                else:
                    pair = tuple(sorted((self.pending_fastener, int(nearest))))
                    if pair in self.fasteners:
                        self.fasteners.remove(pair)
                    elif self.pinned[pair[0]] and self.pinned[pair[1]]:
                        self.pending_fastener = None
                        self.refresh_fasteners()
                        self.fastener_count.setText("Both points are avatar-pinned. Unpin the fabric before fastening it.")
                        self.viewport.GetRenderWindow().Render()
                        return
                    elif np.any(np.sum(np.isin(self.garment[1], pair), axis=1) == 2):
                        self.pending_fastener = None
                        self.refresh_fasteners()
                        self.fastener_count.setText("Those points share a face. Pick fabric on opposite sides of the opening.")
                        self.viewport.GetRenderWindow().Render()
                        return
                    else:
                        self.fasteners.append(pair)
                    self.pending_fastener = None
                self.refresh_fasteners()
                self.viewport.GetRenderWindow().Render()
                return
            # Connected fabric only, so a nearby lining or second garment is not pinned.
            selected = self.selection_surface.brush(self.vertices, int(nearest), self.brush.value() * 0.1,
                                                    np.ones(len(self.vertices), dtype=bool))
            selected[nearest] = True
            self.pinned[selected] = not np.all(self.pinned[selected])
            self.refresh_pins()
            self.viewport.GetRenderWindow().Render()

    @Slot()
    def select_top(self):
        self.pinned = self.top_pins()
        self.refresh_pins()
        self.viewport.GetRenderWindow().Render()

    def top_pins(self):
        pinned = np.zeros(len(self.start_vertices), dtype=bool)
        for start, end in self.garment_ranges:
            heights = self.start_vertices[start:end, 1]
            pinned[start:end] = heights >= heights.max() - 3.0
        return pinned

    @Slot()
    def clear_pins(self):
        self.pinned[:] = False
        self.refresh_pins()
        self.viewport.GetRenderWindow().Render()

    @Slot()
    def reset_dress(self):
        if self.return_worker is not None:
            return
        if self.worker is not None:
            self.reset_pending = True
            self.worker.requestInterruption()
            self.release_grab()
            return
        self.result_ready = False
        self.clear_animation()
        self.update_send_button()
        if self.motion is not None:
            self.motion.reset()
            self.render_colliders(self.motion.preview(0))
        self.frozen[:] = False
        self.render_vertices(self.start_vertices.copy())
        self.frame_number = 0
        self.status.set_state("idle")
        self.status.set_frame()
        self.progress.setText("Reset to the captured position. Pins and material settings retained.")
        if self.proxy_advice.text():
            self.show_size_advice()                      # heavy garment: the advice again (panel note too)

    @Slot()
    def start_drape(self, *_, start=None):          # *_: the button's clicked(checked) argument
        """Run from the captured shape, or, with ``start`` (cm), continue from that shape."""
        if self.preparing:
            self.cancel_preparation()
            return
        if self.worker is not None and self.live is not None and self.return_worker is None:
            if self.pause_simulation.isEnabled():
                self.pause_simulation.toggle()
            return
        if self.worker is not None or self.return_worker is not None:
            return
        self.bar_probe = None
        self.bar_actor.SetVisibility(False)
        self.release_grab()
        self.live = None
        self.live_ready = False
        self.live_at_end = False
        self.frozen[:] = False
        self.paint_drag = False
        self.hold_frame.setChecked(False)
        self.pause_simulation.setChecked(False)
        if self.output_mode.currentData() == "static" and self.interactive_enabled.isChecked():
            self.live = proxy_live(LiveDrape) if self.use_proxy.isChecked() else LiveDrape()
        self.frame_number = 0
        self.simulation_failed = False
        self.result_ready = False
        self.clear_animation()
        self.update_send_button()
        self.reset_pending = False
        initial_duration = self.initial_duration.value() if self.motion is not None else 0
        frame_settle = self.frame_settle.value() if self.motion is not None else 0
        max_move = self.max_move.value() if self.motion is not None else 0
        # Measuring the movement reads every timeline frame (~13 s for a 66k-vertex avatar): the worker
        # does it; here only the cached result is used for the progress estimate.
        known = max_move if max_move in getattr(self.motion, "_spin_parts", {}) else 0
        schedule = DrapeSchedule(self.motion, self.duration.value(), initial_duration, frame_settle, known)
        self.active_duration = schedule.duration
        if self.motion is not None:
            self.motion.reset()
            self.render_colliders(self.motion.preview(0))
        self.render_vertices(self.start_vertices.copy() if start is None else start.copy())
        garment = self.garment if start is None else (np.asarray(start, np.float32) * .01, self.garment[1])
        settings = dict(
            water=self.water_control,
            gravity=self.gravity_control,
            wind=self.wind_control,
            duration=self.duration.value(),
            pinned_vertices=np.flatnonzero(self.pinned).tolist(),
            stretch=self.stretch.value() / 100,
            bending=self.bending.value() / 100,
            fabric_weight_gsm=self.fabric_weight.value(),
            edge_width_mm=self.edge_width.value(),
            edge_stiffness=self.edge_stiffness.value() / 100,
            crease_angle=self.crease_angle.value(),
            crease_stiffness=self.crease_stiffness.value() / 100 if self.crease_enabled.isChecked() else 0.,
            thickness_mm=self.thickness.value(),
            self_collision=self.self_collision.isChecked(),
            self_collision_gap_mm=self.self_collision_gap.value(),
            self_collision_passes=self.self_collision_passes.value(),
            friction=self.friction.value(),
            damping=self.damping.value(),
            iterations=self.solver_passes(),
            motion=self.motion,
            initial_duration=initial_duration,
            frame_settle=frame_settle,
            max_move_cm=max_move,
            rigid_motion=self.rigid_motion.isChecked(),
            loop=self.motion is not None and self.loop_animation.isChecked(),
            loop_frames=self.loop_frames.value(),
            ground=self.ground_settings(),
            garment_ranges=self.garment_ranges,
            button_vertices=self.button_vertices,
            keep_details=self.keep_details.isChecked(),
            fasten_buttons=self.fasten_buttons.isChecked(),
            fasteners=list(self.fasteners),
            cut_pairs=list(self.cut_pairs),
            cut_release_interval=(None if self.cut_release is None or self.motion is None
                                  else self.cut_release - self.motion.start_frame),
            fastener_gap_mm=self.fastening_gap.value() if self.fastening_mode.currentData() == "close" else None,
        )
        if self.live is not None:
            settings["interactive"] = self.live
        if self.use_proxy.isChecked():
            settings["proxy_size"] = self.proxy_size.value()
            settings["proxy"] = self.proxy
        for widget in self.edit_controls:
            widget.setEnabled(False)
        self.run_button.setEnabled(False)
        for widget in (self.cut_apply, self.cut_undo):
            widget.setEnabled(False)
        self.output_mode.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_button.setVisible(self.live is None)
        self.status.set_state("running")
        self.status.set_frame()
        self.progress.setText(f"Preparing GPU… ({settings['iterations']} solver passes)")
        if start is not None:
            settings["proxy"] = None                         # the proxy is rebuilt around the new shape
        self.worker = DrapeWorker(garment, self.colliders, settings, self)
        self.run_started, self.first_frame_time, self.first_frame_count = time.monotonic(), None, 0
        self.preparing = True
        self.status.activity.setStyleSheet("")
        self.worker.preparing.connect(self.on_preparing, Qt.ConnectionType.QueuedConnection)
        self.update_live_controls()
        # Queued, not blocking: the solver keeps going while a frame is drawn (frames stay in order,
        # so the paused/final frame is shown before the pause or finish is handled).
        self.worker.frame.connect(self.on_frame, Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self.report_error, Qt.ConnectionType.QueuedConnection)
        self.worker.paused.connect(self.live_paused, Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.finished.connect(self.finish, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    @Slot(object)
    def on_frame(self, vertices):
        try:
            self.show_frame(vertices)
        finally:
            if self.worker is not None:
                self.worker.preview_busy = False

    def show_frame(self, vertices):
        self.end_preparation()
        count = getattr(self.worker, 'frame_number', self.frame_number + 1)
        elapsed = count / 60
        timeline = ""
        source_frame = None
        phase = "Draping"
        if isinstance(vertices, dict):
            elapsed = vertices["time"]
            if vertices['timeline_frame'] is not None:
                source_frame = vertices['timeline_frame']
                if self.motion is not None:
                    self.status.set_frame(source_frame, self.motion.start_frame + self.motion.intervals)
            phase = vertices.get("phase", "Draping")
            self.render_colliders(vertices["colliders"])
            moving = vertices["colliders"]
            vertices = vertices["vertices"]
        else:
            moving = None
        if self.worker is not None:
            vertices = self.worker.full(vertices, moving)    # proxy frames: carried to the full mesh here
        self.render_vertices(vertices)
        self.frame_number = count
        if self.live is not None:
            self.live_ready = True
            self.live_at_end = self.live.at_end or (self.motion is not None and
                               source_frame == self.motion.start_frame + self.motion.intervals)
            self.update_live_controls()
            if self.pause_simulation.isChecked():
                self.progress.setText("Resume to settle or tug the fabric")
            else:
                self.progress.setText(self.proxy_note() + f"{phase} {elapsed:.1f} s")
        else:
            self.progress.setText(self.proxy_note() + f"{phase} {elapsed:.1f} / {self.active_duration:.1f} s",
                                  progress=elapsed / max(self.active_duration, 1e-6))

    def proxy_note(self):
        proxy = getattr(self.worker, "proxy", None) if self.worker is not None else None
        return f"Proxy {proxy.count:,} of {proxy.full_count:,} vertices · " if proxy is not None else ""

    def render_colliders(self, positions):
        for index, vertices in positions.items():
            self.collider_meshes[index].GetPoints().SetData(numpy_to_vtk(vertices, deep=True))

    @Slot(str)
    def report_error(self, message):
        self.simulation_failed = True
        self.status.set_state("failed")
        self.progress.setText("Stopped because the simulation failed")
        if not self.close_pending:
            QMessageBox.critical(self, "Drape simulation", message)

    @Slot()
    def record_speed(self):
        """Measured frames per wall-clock second since the first frame (for the size advice)."""
        if self.first_frame_time is None or self.worker is None:
            return
        frames = getattr(self.worker, "frame_number", 0) - self.first_frame_count
        seconds = time.monotonic() - self.first_frame_time
        if frames >= 10 and seconds > .5 and self.measured_run():
            drape_performance.record_frames(self.simulated_vertices(), frames, seconds)
        self.first_frame_time = None

    def finish(self):
        cancelled = self.preparing
        self.preparing = False
        self.record_speed()
        if getattr(self.worker, "proxy", None) is not None:
            self.proxy = self.worker.proxy                # reused by the next run of this garment
        self.release_grab()
        self.live = None
        self.live_ready = False
        # A run cancelled while preparing has no result (the finished thread may no longer
        # report its interruption request).
        self.result_ready = not cancelled and not self.simulation_failed and not self.worker.isInterruptionRequested()
        if self.result_ready:
            self.animation = self.worker.animation
            if self.animation is not None and self.motion is not None:
                self.finished_loop = bool(getattr(self.worker, "loop_status", ""))
        elif self.worker.animation is not None:
            self.worker.animation.close()
        if not self.simulation_failed:
            state = "Stopped" if self.worker.isInterruptionRequested() else "Finished"
            fps = self.motion.fps if self.motion is not None else 60
            if cancelled:
                self.status.set_state("idle")
                self.progress.setText("Preparation canceled.")
            else:
                self.status.set_state("finished")
                loop = getattr(self.worker, "loop_status", "") if state == "Finished" else ""
                self.progress.setText(loop.split(" (")[0] + "." if loop else f"{state} at {self.frame_number / fps:.2f} s")
        self.worker = None
        if self.result_ready and self.animation is not None and self.motion is not None and hasattr(self, "playback"):
            # A finished animation opens on its first frame, ready to play.
            self.playback.set_animation(self.motion.start_frame, self.motion.start_frame + self.motion.intervals,
                                        self.motion.fps, looped=getattr(self, "finished_loop", False))
        self.actions.show_run_state(False, False)
        self.stop_button.setText("Stop")
        self.stop_button.setVisible(False)
        self.output_mode.setEnabled(True)
        self.update_live_controls()
        self.update_send_button()
        for widget in self.edit_controls:
            widget.setEnabled(True)
        self.fastening_gap.setEnabled(self.fastening_mode.currentData() == "close")
        self.update_cut_note()
        self.proxy_size.setEnabled(self.use_proxy.isChecked())
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.continue_from is not None and not self.close_pending:
            start, self.continue_from = self.continue_from, None
            self.start_drape(start=start)
            return
        self.continue_from = None
        if self.reset_pending:
            self.reset_pending = False
            self.reset_dress()
        if self.close_pending:
            self.close()

    @Slot()
    @Slot(str, float)
    def on_preparing(self, text, fraction):
        if self.preparing and self.worker is not None and not self.worker.isInterruptionRequested():
            self.progress.setText("Preparing: " + text + "…", progress=fraction)

    @staticmethod
    def measured_run():
        """Only real solver runs feed the speed record (tests replace the solver)."""
        return simulate is warp_drape.simulate

    def switch_proxy_while_paused(self):
        """Proxy on/off while a live run is paused: stop it and continue from the current shape."""
        if (self.worker is None or self.live is None or not self.live.paused_ack or self.return_worker is not None
                or self.continue_from is not None):
            return
        self.continue_from = self.vertices.copy()
        self.progress.setText("Switching proxy simulation " + ("on" if self.use_proxy.isChecked() else "off") + "…")
        self.worker.requestInterruption()
        self.live.finish()

    def ground_settings(self):
        height = self.ground_height()
        if height is None:
            return None
        ground = dict(height_cm=float(height), grip=self.ground_grip.value() / 100)
        if self.ground_mode.currentData() == "lowest" and self.motion is not None and self.ground_frame_height is None:
            ground["lowest"] = self.ground_avatar()                # the worker checks every frame
        return ground

    def solver_passes(self):
        """The Solver passes value, or for Auto about 8 per 30k simulated vertices' worth of density
        (sqrt scaling: 300k -> 25), between 8 and 60."""
        if self.iterations.value():
            return self.iterations.value()
        count = min(len(self.garment[0]), self.proxy_size.value()) if self.use_proxy.isChecked() else len(self.garment[0])
        return int(max(8, min(60, round(8 * (count / 30000) ** .5))))

    def simulated_vertices(self):
        proxy = getattr(self.worker, "proxy", None) if self.worker is not None else None
        return proxy.count if proxy is not None else len(self.garment[0])

    def show_size_advice(self):
        """Status-bar advice for heavy garments, predicted from this PC's measured Drape runs."""
        count = len(self.garment[0])
        proxy = self.use_proxy.isChecked()
        simulated = min(count, self.proxy_size.value()) if proxy else count
        advice = drape_performance.advice(count, simulated, proxy)
        warning = bool(advice) and not proxy
        self.proxy_advice.setText(advice)
        self.proxy_advice.setVisible(bool(advice))
        self.proxy_advice.setStyleSheet("font-style: italic;" + (" color: #f2c94c;" if warning else ""))
        if self.worker is None:                          # while running, the status shows the run
            self.status.setText(drape_performance.short_advice(count, simulated, proxy) or
                                "Ready — press Run simulation" + spin_ui.key_hint("drape", "pause_resume") + ".")
            self.status.activity.setStyleSheet("color: #f2c94c;" if warning else "")

    def end_preparation(self):
        if self.preparing and self.run_started is not None and self.measured_run():
            self.first_frame_time = time.monotonic()
            self.first_frame_count = getattr(self.worker, "frame_number", 0)
            drape_performance.record_preparation(self.simulated_vertices(), self.first_frame_time - self.run_started,
                                                 proxy_full=len(self.garment[0]) if getattr(self.worker, "proxy", None) else 0)
        if self.preparing and getattr(self.worker, "duration", None):
            self.active_duration = self.worker.duration
        if self.preparing and getattr(self.worker, "ground_lowest", None) is not None:
            self.ground_frame_height = self.worker.ground_height_cm
            self.update_ground(render=False)
        if self.preparing:
            self.preparing = False
            self.status.set_progress(None)
            self.actions.show_run_state(True, False)
            self.update_live_controls()

    def cancel_preparation(self):
        """Stop a run before its first frame (Run button, Stop or Esc while preparing)."""
        if self.preparing and self.worker is not None:
            self.worker.requestInterruption()
            self.run_button.setEnabled(False)
            self.progress.setText("Canceling preparation…")

    def stop_drape(self):
        if self.preparing:
            self.cancel_preparation()
            return
        if self.worker is not None:
            self.release_grab()
            if self.live is not None:
                self.pause_simulation.setChecked(True)
                return
            self.worker.requestInterruption()
            self.stop_button.setEnabled(False)
            self.progress.setText("Stopping…")

    def closeEvent(self, event):
        if self.return_worker is not None:
            self.close_pending = True
            self.progress.setText("Finishing return before closing…")
            event.ignore()
            return
        if self.worker is not None:
            self.close_pending = True
            self.worker.requestInterruption()
            self.progress.setText("Stopping cloth solver before closing…")
            event.ignore()
            return
        spin_ui.save_window(self, self.settings, self.splitter)
        self.clear_animation()
        super().closeEvent(event)

    def show_recorded_frame(self, frame):
        """Draw one recorded DAZ frame of the finished animation, with the avatar at that frame."""
        if self.animation is None or self.motion is None or self.worker is not None:
            return
        sample = frame - self.motion.start_frame
        try:
            vertices = np.asarray(self.animation.get(sample), dtype=float)
        except (OSError, ValueError):
            return
        self.render_colliders(self.motion.preview(sample))
        self.render_vertices(vertices)
        self.status.set_frame(frame, self.motion.start_frame + self.motion.intervals)

    def clear_animation(self):
        if hasattr(self, "playback"):
            self.playback.clear()
        if self.animation is not None:
            self.animation.close()
            self.animation = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session")
    parser.add_argument("--garment", nargs=2)
    parser.add_argument("--collision", nargs=2, action="append")
    args = parser.parse_args()
    if args.session and (args.garment or args.collision):
        parser.error("Use a scene session or individual mesh arguments.")
    if not args.session and not args.garment:
        parser.error("Select a scene session or a garment mesh. Collision meshes are optional.")
    app = QApplication([])
    suite_app = str(Path(__file__).resolve().parents[2] / 'app')   # suite app/: taskbar icon
    if suite_app not in sys.path:
        sys.path.append(suite_app)
    import spin_icon
    spin_icon.set_app_icon(app, 'drape')
    live_session = None
    try:
        if args.session:
            if not is_live(args.session):
                raise ValueError("This capture predates the Spin Tools receive window. Recapture it from DAZ.")
            inputs = load_session(args.session)
            live_session = inputs["live_session"] = Path(args.session)
        else:
            inputs = load_inputs(
                [dict(path=args.garment[0], label=args.garment[1])],
                [dict(path=path, label=label) for path, label in (args.collision or [])],
            )
            inputs["garment_paths"] = [args.garment[0]]
        window = DrapeWindow(**inputs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        QMessageBox.critical(None, APP_NAME, str(error))
        return 1
    window.show()
    window.viewport.Initialize()
    try:
        return app.exec()
    finally:
        if live_session is not None:
            try:
                release_capture(live_session)
            except (OSError, ValueError):
                # A cleanup failure must not invalidate a completed revision.
                pass


if __name__ == "__main__":
    sys.exit(main())
