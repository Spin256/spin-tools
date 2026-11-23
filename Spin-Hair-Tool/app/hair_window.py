# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Spin Hair Tool window: view, freeze, gravity/wind/water, live pulling, settle on the GPU and send to DAZ."""
import math
import threading
import time
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from PySide6.QtCore import QSettings, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QSpinBox,
                               QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSlider, QSplitter,
                               QVBoxLayout, QWidget)

from branding import APP_NAME, VERSION
from drape_return import write_static_result
from gravity_controls import GravityPanel
from gravity_settings import GravityControl
from hair_solver import GrabControl
from collider_roles import primary_index
from hair_strands import build_hair, merge_models, tip_anchors, unanchored
from hair_grow_panel import GrowPanel, with_grown
import hair_looks
from eye_guard import EyeGuardControl, eye_vertices, fit_eye_guard
from hair_looks import FLAT, TEXTURES, COLORS as SURFACE_COLORS    # (COLORS below is the strand colors)
from vtkmodules.util.numpy_support import vtk_to_numpy
from live_return import publish_revision
from water_controls import WaterPanel, WaterPreview
from water_settings import WaterControl
from wind_controls import WindPanel
from wind_preview import WindPreview
from wind_settings import WindControl

import sys as _sys
from pathlib import Path as _Path
_SUITE_APP = str(_Path(__file__).resolve().parents[2] / "app")
if _SUITE_APP not in _sys.path:
    _sys.path.append(_SUITE_APP)          # suite app/: shared interface (spin_ui) and keybindings
import spin_ui

COLORS = {"scalp": (70, 160, 255), "parent": (90, 205, 120), "free": (235, 70, 70),
          "frozen": (175, 110, 235), "selected": (255, 215, 60), "pulled": (255, 70, 200),
          "pinned": (60, 215, 215)}
FOREVER = 1 << 60


def polydata(points, cells):
    """VTK poly data from points and (F, k) cells padded with -1 (quads/triangles)."""
    data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(points, np.float32), deep=True))
    data.SetPoints(vtk_points)
    sizes = (cells >= 0).sum(1)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    connectivity = cells[cells >= 0].astype(np.int64)
    array = vtk.vtkCellArray()
    array.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(connectivity, deep=True))
    data.SetPolys(array)
    return data


def strand_lines(particles, start, count):
    data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(particles, np.float32), deep=True))
    data.SetPoints(vtk_points)
    offsets = np.concatenate([start, [len(particles)]]).astype(np.int64)
    array = vtk.vtkCellArray()
    array.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(particles), dtype=np.int64), deep=True))
    data.SetLines(array)
    return data


def set_points(data, points):
    data.GetPoints().SetData(numpy_to_vtk(np.asarray(points, np.float32), deep=True))
    data.Modified()


class BuildWorker(QThread):
    progress = Signal(str)
    done = Signal(object, str)

    def __init__(self, inputs, parent=None, cap=False, cap_density=1.0, captured=None, area_models=None):
        super().__init__(parent)
        self.inputs, self.cap, self.cap_density = inputs, cap, cap_density
        self.result_inputs = inputs
        self.captured = captured              # the captured hair's model, kept from the first analysis
        self.area_models = {} if area_models is None else area_models     # each grown area's own analysis

    def run(self):
        try:
            grown = self.inputs.get("grown")
            if grown and self.captured is not None and len(self.captured.rest) == grown[0]["start"]:
                # Only the grown strips are analyzed; the captured hair's analysis is kept.
                # Each area is analyzed on its own and kept, so changing one area analyzes only that one.
                model, kept, root_start = self.captured, {}, 0
                uvs = self.inputs.get("vertex_uvs")
                for part in grown:
                    start, end = part["start"], part["end"]
                    roots = self.inputs["grown_roots"][root_start:root_start + part["strands"]]
                    root_start += part["strands"]
                    area_model = self.area_models.get(part.get("key")) if part.get("key") is not None else None
                    if area_model is None:
                        self.progress.emit(f"Analyzing the grown hair: {part['name']}…")
                        faces = np.asarray(self.inputs["faces"])[part["face_start"]:part["face_end"]]
                        area_model = build_hair(self.inputs["vertices"][start:end], np.where(faces >= 0, faces - start, -1),
                                                np.asarray(self.inputs["face_surface"])[part["face_start"]:part["face_end"]],
                                                self.inputs["surface_names"],
                                                vertex_uvs=None if uvs is None else uvs[start:end],
                                                extra_roots=roots, extra_from=0)
                    kept[part.get("key")] = area_model
                    model = merge_models(model, area_model, start)
                self.area_models.clear()
                self.area_models.update(kept)
                self.done.emit(model, "")
                return
            model = build_hair(self.inputs["vertices"], self.inputs["faces"], self.inputs["face_surface"],
                               self.inputs["surface_names"], progress=self.progress.emit,
                               vertex_uvs=self.inputs.get("vertex_uvs"),
                               extra_roots=self.inputs.get("grown_roots"),    # grown hair is rooted on the skin
                               extra_from=(self.inputs["grown"][0]["start"] if self.inputs.get("grown") else None),
                               # hair objects with no scalp (a beard, grown hair captured again) root on the skin
                               skin_points=self.inputs["colliders"][primary_index(self.inputs["colliders"])]["vertices"],
                               objects=self.inputs.get("ranges"))
            if self.cap:
                from hair_cap import build_cap, with_cap
                self.progress.emit("Growing the scalp cap…")
                cap = build_cap(self.inputs, model, density=self.cap_density)
                if cap is not None:
                    self.result_inputs = with_cap(self.inputs, cap)
            self.done.emit(model, "")
        except Exception as error:      # reported in the window
            self.done.emit(None, str(error))


class SimulationWorker(QThread):
    """Runs frames until ``run_until``; the window moves that target to pause, resume or keep pulling."""
    frame = Signal()                          # a newer frame is ready; collect it with take_frame()
    failed = Signal(str)
    paused = Signal()

    def __init__(self, model, colliders, frozen, settings, controls, parent=None, faces=None, cap=None,
                 motion=None, frames=None, diagnostic_path=None):
        super().__init__(parent)
        self.faces = faces
        # Timeline: settle at the start frame, then play the avatar's movement; ``frames``
        # (AnimationFrames) records one hair frame per DAZ frame for an animation return.
        self.motion, self.frames, self.next_sample = motion, frames, 0
        # Scalp cap: fixed cards on the scalp, outside the simulation, appended to every frame.
        self.cap = cap
        self.model, self.colliders, self.frozen, self.settings = model, colliders, frozen, settings
        self.controls = controls
        from hair_diagnostics import SimulationTrace
        self.trace = SimulationTrace(diagnostic_path,settings)
        self.lock = threading.Lock()
        self.target = int(round(settings["duration"] * 60))
        self.settle = self.target
        if motion is not None:
            self.target += int(math.ceil(motion.duration * 60)) + 1
            if settings.get("loop"):
                # each warm-up loop ends with one real step into frame 0 (the wrap)
                self.target += int(settings.get("warm_loops", 0)) * (int(math.ceil(motion.duration * 60)) +
                                                                     int(math.ceil(60 / motion.fps)) + 1)
        self.full_target = self.target if motion is not None else None
        self.frames_done = 0
        self.loop_status = ''
        self.looped = False
        self.idle = False
        self.latest = None
        self.notified = False
        self.edits = []
        self.editing = False
        self.requested = False

    def edits_pending(self):
        with self.lock:
            return self.editing or bool(self.edits)

    def edit(self, kind, payload=None):
        with self.lock:
            # Keep the newest drag location, but never discard a release/commit.
            if kind == "pull" and self.edits and self.edits[-1][0] == "pull":
                self.edits[-1] = (kind, payload)
            else:
                self.edits.append((kind, payload))

    def request(self, mode, settle_frames=0):
        """'pause', 'settle' (run settle_frames more) or 'hold' (run until released)."""
        with self.lock:
            self.requested = True
            if mode == "pause":
                self.target = self.frames_done
            elif mode == "settle":
                self.target = self.frames_done + settle_frames
            elif mode == "resume" and self.full_target is not None:
                self.target = self.full_target          # carry on recording to the end of the movement
            else:
                self.target = FOREVER

    def publish(self, sim):
        """Store the newest frame; signal the window only if it has taken the previous one."""
        vertices = sim.vertices()
        normals = sim.vertex_normals(self.faces)
        r, t = sim.head_transform()
        if self.cap is not None:
            vertices = np.concatenate([vertices, self.cap["vertices"] @ r.T + t])
            normals = np.concatenate([normals, self.cap["normals"] @ r.T])
        avatar = None
        if self.motion is not None:
            avatar = {index: self.motion.at(index, sim.time) for index in self.motion.samples}
        frame = (vertices, normals, sim.particles(), sim.frame / sim.fps, avatar,
                 None if self.motion is None else sim.time,
                 sim.wind_control.snapshot()[1] if sim.wind_control is not None else None, (r,t))
        with self.lock:
            self.latest = frame
            notify = not self.notified
            self.notified = True
        if notify:
            self.frame.emit()

    @staticmethod
    def _plain(value):
        if isinstance(value, (bool, int, float, np.floating, np.integer, np.ndarray)):
            return True
        if isinstance(value, tuple):
            return all(SimulationWorker._plain(item) for item in value)
        if isinstance(value, dict):
            return all(SimulationWorker._plain(item) for item in value.values())
        return False

    def carry_loop(self, sim, recorder):
        """Seamless loop for hair. Hair never repeats exactly (strands knock into each other),
        so it is not searched for a meeting point like cloth: the simulation simply continues past the last frame
        with its own momentum. The first run starts from no movement, so the hair is not pulled toward its early
        frames: from frame 1 to the middle of the animation, where the first run is moving too, it is pulled toward the first run's hair for each frame
        (the live-pull projection, collisions still apply), gently at first and fully at the middle, where it is
        placed exactly on the first run. Frame 0 becomes the last frame; frames 1..middle-1 are the carried hair."""
        fps = self.motion.fps
        landing = recorder.middle
        steps = landing          # the whole stretch from frame 1 to the middle (no setting)
        pull_from = landing - steps
        sim.playing = True
        sim.begin_wrap()                       # one real step from the last frame into frame 0
        while getattr(sim, "wrap", None) is not None:
            if self.isInterruptionRequested():
                return None
            sim.step()
        free = (sim.inv.numpy() > 0).astype(np.float32)
        published = [(0, sim.vertices())]
        try:
            for sample in range(1, landing + 1):
                t = max(0., (sample - pull_from) / steps)
                strength = t * t * (3. - 2. * t)
                sim.targets.assign(np.asarray(recorder.states[sample], np.float32))
                sim.weights.assign(free * (.25 * strength))
                sim.grabbing = True
                while int(math.floor(sim.time * fps + 1e-4)) < sample:
                    if self.isInterruptionRequested():
                        return None
                    sim.step()
                if sample == landing:
                    target = np.asarray(recorder.states[sample], np.float32)
                    shift = target - sim.pos.numpy()
                    sim.pos.assign(target)
                    sim.prev.assign(sim.prev.numpy() + shift)
                    sim.output_frame = -1
                else:
                    published.append((sample, sim.vertices()))
                if time.monotonic() - getattr(self, "_loop_preview", 0.) > 1 / 30:
                    self._loop_preview = time.monotonic()
                    self.publish(sim)
        finally:
            sim.weights.zero_()
            sim.grabbing = False
            sim.grab_revision = None          # the live pull control takes over again
        for sample, vertices in published:
            self.frames.put(sample, vertices)
        return f"Loop closed with motion carried (pulled onto frame {self.motion.start_frame + landing})."

    def take_frame(self):
        with self.lock:
            frame, self.latest, self.notified = self.latest, None, False
        return frame

    def is_paused(self):
        with self.lock:
            return self.idle

    def run(self):
        try:
            from hair_solver import HairSimulation
            pairs = tip_anchors(self.model, self.frozen)
            s, c = self.settings, self.controls
            sim = HairSimulation(self.model, self.colliders, self.frozen, pairs, bend=s["bend"], damping=s["damping"],
                                 radius=s["radius"], friction=s["friction"], style=s["style"], style_length=s["style_length"], shelter=s["shelter"],
                                 wind_response=s["wind_response"], hairspray=s["hairspray"],
                                 hair_collision=s["hair_collision"], hair_spacing=s["hair_spacing"],
                                 trapped_distance=s.get("trapped_distance", 1.0),
                                 max_move_cm=s.get("max_move_cm", 0.0),
                                 root_lift=s["root_lift"], stiffness=s["stiffness"], stiffness_swing=True, momentum=s["momentum"], style_rest=s["style_rest"], clumping=s.get("clumping", 0.), head_drag=s.get("head_drag", 0.), pinned_ends=s.get("pinned_ends", ()),
                                 weightless_lift=s["weightless_lift"],
                                 gravity_control=c["gravity"], wind_control=c["wind"], water_control=c["water"],
                                 grab_control=c["grab"], motion=self.motion, head_collider=s.get("head_collider"),
                                 loose=s.get("loose"), eye_guard_control=c.get("eye_guard"))
            if self.motion is not None:
                with self.lock:
                    passes = 1 + (int(s.get("warm_loops", 0)) if s.get("loop") else 0)
                    wrap = int(math.ceil(sim.fps / self.motion.fps)) + 1          # each warm-up loop's step into frame 0
                    self.full_target = (self.settle + passes * int(math.ceil(sim.motion_timing.duration * sim.fps))
                                        + (passes - 1) * wrap + 1)
                    if not self.requested:
                        self.target = self.full_target
            last = 0.0
            recorder = None
            if self.motion is not None and self.frames is not None and s.get("loop") and self.motion.intervals >= 3:
                from spin_loop import LoopRecorder
                recorder = LoopRecorder(self.motion.intervals)
            # Warm-up loops: play the movement this many times before recording, carrying the
            # hair from each loop's end into the next start, so the recorded loop starts already moving.
            self.warm_left = int(s.get("warm_loops", 0)) if recorder is not None else 0
            self.warm_change = None
            warm_end = None
            while not self.isInterruptionRequested():
                with self.lock:
                    running = self.frames_done < self.target
                    was_idle, self.idle = self.idle, not running
                    edits, self.edits = self.edits, []
                    self.editing = bool(edits)
                for kind, payload in edits:
                    if kind == "pull":
                        sim.edit_pull(*payload)
                    elif kind == "hold":
                        sim.hold_pull(payload)
                    elif kind == "release":
                        sim.release_pull_holds()
                if edits:
                    self.publish(sim)
                    with self.lock:
                        self.editing = False
                    if not running:
                        self.paused.emit()
                if not running:
                    if not was_idle:
                        self.trace.record(sim,paused=True)
                        self.publish(sim)
                        self.paused.emit()
                    self.msleep(15)
                    continue
                if self.motion is not None:
                    sim.playing = sim.frame >= self.settle
                    if (sim.playing and self.frames is not None and self.next_sample == 0 and not self.warm_left
                            and getattr(sim, "wrap", None) is None):
                        self.frames.put(0, sim.vertices())
                        self.next_sample = 1
                sim.step()
                self.trace.record(sim)
                if (self.warm_left and sim.playing and getattr(sim, "wrap", None) is None
                        and sim.motion_elapsed >= sim.motion_timing.duration - 1e-9):
                    end = sim.pos.numpy()
                    if warm_end is not None:      # how much the loop's end still moved since the previous loop (cm)
                        self.warm_change = float(np.percentile(np.linalg.norm(end - warm_end, axis=1), 95))
                    warm_end = end
                    self.warm_left -= 1
                    sim.begin_wrap()                                    # one real step into frame 0, then again
                elif (self.motion is not None and sim.playing and self.frames is not None and not self.warm_left
                      and getattr(sim, "wrap", None) is None):
                    reached = int(math.floor(sim.time * self.motion.fps + 1e-4))
                    if self.next_sample <= min(reached, self.frames.count - 1):
                        vertices = sim.vertices()
                        while self.next_sample <= min(reached, self.frames.count - 1):
                            self.frames.put(self.next_sample, vertices)
                            if recorder is not None:
                                recorder.record(self.next_sample, sim.pos.numpy())
                            self.next_sample += 1
                        if recorder is not None and not self.looped and self.next_sample >= self.frames.count:
                            # Seamless loop: from the moment the last frame is recorded,
                            # continue into frames 1, 2, ... until the hair meets the first run.
                            self.looped = True
                            self.loop_status = self.carry_loop(sim, recorder) or ''
                            if self.loop_status and self.warm_change is not None:
                                self.loop_status = self.loop_status[:-2] + (
                                    f'; the last warm-up loop still changed the hair by {self.warm_change:.1f} cm).')
                with self.lock:
                    self.frames_done = sim.frame
                if time.monotonic() - last > 1 / 30:
                    last = time.monotonic()
                    self.publish(sim)
        except Exception as error:
            self.failed.emit(str(error))


class ReturnWorker(QThread):
    def __init__(self, inputs, vertices, parent=None, spec=None, frames=None):
        super().__init__(parent)
        self.inputs, self.vertices, self.error, self.frames = inputs, vertices, None, frames
        self.spec = spec if spec is not None else inputs["return_spec"]

    def run(self):
        try:
            if self.frames is not None:
                from drape_return import write_animation_result
                write = lambda spec: write_animation_result(spec, self.frames, self.inputs["ranges"])
            else:
                def write(spec):
                    grown = self.inputs.get("grown") or []
                    end = grown[0]["start"] if grown else len(self.vertices)
                    write_static_result(spec, self.vertices[:end], self.inputs["ranges"], cap=self.inputs.get("cap"))
                    if grown:
                        from hair_grow_return import add_grown
                        add_grown(spec, self.vertices, self.inputs)
            publish_revision(self.inputs["live_session"], self.spec, write)
        except Exception as error:
            self.error = str(error)


class HairWindow(QWidget):
    def __init__(self, inputs):
        super().__init__()
        self.standalone = inputs.get("standalone")          # files from disk, results exported to files
        # A captured plane named Water is the water surface (as in the Drape Tool), not a solid.
        from hair_water import split_water
        inputs, self.water_source = split_water(inputs)
        self.inputs = inputs
        self.model = None
        self.vertices = inputs["vertices"].copy()
        self.particles = None
        self.frozen, self.selected = set(), set()
        self.pinned = set()          # strands held at both ends
        self.worker = self.return_worker = self.builder = None
        self.result_ready = False
        self.close_pending = False
        self.recorded, self.timeline = None, None
        self.wind_pose = (np.eye(3), np.zeros(3))
        self.wind_head = None
        self.wind_initialized = False
        self.wind_aim_rotation = np.eye(3)
        # No captured hair (an avatar on its own, to grow hair on it): positions fall back to the avatar.
        around = self.vertices if len(self.vertices) else inputs["colliders"][primary_index(inputs["colliders"])]["vertices"]
        self.wind_start_center = (around.min(0) + around.max(0)) * .5
        self.pick_mode = None
        self.drag = None
        self.gravity_control, self.wind_control = GravityControl(), WindControl()
        self.water_control, self.grab_control = WaterControl(), GrabControl()
        self.eye_guard_control = EyeGuardControl()
        self.settings = QSettings('Spin256', APP_NAME)
        self.setWindowTitle(APP_NAME + " — " + inputs["hair_label"])
        self.resize(1300, 860)
        layout = QHBoxLayout(self)
        self.splitter = splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        panel = QWidget()
        controls = QVBoxLayout(panel)

        self.info = QLabel(inputs["hair_label"])
        self.info.setWordWrap(True)
        controls.addWidget(self.info)
        self.sections = spin_ui.Sections(controls, self.settings)

        view = self.section(controls, "View", True)
        self.view = spin_ui.ViewControls(view, self.settings, "hair", self.update_view, avatar_name="avatar", tool="hair")
        self.show_avatar, self.show_hair = self.view.show_avatar, self.view.show_object
        self.opacity, self.avatar_opacity = self.view.object_opacity, self.view.avatar_opacity
        self.show_collisions = QCheckBox("Show collision objects")
        self.show_collisions.setChecked(True)
        self.show_collisions.setToolTip("Preview only. Hidden objects still take part in simulation.")
        self.collision_opacity = QSlider(Qt.Orientation.Horizontal)
        self.collision_opacity.setRange(5, 100)
        self.collision_opacity.setValue(self.settings.value("view/collision_opacity", self.avatar_opacity.value(), type=int))
        self.collision_opacity.setToolTip("Preview opacity for independent collision objects, such as beds and pillows.")
        collision_form = QFormLayout()
        collision_form.addRow("Collision objects opacity", self.collision_opacity)
        # Keep category visibility and opacity together with the shared view controls.
        view.insertWidget(2, self.show_collisions)
        view.insertLayout(4, collision_form)
        self.show_collisions.toggled.connect(lambda *_: self.update_view())
        self.collision_opacity.valueChanged.connect(lambda value: self.settings.setValue("view/collision_opacity", value))
        self.collision_opacity.valueChanged.connect(lambda *_: self.update_view())
        self.xray = QCheckBox("X-ray: strands only (see through to ties and buns)")
        view.addWidget(self.xray)
        # Display: the avatar and hair in their DAZ surface colors or textures (grown hair in
        # its materials' colors), which makes painting easier. Preview only.
        self.display = QComboBox()
        self.display.addItem("Flat", FLAT)
        self.display.addItem("Surface colors", SURFACE_COLORS)
        self.display.addItem("Textures", TEXTURES)
        self.display.setToolTip("Surface colors and textures come from the captured DAZ surfaces (base color and color "
                                "map). Hair textures are drawn without their transparency maps.")
        display_form = QFormLayout()
        display_form.addRow("Display", self.display)
        view.addLayout(display_form)
        # Smooth shading on its own: every Display mode, on by default; off shows the polygons.
        self.smooth_shading = QCheckBox("Smooth shading")
        self.smooth_shading.setChecked(self.settings.value("view/smooth", True, type=bool))
        self.smooth_shading.setToolTip("Off: each polygon is lit flat, which shows the mesh's polygons.")
        view.addWidget(self.smooth_shading)
        self.view_layout = view

        freeze = self.section(controls, "Select and freeze", True)
        help_text = QLabel("Frozen hair stays as styled (fixed to the head). Strand ends touching frozen hair "
                           "are held by it, so a tie or bun keeps the ponytail in place.")
        help_text.setWordWrap(True)
        freeze.addWidget(help_text)
        self.pick_card = QPushButton("Pick strands (click; Ctrl adds, Alt removes)"); self.pick_card.setCheckable(True)
        self.ring = QPushButton("Ring select (click; Ctrl adds, Alt removes)"); self.ring.setCheckable(True)
        self.ring_radius = self.number(.2, 20, 2.0, .2, " cm")
        form = QFormLayout(); form.addRow("Ring radius", self.ring_radius)
        self.surface = QComboBox()
        for name in inputs["surface_names"]:
            self.surface.addItem(name)
        select_surface = QPushButton("Select surface")
        for widget in (self.pick_card, self.ring):
            freeze.addWidget(widget)
        freeze.addLayout(form)
        surface_row = QHBoxLayout(); surface_row.addWidget(self.surface, 1); surface_row.addWidget(select_surface)
        freeze.addLayout(surface_row)
        buttons = QHBoxLayout()
        self.freeze_button = QPushButton("Freeze selected")
        self.unfreeze_button = QPushButton("Unfreeze selected")
        clear = QPushButton("Clear selection")
        for widget in (self.freeze_button, self.unfreeze_button, clear):
            buttons.addWidget(widget)
        freeze.addLayout(buttons)
        pin_buttons = QHBoxLayout()
        self.pin_button = QPushButton("Pin both ends")
        self.pin_button.setToolTip("Hold the selected strands at the scalp and at their far end (for hair swept back into a "
                                   "tie or tucked in); the hair in between still moves. Pinned strands show cyan.")
        self.unpin_button = QPushButton("Unpin selected")
        for widget in (self.pin_button, self.unpin_button):
            pin_buttons.addWidget(widget)
        freeze.addLayout(pin_buttons)
        self.shortcut_note = QLabel(); self.shortcut_note.setWordWrap(True)
        self.selection_note = QLabel()
        self.selection_note.setWordWrap(True)
        freeze.addWidget(self.selection_note)

        # Grow hair: named areas painted on the avatar grow straight strands.
        grow = self.section(controls, "Grow hair", False)
        avatar = inputs["colliders"][primary_index(inputs["colliders"])]
        capture = (inputs.get("return_spec") or {}).get("session_id") or str(inputs.get("live_session") or "none")
        self.grow_panel = GrowPanel(grow, self.settings, avatar["vertices"], avatar["triangles"], self, capture=capture,
                                    avatar_label=avatar.get("label", "Avatar"))
        self.grow_panel.paint.toggled.connect(lambda on: self.set_pick("grow" if on else None))
        self.grow_panel.changed.connect(self.refresh_grow_overlay)
        self.grow_panel.grow_requested.connect(self.grow_hair)
        self.grow_overlay = None

        simulation = self.section(controls, "Simulation", True)
        # Presets: Static and Animation built in, named presets saved to hair_presets.json.
        preset_form = QFormLayout()
        self.preset = QComboBox()
        self.preset.setToolTip("Static: free, lively hair for a still pose you refine. Animation: hair that moves "
                               "together and stays near its style, for every frame of a movement. Changing a setting "
                               "below makes it Custom; Save keeps your settings as a named preset.")
        preset_form.addRow("Preset", self.preset)
        preset_buttons = QHBoxLayout()
        self.save_preset_button = QPushButton("Save preset…")
        self.delete_preset_button = QPushButton("Delete preset")
        preset_buttons.addWidget(self.save_preset_button); preset_buttons.addWidget(self.delete_preset_button)
        preset_form.addRow(preset_buttons)
        simulation.addLayout(preset_form)
        sim_form = QFormLayout()
        self.duration = self.number(.2, 30, 3, .5, " s")
        self.duration.setToolTip("Run settles for this long, then pauses. Resume then runs until you pause"
                                 + spin_ui.key_hint("hair", "pause_resume") + ".")
        self.style = self.number(0, 100, 50, 5, " %")
        self.style.setToolTip("Holds the styled shape: strongest at the roots, fading out over the style reach.")
        self.style_reach = self.number(1, 300, 6, 1, " cm")
        self.style_reach.setToolTip("How far from each root the style hold reaches (parting, bun, ponytail base). "
                                    "Beyond it hair is held only by its root, so it lifts off shoulders, blows and "
                                    "stays where you pull it. Longer reaches pull lengths back to their captured place.")
        self.shelter = self.number(0, 100, 60, 5, " %")
        self.shelter.setToolTip("How much the avatar and props block wind reaching hair behind them "
                                "(with shelter on in the Wind panel). Air flows around a head, so not 100%.")
        self.bend = self.number(0, 100, 20, 5, " %")
        self.bend.setToolTip("Resists small kinks along each strand (share undone per frame).")
        self.hairspray = self.number(0, 100, 0, 5, " %")
        self.root_lift = self.number(0, 99.9, 80, 5, " %")
        self.root_lift.setToolTip("How firmly hair keeps the angle it leaves the scalp or tie at (like a follicle).")
        self.stiffness = self.number(0, 99.9, 90, 5, " %")
        self.stiffness.setToolTip("How strongly each strand keeps its own shape over its whole length (real hair's "
                                  "stiffness). 0% hangs like string; 90% holds a styled ponytail's arc.")
        self.lift = self.number(0, 100, 60, 5, " %")
        self.lift.setToolTip("How far hair stands out from the head when weightless (low gravity, or under "
                             "water where buoyancy carries its weight), keeping its curves. 0% keeps the style; "
                             "100% leaves the scalp straight out. No effect at normal gravity.")
        self.style_rest = self.number(0, 100, 100, 5, " %")
        self.style_rest.setToolTip("How much of the DAZ style an upright figure keeps under normal gravity with "
                                   "nothing else acting. The rest of a full head of hair holds these strands up; "
                                   "stronger, weaker or sideways gravity, wind and pulling still move them. "
                                   "0% hangs freely.")
        self.hair_collide = QCheckBox("Hair-to-hair collision (keeps volume)")
        self.hair_collide.setChecked(True)
        self.hair_collide.setToolTip("Strands keep their captured spacing, so layers don't sink into each other.")
        self.hair_spacing = self.number(.1, 5, .6, .1, " cm")
        self.hair_spacing.setToolTip("Most space strands keep between each other; never more than they had as captured.")
        self.hairspray.setToolTip("Hairspray / gel: keeps each strand's own curls and arcs over longer spans while it "
                                  "still moves freely as a whole. 0% loose hair; 50% sprayed; 90% firmly gelled.")
        self.damping = self.number(0, 20, 2, .2, " /s")
        self.clumping = self.number(0, 100, 0, 5, " %")
        self.clumping.setToolTip("Draws each bit of hair's motion toward the motion of the hair around it (within 2 cm), "
                                 "so locks and ponytails move together instead of every strand on its own.")
        self.head_drag = self.number(0, 30, 0, .5, " /s")
        self.head_drag.setToolTip("Air drag on motion relative to the head: hair moving with the head is not held back, "
                                  "hair whipping about relative to it settles. Higher keeps it closer to its style.")
        self.thickness = self.number(.5, 20, 5, .5, " mm")
        self.thickness.setToolTip("Hair keeps half this distance from the avatar.")
        self.trapped_distance = self.number(0, 30, 10, 1, " mm")
        self.trapped_distance.setToolTip("Hold hair between opposing surfaces when it is within this distance "
                                        "of both colliders. Contact follows the surfaces and releases when the "
                                        "gap opens. 0 disables. Change after Reset.")
        self.friction = self.number(0, 1, 0, .05)
        self.friction.setToolTip("How much the avatar resists hair sliding over it. Zero (default): hair slides freely "
                                 "off shoulders and the back.")
        self.wind_catch = self.number(0, 500, 100, 10, " %")
        self.wind_catch.setToolTip("How strongly wind grips the hair. Higher streams it out flat; lower lets gravity "
                                   "keep more shape (a ponytail arcs instead of forming a straight slope).")
        # Shown: what shapes the result. Bend stiffness and Hairspray are covered by Stiffness;
        # Style hold/reach and Root lift help a little and stay on at their defaults, not shown.
        # Wind catch and shelter sit in the Wind section.
        for label, widget in (("Initial settle", self.duration), ("Style at rest", self.style_rest),
                              ("Weightless lift", self.lift), ("Stiffness", self.stiffness),
                              ("Hair spacing", self.hair_spacing), ("Damping", self.damping),
                              ("Clumping", self.clumping), ("Head drag", self.head_drag),
                              ("Collision thickness", self.thickness),
                              ("Trapped contact distance", self.trapped_distance), ("Friction", self.friction)):
            sim_form.addRow(label, widget)
        simulation.addLayout(sim_form)
        simulation.addWidget(self.hair_collide)
        # Timeline capture (as the Drape Tool): settle at the start frame, play the movement, and
        # return either the final frame or every frame as an animation.
        self.motion = inputs.get("motion")
        if self.motion is not None:
            self.motion.prefetch()          # read the avatar's frames while the hair settles
        self.momentum = self.number(0, 100, 50, 5, " %")
        self.max_move = self.number(0, 20, .75, .25, " cm")
        self.max_move.setValue(self.settings.value("motion/max_move_cm", .75, type=float))
        self.max_move.setToolTip("Slows fast captured poses by adding simulation steps, as in the Drape Tool. "
                                "Smaller limits give hair more time to follow. 0 preserves DAZ timing. "
                                "Rigid limbs rotate in arcs between poses instead of shrinking. "
                                "All moving collision objects follow the same timing. Animation returns "
                                "keep the original DAZ frame numbers and frame rate.")
        self.max_move.valueChanged.connect(lambda value: self.settings.setValue("motion/max_move_cm", value))
        # Seamless loop.
        self.loop_animation = QCheckBox("Loop animation (carry motion)")
        self.loop_animation.setChecked(self.settings.value("motion/loop", False, type=bool))
        self.loop_animation.setToolTip("Animation return: after the last frame the hair carries on into frame 0 and on, and is "
                                       "eased onto the first run by the middle of the animation, so it repeats without "
                                       "restarting from rest. End the avatar's animation one pose before its first.")
        self.loop_animation.toggled.connect(lambda on: self.settings.setValue("motion/loop", on))
        self.warm_loops = QSpinBox()
        self.warm_loops.setRange(0, 10)
        self.warm_loops.setValue(self.settings.value("motion/warm_loops", 2, type=int))
        self.warm_loops.setToolTip("Play the movement this many times before recording, so the recorded loop starts with "
                                   "the hair already moving. More loops let lively hair settle into a repeating swing; "
                                   "each costs one more run of the movement.")
        self.warm_loops.valueChanged.connect(lambda value: self.settings.setValue("motion/warm_loops", value))
        self.momentum.setToolTip("While the movement plays, hair moving fast relative to its root eases its "
                                 "stiffness and carries on (a flick of the head throws the hair up and over); it "
                                 "springs back to its style as it slows. 0% keeps the stiffness throughout.")
        self.output_mode = QComboBox()
        self.output_mode.addItem("Final frame (still hair)", "static")
        self.output_mode.addItem("Animation (every frame)", "animation")
        if self.motion is not None:
            m = self.motion
            note = QLabel(f"Movement: DAZ frames {m.start_frame}–{m.end_frame} ({m.duration:.1f} s at {m.fps:g} fps). "
                          "Run settles at the start frame, then plays the movement. "
                          "The avatar speed limit may extend the simulation time.")
            note.setWordWrap(True)
            animation_note = note                    # the Animation section is added at the bottom (below)
            self.output_mode.setCurrentIndex(1 if inputs.get("output_mode") == "animation" else 0)
            self.output_mode.setToolTip("Final frame: the hair animates through the movement and the last frame is "
                                        "sent (you can keep settling it). Animation: every DAZ frame is recorded "
                                        "and sent as an animated hair.")
            mode_form = QFormLayout(); mode_form.addRow("Return preference", self.output_mode)
            # DAZ return format: DAZ native (DUF) or FBX.
            self.return_format = QComboBox()
            self.return_format.addItem("DAZ native (DUF)", "duf")
            self.return_format.addItem("FBX (compatibility)", "fbx")
            self.return_format.setCurrentIndex(max(0, self.return_format.findData(
                self.settings.value("return/animation_format", "duf"))))
            self.return_format.currentIndexChanged.connect(
                lambda *_: self.settings.setValue("return/animation_format", self.return_format.currentData()))
            self.return_format.setToolTip("How an animation is sent back to DAZ. DAZ native loads quickly; FBX is "
                                          "much slower for large hair.")
            mode_form.addRow("Momentum", self.momentum)
            mode_form.addRow("Avatar speed limit", self.max_move)
            mode_form.addRow(self.loop_animation)
            mode_form.addRow("Warm-up loops", self.warm_loops)
            # Playback of a recorded animation, with the animation settings.
            self.playback = spin_ui.PlaybackBar(self.show_recorded_frame)
            mode_form.addRow("Playback", self.playback)
            if self.standalone is None:
                mode_form.addRow("DAZ return format", self.return_format)   # same order as Drape and Soft Prop
            self.output_mode.currentIndexChanged.connect(lambda *_: self.mode_changed())
        self.scalp_cap = QCheckBox("Scalp cap (for extreme poses)")
        self.scalp_cap.setChecked(False)          # off at every start: only for extreme poses
        self.scalp_cap.setToolTip("Grows short extra hair on scalp the hair asset leaves without strands (usually "
                                  "hidden under hair), copying the nearest cards' look, so a figure bending over "
                                  "shows no bald patch. Adds polygons to the result. Leave off otherwise.")
        simulation.addWidget(self.scalp_cap)
        cap_form = QFormLayout()
        self.cap_density = self.number(25, 400, self.settings.value("scalp_cap_density", 100.0, type=float), 25, " %")
        self.cap_density.setToolTip("How many cap cards to grow. Hair shaders are see-through, so a thin cap still "
                                    "shows skin; raise this if it does (more polygons in the result).")
        self.cap_density.editingFinished.connect(self.cap_density_changed)
        cap_form.addRow("Cap density", self.cap_density)
        simulation.addLayout(cap_form)
        self.setup_presets()

        pull = self.section(controls, "Live pulling", True)
        self.pull = QPushButton("Pull hair (drag in the view)"); self.pull.setCheckable(True)
        self.pull_radius = self.number(.5, 30, 3, .5, " cm")
        self.pull_strength = self.number(5, 100, 70, 5, " %")
        self.pull_mode = QComboBox()
        self.pull_mode.addItem("Adjust (free on resume)", "adjust")
        self.pull_mode.addItem("Hold after pulling", "hold")
        self.release_holds = QPushButton("Release holds")
        self.release_holds.clicked.connect(self.release_pull_holds)
        pull.addWidget(self.pull)
        pull_form = QFormLayout(); pull_form.addRow("Pull radius", self.pull_radius)
        pull_form.addRow("Pull strength", self.pull_strength)
        pull_form.addRow("After pulling", self.pull_mode); pull.addLayout(pull_form)
        pull.addWidget(self.release_holds)
        pull_help = QLabel("Drag while running or paused. Paused pulls adjust the current result without advancing "
                           "time. Hold retains the pulled points when simulation resumes; Release holds frees them. "
                           "Scalp roots and frozen hair stay fixed. Pulling is unavailable for animation output.")
        pull_help.setWordWrap(True)
        pull.addWidget(pull_help)

        start = inputs["vertices"] if len(inputs["vertices"]) else around      # (no hair: the avatar)
        # Keep hair out of the eyes: a see-through oval over both eyes that hair drapes
        # around. Fitted from the eye surfaces on each capture (not saved); it moves with the head in animations.
        eyes = self.section(controls, "Keep hair out of eyes", False)
        self.eye_guard_on = QCheckBox("Eye guard: hair drapes around an oval tube over both eyes")
        self.eye_guard_show = QCheckBox("Show the tube"); self.eye_guard_show.setChecked(True)
        self.eye_width = self.number(.5, 40, 8, .1, " cm")
        self.eye_height = self.number(.5, 30, 4, .1, " cm")
        self.eye_depth = self.number(.5, 100, 25, .5, " cm")
        self.eye_raise = self.number(-10, 10, 0, .1, " cm")
        self.eye_width.setToolTip("Width of the oval, across both eyes.")
        self.eye_height.setToolTip("Height of the oval.")
        self.eye_depth.setToolTip("How far the tube reaches forward from the eyes.")
        self.eye_raise.setToolTip("Moves the oval up (or down, below 0) from the middle of the eyes.")
        self.eye_fit = QPushButton("Fit to eyes")
        self.eye_fit.setToolTip("Size and place the oval over both eyes again.")
        self.eye_note = QLabel(); self.eye_note.setWordWrap(True)
        eyes.addWidget(self.eye_guard_on)
        eye_form = QFormLayout()
        for label, widget in (("Width", self.eye_width), ("Height", self.eye_height), ("Depth", self.eye_depth),
                              ("Raise", self.eye_raise)):
            eye_form.addRow(label, widget)
        eyes.addLayout(eye_form)
        eye_row = QHBoxLayout(); eye_row.addWidget(self.eye_guard_show); eye_row.addWidget(self.eye_fit)
        eyes.addLayout(eye_row)
        eyes.addWidget(self.eye_note)
        gravity = self.section(controls, "Gravity", False)
        self.gravity_panel = GravityPanel(); gravity.addWidget(self.gravity_panel)
        wind = self.section(controls, "Wind", False)
        self.wind_panel = WindPanel(start); wind.addWidget(self.wind_panel)
        self.wind_panel.center_provider = self.wind_center
        self.wind_track = QCheckBox("Track avatar during animation")
        self.wind_track.setToolTip("Keeps the wind source position and direction relative to the avatar's head. "
                                   "Off: wind stays fixed in the scene. Source fields and overlay show the current world position.")
        self.wind_track.setChecked(self.settings.value("wind/track_avatar", False, type=bool))
        self.wind_track.setEnabled(self.motion is not None and self.animation())
        wind.addWidget(self.wind_track)
        self.wind_track.toggled.connect(self.wind_tracking_changed)
        wind_form = QFormLayout()
        wind_form.addRow("Wind catch", self.wind_catch)
        wind_form.addRow("Wind shelter", self.shelter)
        wind.addLayout(wind_form)
        # Hair defaults: flutter on, mild gusts (the panel's Turbulence is hair flutter here).
        self.wind_panel.fields["turbulence"].setValue(50)
        self.wind_panel.fields["gusts"].setValue(20)
        self.wind_panel.fields["turbulence"].setToolTip("Flutter: small eddies that make each strand whip on its own, "
                                                        "including in the wake behind the avatar.")
        water = self.section(controls, "Water", False)
        self.water_panel = WaterPanel(start, water_source=self.water_source); water.addWidget(self.water_panel)
        if self.motion is not None:
            # Animation at the bottom, as in Drape and Soft Prop.
            animation_box = self.section(controls, "Animation", True)
            animation_box.addWidget(animation_note)
            animation_box.addLayout(mode_form)

        if self.standalone is None:
            self.actions = spin_ui.ActionRow(self.settings, tool="hair", close_tip="Off: minimize after each send so "
                                             "you can inspect the hair in DAZ and send again. On: close once it is written.")
        else:
            config = self.standalone.config
            bridge = bool(config.get("after_export") or config.get("callback"))
            self.actions = spin_ui.ActionRow(
                self.settings, tool="hair", send_text="Send…" if bridge else "Export…",
                close_context="bridge" if bridge else "standalone",
                close_text="Close after send" if bridge else "Close after export", close_default=bridge,
                close_tip="On: close once the files are written. Off: keep the window open to adjust and export again.",
                close_override=config.get("close_after"))
        self.run_button, self.reset_button = self.actions.run_button, self.actions.reset_button
        self.send_button, self.close_after = self.actions.send_button, self.actions.close_after
        controls.addStretch()
        column, self.copyright_footer = spin_ui.control_column(panel, self.actions, 400, version=VERSION, tool_folder=_Path(__file__).resolve().parent.parent)
        splitter.addWidget(column)

        self.viewport = QVTKRenderWindowInteractor(self)
        holder, self.status = spin_ui.viewport_with_status(self.viewport)
        self.status.setText("Analyzing hair cards…")
        splitter.addWidget(holder)
        splitter.setStretchFactor(1, 1)
        # Own position, size and panel width; a first opening borrows the last Spin tool's.
        spin_ui.remember_window(self, self.settings, splitter)
        self.renderer = spin_ui.style_renderer(vtk.vtkRenderer())
        self.viewport.GetRenderWindow().AddRenderer(self.renderer)
        self.collider_actors, self.collider_data = [], []
        for collider in inputs["colliders"]:
            data = polydata(collider["vertices"], np.pad(collider["triangles"], ((0, 0), (0, 1)), constant_values=-1))
            self.collider_data.append(data)
            actor = self.actor(data, spin_ui.AVATAR_COLOR, .55)
            actor.PickableOff()
            self.collider_actors.append(actor)
        self.hair_data = polydata(self.vertices, inputs["faces"])
        normals = vtk.vtkPolyDataNormals(); normals.SetInputData(self.hair_data); normals.SplittingOff()
        normals.ComputeCellNormalsOff(); normals.ConsistencyOff(); normals.Update()
        self.hair_data.GetPointData().SetNormals(normals.GetOutput().GetPointData().GetNormals())
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(self.hair_data)
        self.hair_actor = vtk.vtkActor(); self.hair_actor.SetMapper(mapper)
        self.hair_actor.GetProperty().SetColor(.55, .38, .22)
        self.renderer.AddActor(self.hair_actor)
        self.captured_normals = vtk.vtkFloatArray()
        if self.hair_data.GetPointData().GetNormals() is not None:          # (none with no hair)
            self.captured_normals.DeepCopy(self.hair_data.GetPointData().GetNormals())
        else:
            self.captured_normals.SetNumberOfComponents(3)
        self.interacting = False
        self.pending_frame = None
        self.lines_data = None
        self.lines_actor = None
        self.marker = vtk.vtkActor()
        sphere = vtk.vtkSphereSource(); sphere.SetThetaResolution(24); sphere.SetPhiResolution(16)
        self.marker_source = sphere
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(sphere.GetOutputPort())
        self.marker.SetMapper(mapper); self.marker.GetProperty().SetColor(1, .85, .2)
        self.marker.GetProperty().SetOpacity(.35); self.marker.PickableOff(); self.marker.VisibilityOff()
        self.renderer.AddActor(self.marker)
        self.wind_preview = WindPreview(self.renderer)
        self.water_preview = WaterPreview(self.renderer, start)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.viewport.SetInteractorStyle(style)
        style.AddObserver("LeftButtonPressEvent", self.left_press)
        style.AddObserver("LeftButtonReleaseEvent", self.left_release)
        style.AddObserver("MouseMoveEvent", self.mouse_move)
        # While the camera moves, hold hair updates and apply the newest one afterwards.
        style.AddObserver("StartInteractionEvent", lambda *a: setattr(self, "interacting", True))
        style.AddObserver("EndInteractionEvent", self.end_interaction)
        self.camera_style = style
        self.camera = spin_ui.ViewportCamera(self.renderer, self.render,
                                             lambda: [self.hair_actor] + self.collider_actors)
        self.camera.remember_start()
        self.view.lighting.attach(self.renderer, self.render)
        self.setup_shortcuts()

        self.xray.toggled.connect(self.update_view)
        self.view.reset_view_button.clicked.connect(self.reset_view)
        self.update_view()
        self.pick_card.toggled.connect(lambda on: self.set_pick("card" if on else None))
        self.ring.toggled.connect(lambda on: self.set_pick("ring" if on else None))
        self.pull.toggled.connect(lambda on: self.set_pick("pull" if on else None))
        select_surface.clicked.connect(self.select_surface)
        self.freeze_button.clicked.connect(lambda: self.freeze(True))
        self.pin_button.clicked.connect(lambda: self.pin(True))
        self.unpin_button.clicked.connect(lambda: self.pin(False))
        self.unfreeze_button.clicked.connect(lambda: self.freeze(False))
        clear.clicked.connect(self.clear_selection)
        self.run_button.clicked.connect(self.run_or_pause)
        self.reset_button.clicked.connect(self.reset)
        self.send_button.clicked.connect(self.send)
        self.gravity_panel.changed.connect(self.gravity_control.set)
        self.wind_panel.changed.connect(self.wind_changed)
        self.water_panel.changed.connect(self.water_changed)
        self.gravity_control.set(self.gravity_panel.settings())
        self.wind_changed(self.wind_panel.settings())
        self.water_changed(self.water_panel.settings())
        self.update_buttons()

        self.base_inputs = self.captured_inputs = inputs
        # Display looks from the capture's session (avatar with UVs, surfaces' colors and textures).
        self.textures = hair_looks.Textures()
        self.avatar_entry, self.hair_entries = hair_looks.session_looks(inputs["live_session"]) \
            if inputs.get("live_session") else (None, [])
        self.avatar_look = None
        if self.avatar_entry is not None:
            primary = primary_index(inputs["colliders"])
            self.avatar_look = hair_looks.avatar_look(self.renderer, self.avatar_entry,
                                                      len(inputs["colliders"][primary]["vertices"]), self.textures)
        self.hair_look = None
        self.setup_eye_guard(inputs)
        model = self.display.model()
        model.item(1).setEnabled(self.avatar_look is not None or any(e.get("looks") for e in self.hair_entries))
        model.item(2).setEnabled(model.item(1).isEnabled())
        wanted = self.display.findData(self.settings.value("view/display", FLAT))
        self.display.setCurrentIndex(wanted if wanted >= 0 and model.item(wanted).isEnabled() else 0)
        self.display.currentIndexChanged.connect(self.display_changed)
        self.smooth_shading.toggled.connect(self.smooth_changed)
        self.smooth_changed(self.smooth_shading.isChecked())
        mapper = vtk.vtkPolyDataMapper(); mapper.SetScalarModeToUseCellData(); mapper.SetColorModeToDirectScalars()
        self.grow_overlay = vtk.vtkActor(); self.grow_overlay.SetMapper(mapper); self.grow_overlay.PickableOff()
        # Solid and unlit (some painted polygons were hard to see depending on the light).
        self.grow_overlay.GetProperty().LightingOff(); self.grow_overlay.GetProperty().EdgeVisibilityOn()
        self.grow_overlay.GetProperty().SetEdgeColor(.45, .05, .35)
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.renderer.AddActor(self.grow_overlay)
        self.refresh_grow_overlay()
        self.scalp_cap.toggled.connect(self.cap_changed)
        self.start_build()

    def start_build(self):
        self.builder = BuildWorker(self.base_inputs, self, cap=self.scalp_cap.isChecked(),
                                   cap_density=self.cap_density.value() / 100,
                                   captured=getattr(self, "captured_model", None),
                                   area_models=self.__dict__.setdefault("area_models", {}))
        self.builder.progress.connect(self.status.setText)
        self.builder.done.connect(self.model_ready)
        self.update_buttons()
        self.builder.start()

    def cap_density_changed(self):
        value = self.cap_density.value()
        if value != self.settings.value("scalp_cap_density", 100.0, type=float):
            self.settings.setValue("scalp_cap_density", value)
            if self.scalp_cap.isChecked():
                self.cap_changed(True)

    def cap_changed(self, on):
        if self.builder is not None:
            self.builder.done.disconnect(self.model_ready)
            self.builder.done.connect(lambda *_: self.start_build())
            return
        self.reset()
        self.model = None
        if self.lines_actor is not None:
            self.renderer.RemoveActor(self.lines_actor)
            self.lines_actor = None
        self.start_build()

    # ----- construction helpers ---------------------------------------------------
    def section(self, parent_layout, title, expanded):
        return self.sections.add(title, expanded)

    @staticmethod
    def number(low, high, value, step, suffix=""):
        widget = QDoubleSpinBox(); widget.setRange(low, high); widget.setDecimals(2)
        widget.setSingleStep(step); widget.setValue(value); widget.setSuffix(suffix)
        return widget

    def actor(self, data, color, opacity):
        normals = vtk.vtkPolyDataNormals(); normals.SetInputData(data); normals.SplittingOff()
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(normals.GetOutputPort())
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color); actor.GetProperty().SetOpacity(opacity)
        self.renderer.AddActor(actor)
        return actor

    def wind_changed(self, settings):
        settings = dict(settings)
        settings['track_avatar'] = self.motion is not None and self.animation() and self.wind_track.isChecked()
        r,t = self.wind_pose
        settings['reference_rotation'] = r.tolist()
        settings['reference_translation'] = t.tolist()
        self.wind_control.set(settings)
        self.wind_preview.update(settings)
        self.render()

    def wind_tracking_changed(self, *_):
        self.settings.setValue("wind/track_avatar", self.wind_track.isChecked())
        self.wind_panel.center=self.wind_center()
        self.wind_aim_rotation=self.wind_target_pose()[0].copy()
        self.wind_changed(self.wind_panel.settings())

    def wind_target_pose(self):
        if self.motion is None:
            return np.eye(3),np.zeros(3)
        if self.animation():
            return self.wind_pose
        if self.model is None:
            return np.eye(3),np.zeros(3)
        if self.wind_head is None:
            from hair_animation import HeadMotion
            roots=self.model.particles[self.model.particle_start[self.model.root_kind==0]]
            self.wind_head=HeadMotion(self.motion,roots if len(roots) else self.model.particles,
                                      collider_index=primary_index(self.inputs['colliders']))
        return self.wind_head.transform(self.motion.duration)

    def wind_center(self):
        # The scalp pivot is fixed in head space. Tugging the free lengths must
        # never move the target, source, or direction of the wind.
        r,t=self.wind_target_pose()
        return r @ self.wind_start_center + t

    def recenter_wind(self):
        from wind_settings import wind_direction
        panel=self.wind_panel
        source=np.array([panel.fields[key].value() for key in ('x','y','z')])
        rotation,_=self.wind_target_pose()
        turn=rotation @ self.wind_aim_rotation.T
        center=self.wind_center()
        position=center+turn @ (source-panel.center)
        direction=turn @ wind_direction(panel.fields['heading'].value(),panel.fields['elevation'].value())
        panel.center=center
        panel.orbit_distance=None
        self.wind_aim_rotation=rotation.copy()
        def change():
            for key,value in zip(('x','y','z'),position):panel.fields[key].setValue(float(value))
            if panel.follow_box.isChecked():
                panel.aim_at_center()
            else:
                panel.set_direction(math.degrees(math.atan2(direction[0],-direction[2]))%360,
                                    math.degrees(math.atan2(direction[1],math.hypot(direction[0],direction[2]))))
        panel.batch(change)

    def water_changed(self, settings):
        self.water_control.set(settings)
        self.water_preview.update(settings)
        self.render()

    # ----- model and view ---------------------------------------------------------
    @Slot(object, str)
    def model_ready(self, model, error):
        inputs = self.builder.result_inputs if self.builder is not None else self.inputs
        self.builder = None
        if model is not None and not self.base_inputs.get("grown") and self.base_inputs is self.captured_inputs:
            self.captured_model = model           # kept, so growing only analyzes the grown strips
        if model is not None and inputs is not self.inputs:
            self.use_inputs(inputs)
        if model is None:
            self.info.setText("Could not analyze the hair.")
            self.status.set_state("failed")
            self.status.setText("Could not analyze the hair: " + error)
            self.update_buttons()                      # unlock the panel
            QMessageBox.critical(self, APP_NAME, error)
            return
        if model.strand_count:
            self.status.setText("Ready — press Run simulation" + spin_ui.key_hint("hair", "pause_resume") + ".")
        else:
            self.status.setText("No hair yet: open Grow hair, paint an area on the avatar and grow it.")
        self.model = model
        self.wind_head = None
        if not self.wind_initialized and model.strand_count:
            roots=model.particles[model.particle_start[model.root_kind==0]]
            self.wind_start_center=(roots if len(roots) else model.particles[model.particle_start]).mean(0)
            self.recenter_wind()
            self.wind_initialized=True
        self.frozen = {s for s in self.frozen if s < model.strand_count}
        self.pinned = {s for s in self.pinned if s < model.strand_count}
        self.owner = np.repeat(np.arange(model.strand_count), model.particle_count)
        self.held = np.zeros(len(model.particles), bool)
        roots = model.particle_start[model.root_kind == 0]
        self.held[roots] = True
        self.held[np.minimum(roots + 1, len(model.particles) - 1)] = True
        self.lines_data = strand_lines(model.particles, model.particle_start, model.particle_count)
        self.line_colors = vtk.vtkUnsignedCharArray(); self.line_colors.SetNumberOfComponents(3)
        self.line_colors.SetNumberOfTuples(model.strand_count)
        self.lines_data.GetCellData().SetScalars(self.line_colors)
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(self.lines_data)
        mapper.SetScalarModeToUseCellData(); mapper.SetColorModeToDirectScalars()
        self.lines_actor = vtk.vtkActor(); self.lines_actor.SetMapper(mapper)
        self.lines_actor.GetProperty().SetLineWidth(2); self.lines_actor.GetProperty().LightingOff()
        self.renderer.AddActor(self.lines_actor)
        kinds = np.bincount(model.root_kind, minlength=3)
        self.info.setText(
            f"{self.inputs['hair_label']}: {model.strand_count:,} strands ({len(model.particles):,} points). "
            f"Rooted on the scalp: {kinds[0]:,} · hanging from other hair: {kinds[1]:,} · free: {kinds[2]:,}."
            + (f"\n{model.unsupported_cards} pieces are not hair cards and stay as captured."
               if model.unsupported_cards else "")
            + "\nStrand colors: blue scalp-rooted · green hanging from hair · red free · purple frozen · yellow selected.")
        self.refresh_colors()
        self.rebuild_looks()                      # the hair (and grown hair) in the Display choice
        self.update_buttons()

    def use_inputs(self, inputs):
        """Swap the hair mesh (with or without the scalp cap). Freezes keep original strands."""
        self.inputs = inputs
        self.vertices = inputs["vertices"].copy()
        self.hair_data = polydata(self.vertices, inputs["faces"])
        normals = vtk.vtkPolyDataNormals(); normals.SetInputData(self.hair_data); normals.SplittingOff()
        normals.ComputeCellNormalsOff(); normals.ConsistencyOff(); normals.Update()
        self.hair_data.GetPointData().SetNormals(normals.GetOutput().GetPointData().GetNormals())
        self.captured_normals = vtk.vtkFloatArray()
        if self.hair_data.GetPointData().GetNormals() is not None:          # (none with no hair)
            self.captured_normals.DeepCopy(self.hair_data.GetPointData().GetNormals())
        else:
            self.captured_normals.SetNumberOfComponents(3)
        self.hair_actor.GetMapper().SetInputData(self.hair_data)
        self.selected = set()

    def refresh_colors(self, pulled=()):
        if self.model is None:
            return
        colors = np.empty((self.model.strand_count, 3), np.uint8)
        kind = self.model.root_kind
        colors[kind == 0] = COLORS["scalp"]; colors[kind == 1] = COLORS["parent"]; colors[kind == 2] = COLORS["free"]
        if self.pinned:
            colors[list(self.pinned)] = COLORS["pinned"]
        if self.frozen:
            colors[list(self.frozen)] = COLORS["frozen"]
        if self.selected:
            colors[list(self.selected)] = COLORS["selected"]
        if len(pulled):
            colors[list(pulled)] = COLORS["pulled"]
        for i, c in enumerate(colors):
            self.line_colors.SetTypedTuple(i, c)
        self.line_colors.Modified()
        free = len(unanchored(self.model, self.frozen))
        self.selection_note.setText(f"{len(self.selected):,} selected · {len(self.frozen):,} frozen · "
                                    f"{len(self.pinned):,} pinned · "
                                    f"{free:,} free strands" + (" — freeze or accept them" if free else ""))
        self.render()

    # ----- eye guard -------------------------------------------------
    def setup_eye_guard(self, inputs):
        primary = primary_index(inputs["colliders"])
        avatar = inputs["colliders"][primary]["vertices"]
        from eye_guard import EYE_FRONT
        self.eye_guard_fitted = fit_eye_guard(avatar, eye_vertices(self.avatar_entry, len(avatar)),
                                              eye_vertices(self.avatar_entry, len(avatar), EYE_FRONT))
        source = vtk.vtkCylinderSource(); source.SetResolution(48); source.CappingOn()   # axis y, radius .5, height 1
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(source.GetOutputPort())
        self.eye_guard_actor = vtk.vtkActor(); self.eye_guard_actor.SetMapper(mapper)
        self.eye_guard_actor.PickableOff(); self.eye_guard_actor.VisibilityOff()
        self.eye_guard_actor.GetProperty().SetColor(.45, .75, 1.)
        self.eye_guard_actor.GetProperty().SetOpacity(.25)
        self.renderer.AddActor(self.eye_guard_actor)
        found = self.eye_guard_fitted is not None
        for widget in (self.eye_guard_on, self.eye_width, self.eye_height, self.eye_depth, self.eye_raise,
                       self.eye_guard_show, self.eye_fit):
            widget.setEnabled(found)
        self.eye_note.setText("" if found else "No eye surfaces were found on the avatar (Cornea, Sclera, Irises…), "
                              "so the eye guard is not available for this capture.")
        self.eye_note.setVisible(not found)
        if found:
            self.fit_eye_guard()
        self.eye_guard_on.toggled.connect(self.eye_guard_changed)
        self.eye_guard_show.toggled.connect(self.eye_guard_changed)
        for widget in (self.eye_width, self.eye_height, self.eye_depth, self.eye_raise):
            widget.valueChanged.connect(self.eye_guard_changed)
        self.eye_fit.clicked.connect(self.fit_eye_guard)

    def fit_eye_guard(self):
        guard = self.eye_guard_fitted
        if guard is None:
            return
        boxes = (self.eye_width, self.eye_height, self.eye_depth, self.eye_raise)
        blocked = [box.blockSignals(True) for box in boxes]
        for box, value in zip(boxes, (*guard.sizes, 0.)):
            box.setValue(float(value))
        for box, state in zip(boxes, blocked):
            box.blockSignals(state)
        self.eye_guard_changed()

    def eye_guard(self):
        """The oval as set in the panel (capture pose), or None when off or not available."""
        if self.eye_guard_fitted is None or not self.eye_guard_on.isChecked():
            return None
        return self.eye_guard_fitted.copy(sizes=(self.eye_width.value(), self.eye_height.value(),
                                                 self.eye_depth.value()), raise_cm=self.eye_raise.value())

    def eye_guard_changed(self, *_):
        self.eye_guard_control.set(self.eye_guard())
        self.update_eye_guard_actor()
        self.render()

    def update_eye_guard_actor(self):
        """The see-through tube, carried with the head (the frame's head pose)."""
        actor = getattr(self, "eye_guard_actor", None)
        if actor is None:
            return
        guard = self.eye_guard()
        actor.SetVisibility(guard is not None and self.eye_guard_show.isChecked())
        if guard is None:
            return
        turn, shift = getattr(self, "wind_pose", (np.eye(3), np.zeros(3)))
        axes = guard.axes @ turn.T                      # rows: across, up, forward in the current pose
        # Cylinder source: x and z across its radius 0.5, y along its height 1 (centered). x -> across (width),
        # y -> forward (depth), z -> up (height); its middle half the depth in front of the eyes.
        from eye_guard import BACK
        width, height, depth = (float(v) for v in guard.sizes)
        linear = np.stack([axes[0] * width, axes[2] * (depth + BACK), axes[1] * height], axis=1)
        center = turn @ (guard.placed_center() + guard.axes[2] * (depth - BACK) * .5) + shift
        matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                matrix.SetElement(i, j, float(linear[i, j]))
            matrix.SetElement(i, 3, float(center[i]))
        actor.SetUserMatrix(matrix)

    def display_mode(self):
        return self.display.currentData() if hasattr(self, "display") else FLAT

    def display_changed(self, *_):
        self.settings.setValue("view/display", self.display.currentData())
        self.rebuild_looks()

    def rebuild_looks(self):
        """Colored or textured copies of the avatar and hair for the Display choice (none for Flat)."""
        if not hasattr(self, "avatar_look"):
            return
        mode = self.display_mode()
        if self.avatar_look is not None:
            self.avatar_look.clear()
        if self.hair_look is not None:
            self.hair_look.clear()
            self.hair_look = None
        if mode != FLAT:
            if self.avatar_look is not None:
                self.avatar_look.build(mode)
            self.hair_look = hair_looks.hair_look(self.renderer, self.inputs, self.hair_entries, self.textures)
            if self.hair_look is not None:
                self.hair_look.build(mode)
            self.update_looks()
        self.update_view()

    def smooth_changed(self, on):
        """Smooth or flat shading for the avatar and hair in every Display mode."""
        self.settings.setValue("view/smooth", bool(on))
        primary = primary_index(self.inputs["colliders"])
        for actor in (self.hair_actor, self.collider_actors[primary]):
            if on:
                actor.GetProperty().SetInterpolationToGouraud()
            else:
                actor.GetProperty().SetInterpolationToFlat()
        for look in (getattr(self, "avatar_look", None), getattr(self, "hair_look", None)):
            if look is not None:
                look.set_smooth(on)
        self.render()

    def update_looks(self):
        """The looks follow the moving avatar and the simulated hair, with their smooth normals."""
        from hair_grow import vertex_normals
        if getattr(self, "avatar_look", None) is not None and self.avatar_look.actors:
            index = primary_index(self.inputs["colliders"])
            points = vtk_to_numpy(self.collider_data[index].GetPoints().GetData())
            self.avatar_look.smooth = self.smooth_shading.isChecked()
            self.avatar_look.update(points, vertex_normals(np.asarray(points, float),
                                                           np.asarray(self.inputs["colliders"][index]["triangles"])))
        if getattr(self, "hair_look", None) is not None and self.hair_look.actors:
            normals = self.hair_data.GetPointData().GetNormals()
            self.hair_look.smooth = self.smooth_shading.isChecked()
            self.hair_look.update(self.vertices, vtk_to_numpy(normals) if normals is not None else None)

    def update_view(self):
        if not hasattr(self, "hair_actor"):
            return
        primary = primary_index(self.inputs["colliders"])
        avatar_actors = []
        for index, (actor, collider) in enumerate(zip(self.collider_actors, self.inputs["colliders"])):
            if index == primary or collider.get("role") == "avatar_attachment":
                avatar_actors.append(actor)
            else:
                actor.SetVisibility(self.show_collisions.isChecked())
                actor.GetProperty().SetOpacity(self.collision_opacity.value() / 100)
        self.view.apply([self.hair_actor], avatar_actors)
        xray = self.xray.isChecked()
        self.hair_actor.SetVisibility(self.show_hair.isChecked() and not xray)
        # Display looks stand in for the plain avatar and hair (same show/hide and opacity).
        primary_actor = self.collider_actors[primary]
        for look, original, shown, opacity in (
                (getattr(self, "avatar_look", None), primary_actor, self.show_avatar.isChecked(),
                 self.avatar_opacity.value() / 100),
                (getattr(self, "hair_look", None), self.hair_actor, self.show_hair.isChecked() and not xray,
                 self.opacity.value() / 100)):
            if look is not None and look.actors:
                original.SetVisibility(False)
                for actor, _ in look.actors:
                    actor.SetVisibility(shown)
                    actor.GetProperty().SetOpacity(opacity)
        if self.lines_actor is not None:
            self.lines_actor.SetVisibility(xray or bool(self.selected) or self.pick_mode in ("card", "ring"))
        self.render()

    def render(self):
        if hasattr(self, "viewport"):
            self.viewport.GetRenderWindow().Render()

    # ----- picking, selection and pulling -----------------------------------------
    def set_pick(self, mode):
        self.pick_mode = mode
        for button, name in ((self.pick_card, "card"), (self.ring, "ring"), (self.pull, "pull"),
                             (self.grow_panel.paint, "grow")):
            button.blockSignals(True); button.setChecked(mode == name); button.blockSignals(False)
        self.grow_painting = False
        self.marker.SetVisibility(False)
        self.release_pull()
        self.update_view()

    def pick_point(self):
        x, y = self.viewport.GetEventPosition()
        picker = vtk.vtkCellPicker(); picker.SetTolerance(.004); picker.PickFromListOn()
        use_lines = self.xray.isChecked() or not self.hair_actor.GetVisibility()
        target = self.lines_actor if use_lines else self.hair_actor
        # Surface colors / Textures hide the plain hair actor behind the hair look: pick on the
        # look's actors then; the pick point is all a pull needs.
        look = getattr(self, "hair_look", None)
        targets = [target]
        if not self.xray.isChecked() and look is not None and look.actors and look.actors[0][0].GetVisibility():
            targets = [actor for actor, _ in look.actors]
        for each in targets:
            if each is not None:
                picker.AddPickList(each)
        if not picker.Pick(x, y, 0, self.renderer):
            return None, None, None
        return picker, picker.GetActor() or target, np.array(picker.GetPickPosition())

    def grow_pick(self):
        """The point under the mouse on the avatar, or None."""
        x, y = self.viewport.GetEventPosition()
        picker = vtk.vtkCellPicker(); picker.SetTolerance(.002); picker.PickFromListOn()
        avatar = self.collider_actors[primary_index(self.inputs["colliders"])]
        look = getattr(self, "avatar_look", None)
        targets = [actor for actor, _ in look.actors] if look is not None and look.actors else [avatar]
        for target in targets:
            picker.AddPickList(target)
            # Collider actors are not pickable (hair picking must not hit them): allow it for this pick only.
            target.PickableOn()
        try:
            hit = picker.Pick(x, y, 0, self.renderer)
        finally:
            for target in targets:
                target.PickableOff()
        return np.array(picker.GetPickPosition()) if hit else None

    def grow_paint(self):
        point = self.grow_pick()
        if point is not None:
            removing = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)
            self.grow_panel.paint_at(point, removing)

    def refresh_grow_overlay(self):
        if getattr(self, "grow_overlay", None) is None:
            return
        self.grow_overlay.GetMapper().SetInputData(self.grow_panel.overlay())
        self.render()

    def grow_hair(self):
        """Grow every painted area into the hair mesh and analyze it again (replacing hair grown before)."""
        if self.running() or self.return_worker is not None:
            return
        grown = self.grow_panel.grow()
        if self.scalp_cap.isChecked():
            self.scalp_cap.blockSignals(True); self.scalp_cap.setChecked(False); self.scalp_cap.blockSignals(False)
        self.base_inputs = with_grown(self.captured_inputs, grown) if grown else self.captured_inputs
        count = sum(item["strands"] for item in grown)
        self.grow_panel.summary.setText(f"Grew {count:,} strands: " + ", ".join(
            f"{item['area'].name} {item['strands']:,}" for item in grown) + "." if grown else
                                        "Nothing to grow: paint polygons into an area first.")
        self.cap_changed(False)

    def left_press(self, caller, event):
        if self.pick_mode == "grow":
            self.grow_painting = True
            self.grow_paint()
            return
        if self.pick_mode is None or self.model is None:
            self.camera_style.OnLeftButtonDown()
            return
        picker, target, point = self.pick_point()
        if picker is None:
            return
        if self.pick_mode == "pull":
            self.begin_pull(point)
            return
        # Ctrl adds to the selection, Alt removes from it; a plain click replaces it.
        modifiers = QApplication.keyboardModifiers()   # the VTK interactor is only given Ctrl and Shift, not Alt
        additive = bool(self.viewport.GetControlKey()) or bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        removing = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        if self.pick_mode == "card":
            cell = picker.GetCellId()
            look = getattr(self, "hair_look", None)
            if look is not None and target is not self.lines_actor and target is not self.hair_actor:
                face = look.face_of(target, cell)                  # a hair look's cell -> the hair face
                cell = -1 if face is None else face
            if cell < 0:
                strands = set()
            elif target is self.lines_actor:
                strands = {cell}
            else:
                corner = self.inputs["faces"][cell][0]
                strand = self.model.strand_of_vertex[corner] if corner < len(self.model.rest) else -1
                strands = {int(strand)} if strand >= 0 else set()       # scalp cap cards are not strands
            self.selected = self.combine(strands, additive, removing)
        else:
            radius = self.ring_radius.value()
            near = np.linalg.norm(self.current_particles() - point, axis=1) <= radius
            strands = set(np.unique(self.owner[near]).tolist())
            self.marker_source.SetCenter(*point); self.marker_source.SetRadius(radius); self.marker.VisibilityOn()
            self.selected = self.combine(strands, additive, removing)
        self.refresh_colors()
        self.update_view()

    def combine(self, strands, additive, removing):
        if removing:
            return self.selected - strands
        return (self.selected | strands) if additive else strands

    def begin_pull(self, point):
        if self.animation() or self.return_worker is not None:
            return
        paused = not self.running()
        if self.worker is None:
            self.start_simulation()
            self.worker.request("pause")
        particles = self.current_particles()
        radius = self.pull_radius.value()
        distance = np.linalg.norm(particles - point, axis=1)
        frozen = np.isin(self.owner, list(self.frozen)) if self.frozen else np.zeros(len(particles), bool)
        near = np.flatnonzero((distance <= radius) & ~self.held & ~frozen)
        if not len(near):
            self.status.setText("Nothing to pull there: that hair is frozen or held at the scalp.")
            return
        weights = np.maximum(.05, 1 - distance[near] / radius) * (self.pull_strength.value() / 100)
        self.renderer.SetWorldPoint(*point, 1.0)
        self.renderer.WorldToDisplay()
        self.drag = dict(indices=near, weights=weights, origin=point, positions=particles[near].copy(),
                         depth=self.renderer.GetDisplayPoint()[2], paused=paused,
                         hold=self.pull_mode.currentData() == "hold")
        self.marker_source.SetCenter(*point); self.marker_source.SetRadius(radius); self.marker.VisibilityOn()
        self.refresh_colors(pulled=set(np.unique(self.owner[near]).tolist()))
        if not paused:
            self.grab_control.set(near, self.drag["positions"], weights)
        self.status.setText("Pulling paused result — time stays paused." if paused else "Pulling — let go to release.")
        self.update_buttons()

    def mouse_move(self, caller, event):
        if self.pick_mode == "grow" and getattr(self, "grow_painting", False):
            self.grow_paint()
            return
        if self.drag is None:
            self.camera_style.OnMouseMove()
            return
        x, y = self.viewport.GetEventPosition()
        self.renderer.SetDisplayPoint(x, y, self.drag["depth"])
        self.renderer.DisplayToWorld()
        world = np.array(self.renderer.GetWorldPoint())
        if abs(world[3]) < 1e-12:
            return
        delta = world[:3] / world[3] - self.drag["origin"]
        targets = self.drag["positions"] + delta
        if np.isfinite(targets).all():
            if self.drag["paused"]:
                self.result_ready = False
                self.worker.edit("pull", (self.drag["indices"], targets.copy(), self.drag["weights"]))
            else:
                self.grab_control.set(self.drag["indices"], targets, self.drag["weights"])
            self.marker_source.SetCenter(*(self.drag["origin"] + delta)); self.render()

    def left_release(self, caller, event):
        if self.pick_mode == "grow":
            self.grow_painting = False
            self.grow_panel.finish_paint()
            return
        if self.drag is None:
            self.camera_style.OnLeftButtonUp()
            return
        self.release_pull()

    def release_pull(self):
        if self.drag is None:
            return
        if self.drag.get("hold") and self.worker is not None:
            self.result_ready = False
            self.worker.edit("hold", self.drag["indices"].copy())
        self.drag = None
        self.grab_control.clear()
        self.marker.VisibilityOff()
        self.refresh_colors()
        self.update_buttons()
        if self.running():
            self.status.setText("Running — press Pause" + spin_ui.key_hint("hair", "pause_resume", " or ", "")
                                + " to stop.")

    def release_pull_holds(self):
        if self.worker is not None:
            self.worker.edit("release")
        self.status.setText("Pull holds released. Paused hair stays in place until you resume.")

    def current_particles(self):
        return self.particles if self.particles is not None else self.model.particles

    def select_surface(self):
        if self.model is None:
            return
        index = self.surface.currentIndex()
        self.selected = set(np.flatnonzero(self.model.strand_surface == index).tolist())
        if not self.selected:
            self.status.setText(f"Surface {self.surface.currentText()} has no simulated strands "
                                "(it stays as captured).")
        self.refresh_colors(); self.update_view()

    def clear_selection(self):
        self.selected = set(); self.marker.VisibilityOff(); self.refresh_colors(); self.update_view()

    def pin(self, on):
        if self.worker is not None:
            return
        if on:
            self.pinned |= self.selected - self.frozen
        else:
            self.pinned -= self.selected
        self.refresh_colors()

    def freeze(self, on):
        if self.worker is not None:
            return
        if on:
            self.frozen |= self.selected
        else:
            self.frozen -= self.selected
        self.refresh_colors()

    # ----- simulation -------------------------------------------------------------
    def settings_now(self):
        return dict(duration=self.duration.value(), style=self.style.value() / 100, bend=self.bend.value() / 100,
                    damping=self.damping.value(), radius=self.thickness.value() * .05, friction=self.friction.value(),
                    style_length=self.style_reach.value(), shelter=1 - self.shelter.value() / 100,
                    wind_response=0.006 * self.wind_catch.value() / 100, hairspray=self.hairspray.value() / 100,
                    hair_collision=self.hair_collide.isChecked(), hair_spacing=self.hair_spacing.value(),
                    trapped_distance=self.trapped_distance.value() * .1,
                    max_move_cm=self.max_move.value() if self.motion is not None else 0.,
                    root_lift=self.root_lift.value() / 100, stiffness=self.stiffness.value() / 100,
                    style_rest=self.style_rest.value() / 100, weightless_lift=self.lift.value() / 100,
                    momentum=self.momentum.value() / 100,
                    clumping=self.clumping.value() / 100, head_drag=self.head_drag.value(),
                    loop=self.motion is not None and self.loop_animation.isChecked(),
                    warm_loops=self.warm_loops.value(), loose=self.loose_strands())

    def loose_strands(self):
        """Grown strands -> their area's stiffness share: they fall under full gravity."""
        loose = {}
        if self.model is None:
            return loose
        for part in self.inputs.get("grown") or []:
            owners = self.model.strand_of_vertex[part["start"]:part["end"]]
            for strand in np.unique(owners[owners >= 0]):
                # [stiffness share, root lift] (root lift per area)
                loose[int(strand)] = [part.get("stiffness", 0.), part.get("root_lift", 0.)]
        return loose

    def settle_frames(self):
        return int(round(self.duration.value() * 60))

    def running(self):
        return self.worker is not None and not self.worker.is_paused()

    def animation(self):
        return self.motion is not None and self.output_mode.currentData() == "animation"

    # ----- presets ----------------------------------------------------------------------------
    PRESET_FIELDS = ("style_rest", "lift", "stiffness", "hair_spacing", "damping", "friction", "clumping", "head_drag",
                     "momentum", "max_move", "hair_collide")
    BUILT_IN_PRESETS = {
        "Static": dict(style_rest=100, lift=60, stiffness=90, hair_spacing=.6, damping=2, friction=0, clumping=0,
                       head_drag=0, momentum=50, max_move=.75, hair_collide=True),
        # Moves together and stays near its style.
        "Animation": dict(style_rest=100, lift=60, stiffness=90, hair_spacing=.6, damping=.5, friction=0, clumping=50,
                          head_drag=6, momentum=0, max_move=.75, hair_collide=True),
    }
    PRESET_FILE = Path(__file__).resolve().parent.parent / "hair_presets.json"

    def saved_presets(self):
        import json
        try:
            data = json.loads(self.PRESET_FILE.read_text(encoding="utf-8"))
            return {str(name): dict(values) for name, values in data.items() if isinstance(values, dict)}
        except (OSError, ValueError):
            return {}

    def setup_presets(self):
        self.applying_preset = False
        self.refill_presets("Animation" if self.animation() else "Static")
        self.apply_preset(self.preset.currentText())
        self.preset.activated.connect(lambda *_: self.apply_preset(self.preset.currentText()))
        self.save_preset_button.clicked.connect(self.save_preset)
        self.delete_preset_button.clicked.connect(self.delete_preset)
        for name in self.PRESET_FIELDS:
            widget = getattr(self, name)
            signal = widget.toggled if isinstance(widget, QCheckBox) else widget.valueChanged
            signal.connect(self.preset_edited)

    def refill_presets(self, current):
        names = list(self.BUILT_IN_PRESETS) + sorted(self.saved_presets())
        if current == "Custom":
            names.append("Custom")
        self.preset.blockSignals(True)
        self.preset.clear(); self.preset.addItems(names)
        self.preset.setCurrentText(current if current in names else names[0])
        self.preset.blockSignals(False)
        self.delete_preset_button.setEnabled(self.preset.currentText() not in (*self.BUILT_IN_PRESETS, "Custom"))

    def preset_values(self):
        return {name: (getattr(self, name).isChecked() if isinstance(getattr(self, name), QCheckBox)
                       else getattr(self, name).value()) for name in self.PRESET_FIELDS}

    def apply_preset(self, name):
        values = self.BUILT_IN_PRESETS.get(name) or self.saved_presets().get(name)
        if values is None:
            return
        self.applying_preset = True
        try:
            for field, value in values.items():
                widget = getattr(self, field, None)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif widget is not None and field in self.PRESET_FIELDS:
                    widget.setValue(float(value))
        finally:
            self.applying_preset = False
        self.refill_presets(name)

    def preset_edited(self, *_):
        if not getattr(self, "applying_preset", True) and self.preset.currentText() != "Custom":
            self.refill_presets("Custom")

    def save_preset(self):
        import json
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, APP_NAME, "Preset name:")
        name = name.strip()
        if not ok or not name:
            return
        if name in self.BUILT_IN_PRESETS or name == "Custom":
            self.status.setText(f'"{name}" is a built-in name; choose another.')
            return
        presets = self.saved_presets()
        presets[name] = self.preset_values()
        try:
            self.PRESET_FILE.write_text(json.dumps(presets, indent=2), encoding="utf-8")
        except OSError as error:
            self.status.setText(f"Could not save the preset: {error}")
            return
        self.refill_presets(name)
        self.status.setText(f'Preset "{name}" saved.')

    def delete_preset(self):
        import json
        name = self.preset.currentText()
        presets = self.saved_presets()
        if name not in presets:
            return
        del presets[name]
        try:
            self.PRESET_FILE.write_text(json.dumps(presets, indent=2), encoding="utf-8")
        except OSError as error:
            self.status.setText(f"Could not delete the preset: {error}")
            return
        self.refill_presets("Custom")
        self.status.setText(f'Preset "{name}" deleted; the settings are kept as Custom.')

    def mode_changed(self):
        # A different return needs a fresh run (an animation records from the start frame).
        if getattr(self, "preset", None) is not None and self.preset.currentText() in self.BUILT_IN_PRESETS:
            self.apply_preset("Animation" if self.animation() else "Static")
        if self.worker is not None:
            self.reset()
        self.wind_track.setEnabled(self.motion is not None and self.animation())
        if self.wind_initialized:
            self.recenter_wind()
        else:
            self.wind_changed(self.wind_panel.settings())
        self.scalp_cap.setEnabled(len(self.inputs["ranges"]) == 1 and not self.animation() and self.return_worker is None)
        if self.animation() and self.scalp_cap.isChecked():
            self.scalp_cap.setChecked(False)
        self.update_buttons()

    def start_simulation(self, hold=False):
        self.result_ready = False
        controls = dict(gravity=self.gravity_control, wind=self.wind_control, water=self.water_control,
                        grab=self.grab_control, eye_guard=self.eye_guard_control)
        self.recorded = None
        if hasattr(self, "playback"):
            self.playback.clear()
        if self.animation():
            from hair_animation import AnimationFrames
            from project_settings import session_directory
            m = self.motion
            self.recorded = AnimationFrames(m.start_frame, m.end_frame, m.fps, len(self.base_inputs["vertices"]),
                                            folder=str(session_directory()))
        colliders = self.inputs["colliders"]
        settings = self.settings_now()
        settings["head_collider"] = primary_index(colliders)
        settings["pinned_ends"] = sorted(self.pinned - self.frozen)
        self.worker = SimulationWorker(self.model, colliders, set(self.frozen), settings,
                                       controls, self, faces=self.base_inputs["faces"], cap=self.inputs.get("cap"),
                                       motion=self.motion, frames=self.recorded,
                                       diagnostic_path=(Path(self.inputs['live_session']).with_suffix('.simulation.json')
                                                        if self.inputs.get('live_session') else None))
        if hold:
            self.worker.request("hold")
        self.worker.frame.connect(self.on_frame, Qt.ConnectionType.QueuedConnection)
        self.worker.paused.connect(self.on_paused, Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self.on_failed, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self.on_finished, Qt.ConnectionType.QueuedConnection)
        self.status.set_state("running")
        self.status.setText("Preparing the GPU…")
        self.worker.start()

    def run_or_pause(self):
        if self.model is None:
            return
        if self.worker is None:
            self.start_simulation()
        elif self.running():
            self.release_pull()
            self.worker.request("pause")
        elif (self.animation() and self.recorded is not None
              and not self.recorded.complete()):
            # Paused part-way through recording: carry on from where it stopped.
            self.result_ready = False
            self.worker.request("resume")
            self.status.set_state("running")
            self.status.setText("Recording the movement…")
        else:
            # After the initial settle, Resume keeps running until the user pauses.
            self.result_ready = False
            self.worker.request("hold")
            self.status.set_state("running")
            self.status.setText("Running — press Pause" + spin_ui.key_hint("hair", "pause_resume", " or ", "")
                                + " to stop.")
        self.update_buttons()

    # ----- keyboard shortcuts (shared suite file, as the Drape Tool) ---------------------
    def setup_shortcuts(self):
        """Keys from Spin-Tools/keybindings.json ("common" plus "hair") via the shared spin_ui."""
        toggle = lambda box: (lambda: box.isEnabled() and box.setChecked(not box.isChecked()))
        self.shortcut_actions = {
            "pause_resume": self.space_pressed, "run_simulation": self.space_pressed,
            "reset_simulation": lambda: self.reset_button.isEnabled() and self.reset(),
            "send_to_daz": lambda: self.send_button.isEnabled() and self.send(),
            "freeze_selected": lambda: self.freeze(True), "unfreeze_selected": lambda: self.freeze(False),
            "freeze_unselected": self.freeze_unselected, "unfreeze_all": self.unfreeze_all,
            "clear_selection": self.clear_selection, "grow_selection": lambda: self.grow_selection(1),
            "shrink_selection": lambda: self.grow_selection(-1), "reset_view": self.reset_view,
            "cancel_interaction": lambda: self.set_pick(None), "show_avatar": toggle(self.show_avatar),
            "toggle_xray": toggle(self.xray), "pull_hair": toggle(self.pull),
            "pick_strands": toggle(self.pick_card), "ring_select": toggle(self.ring)}
        self.keys = spin_ui.Shortcuts(
            self, "hair", self.shortcut_actions, camera=self.camera,
            allowed=lambda action: self.model is not None or action in ("show_avatar", "toggle_xray", "reset_view"))
        errors = self.keys.errors
        self.shortcut_note.setText("Keyboard shortcuts come from keybindings.json in the Spin Tools folder (shared by "
                                   "all tools). Press F1 for the list."
                                   if not errors else "Some shortcuts could not load. Check keybindings.json.")
        import keybindings
        self.shortcut_note.setToolTip(str(keybindings.PATH) + "\n" + "\n".join(
            [f"{action}: {key}" for action, keys in self.keys.bindings.items() for key in keys] + errors))
        self.view_layout.addWidget(self.keys.help_button())
        self.view_bar = spin_ui.add_view_toolbar(self.keys, self.status)   # orbit/pan/zoom/reset buttons

    def invoke_shortcut(self, action):
        self.keys.invoke(action)

    def reset_view(self):
        self.camera.reset()

    def freeze_unselected(self):
        if self.worker is None and self.model is not None:
            self.frozen |= set(range(self.model.strand_count)) - self.selected
            self.refresh_colors()

    def unfreeze_all(self):
        if self.worker is None:
            self.frozen = set()
            self.refresh_colors()

    def grow_selection(self, step, reach=1.0):
        """Add (or, step -1, drop from the edge) strands within ``reach`` cm of the selection."""
        if self.model is None or not self.selected:
            return
        from hair_strands import nearest_distance
        points = self.current_particles()
        chosen = np.isin(self.owner, list(self.selected))
        if step > 0:
            near = nearest_distance(points[~chosen], points[chosen]) <= reach
            self.selected |= set(np.unique(self.owner[~chosen][near]).tolist())
        else:
            near = nearest_distance(points[chosen], points[~chosen]) <= reach if (~chosen).any() else np.zeros(chosen.sum(), bool)
            self.selected -= set(np.unique(self.owner[chosen][near]).tolist())
        self.refresh_colors(); self.update_view()

    def space_pressed(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QAbstractSpinBox, QLineEdit, QComboBox)):
            return
        if self.run_button.isEnabled():
            self.run_or_pause()

    def end_interaction(self, *args):
        self.interacting = False
        if self.pending_frame is not None:
            frame, self.pending_frame = self.pending_frame, None
            self.apply_frame(*frame)

    def show_recorded_frame(self, frame):
        """Draw one recorded DAZ frame of the finished animation, with the avatar at that frame."""
        if self.recorded is None or not self.recorded.complete() or self.motion is None:
            return
        sample = frame - self.motion.start_frame
        try:
            vertices = np.asarray(self.recorded.get(sample), dtype=float)
        except (OSError, ValueError):
            return
        faces = np.asarray(self.base_inputs["faces"], dtype=np.int64)
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals = np.zeros_like(vertices)
        for corner in range(3):
            np.add.at(normals, faces[:, corner], cross)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
        avatar = {index: self.motion.at(index, sample / self.motion.fps) for index in self.motion.samples}
        self.apply_frame(vertices, normals, self.particles, 0., avatar)
        self.status.set_frame(frame, self.motion.end_frame)
        self.render()

    def apply_frame(self, vertices, normals, particles, seconds, avatar=None, timeline=None, wind=None, pose=None):
        self.vertices, self.particles = vertices, particles
        if pose is not None:
            self.wind_pose = pose
        for index, points in (avatar or {}).items():
            set_points(self.collider_data[index], points)
        self.timeline = timeline
        if wind is not None:
            self.wind_preview.update(wind)
            if wind.get('track_avatar',False):
                self.wind_panel.display_pose(wind,self.wind_center())
                self.wind_aim_rotation=self.wind_pose[0].copy()
        set_points(self.hair_data, vertices)
        self.update_eye_guard_actor()
        self.hair_data.GetPointData().SetNormals(numpy_to_vtk(np.asarray(normals, np.float32), deep=True))
        self.update_looks()
        if self.lines_data is not None:
            set_points(self.lines_data, particles)
        return seconds

    @Slot()
    def on_frame(self):
        frame = self.worker.take_frame() if self.worker is not None else None
        if frame is None:
            return
        if self.interacting:
            self.pending_frame = frame          # applied when the camera stops
            self.vertices, self.particles = frame[0], frame[2]
            return
        seconds = self.apply_frame(*frame)
        self.update_buttons()
        if self.running() and self.drag is None:
            held = self.worker is not None and self.worker.target >= FOREVER
            timeline = getattr(self, "timeline", None)
            if timeline is not None and self.worker.frames_done >= self.worker.settle and                     timeline < self.motion.duration - 1e-6:
                frame = self.motion.start_frame + timeline * self.motion.fps
                self.status.set_frame(frame, self.motion.end_frame)
                self.status.setText("Playing the movement" + (" — recording frames" if self.animation() else ""),
                                    progress=timeline / max(self.motion.duration, 1e-6))
            elif held:
                self.status.setText(f"Running {seconds:.1f} s" + spin_ui.key_hint("hair", "pause_resume", " — ", " pauses"))
            else:
                settle = self.worker.settle / 60
                self.status.setText(f"Settling {seconds:.1f} s", progress=min(1., seconds / max(settle, 1e-6)))
        self.render()

    @Slot()
    def on_paused(self):
        if self.worker is None or not self.worker.is_paused():
            return
        self.on_frame()                 # readiness always refers to the latest displayed result
        if self.animation() and not (self.recorded is not None and self.recorded.complete()):
            self.result_ready = False
            self.status.set_state("paused")
            self.status.setText("Paused part-way through the movement. Resume to carry on recording, or Reset to start "
                                "again.")
            self.update_buttons()
            return
        self.result_ready = True
        if self.animation() and hasattr(self, "playback") and not self.playback.isEnabled():
            m = self.motion
            self.playback.set_animation(m.start_frame, m.end_frame, m.fps,
                                        looped=bool(getattr(self.worker, "loop_status", "")))
        self.status.set_state("finished" if self.animation() else "paused")
        loop = getattr(self.worker, "loop_status", "").rstrip(".")
        self.status.setText(((f"{loop}. " if loop else "") +
                             "Animation recorded — send it to DAZ, or Reset to record it again." if self.animation() else
                             "Change gravity/wind/water, Resume" + spin_ui.key_hint("hair", "pause_resume") +
                             " to keep simulating, pull while paused to adjust, or send to DAZ."))
        self.update_buttons()

    @Slot(str)
    def on_failed(self, message):
        self.result_ready = False
        self.status.set_state("failed")
        self.status.setText("Simulation stopped: " + message)
        if not self.close_pending:
            QMessageBox.critical(self, APP_NAME, message)

    @Slot()
    def on_finished(self):
        # A run stopped by Reset finishes later; only the current run's end counts.
        if self.worker is not None and self.sender() is not None and self.sender() is not self.worker:
            return
        self.worker = None
        self.drag = None
        self.grab_control.clear()
        self.update_buttons()
        if self.close_pending:
            self.close()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.requestInterruption()

    def reset(self):
        self.release_pull()
        if self.worker is not None:
            self.stop_worker()
            self.worker.wait(5000)
            self.worker = None
        self.result_ready = False
        self.vertices = self.inputs["vertices"].copy()
        self.particles = None
        self.pending_frame = None
        set_points(self.hair_data, self.vertices)
        self.hair_data.GetPointData().SetNormals(self.captured_normals)
        if self.lines_data is not None:
            set_points(self.lines_data, self.model.particles)
        for data, collider in zip(self.collider_data, self.inputs["colliders"]):
            set_points(data, collider["vertices"])          # the avatar back at the start frame
        self.timeline = None
        from hair_relative_wind import wind_at_pose
        _, wind = self.wind_control.snapshot()
        self.wind_pose = (np.eye(3),np.zeros(3))
        self.update_eye_guard_actor()
        if wind.get('track_avatar',False):
            self.wind_panel.display_pose(wind_at_pose(wind,self.wind_pose),self.wind_start_center)
            self.wind_aim_rotation=np.eye(3)
            self.wind_changed(self.wind_panel.settings())
        self.status.set_state("idle")
        self.status.set_frame()
        self.status.setText("Reset to the captured hair. Freezes and settings are kept.")
        self.update_buttons(); self.render()

    def busy(self):
        """Analyzing or growing hair, or sending: the panel is locked."""
        return self.builder is not None or self.return_worker is not None

    def update_buttons(self):
        running = self.running()
        if hasattr(self, "sections"):
            self.sections.set_busy(self.busy())
        self.run_button.setEnabled(self.model is not None and self.model.strand_count > 0 and self.return_worker is None)
        grown = bool(self.base_inputs.get("grown")) if hasattr(self, "base_inputs") else False
        self.scalp_cap.setEnabled(len(self.inputs["ranges"]) == 1 and self.return_worker is None and not self.running()
                                  and not self.animation() and not grown)
        # Growing: still captures only (grown roots follow the head, not the skin, in a timeline).
        can_grow = self.worker is None and self.return_worker is None and self.motion is None
        self.grow_panel.grow_button.setEnabled(can_grow)
        self.grow_panel.grow_button.setToolTip("" if self.motion is None else
                                               "Growing hair is for still captures.")
        self.cap_density.setEnabled(len(self.inputs["ranges"]) == 1 and self.return_worker is None and not self.running())
        self.actions.show_run_state(self.worker is not None, not running)
        pending_edits = self.worker is not None and self.worker.edits_pending()
        self.send_button.setEnabled(self.result_ready and not running and self.drag is None and not pending_edits and self.return_worker is None)
        for widget in (self.freeze_button, self.unfreeze_button, self.pin_button, self.unpin_button):
            widget.setEnabled(self.worker is None and self.model is not None)
        self.trapped_distance.setEnabled(self.worker is None and self.model is not None)
        self.max_move.setEnabled(self.worker is None and self.model is not None)
        for widget in (self.loop_animation, self.warm_loops):
            widget.setEnabled(self.worker is None and self.model is not None)
        self.reset_button.setEnabled(self.return_worker is None and self.model is not None)
        # Sending an animation records one uninterrupted run (as the Drape Tool): no live pulling,
        # and no resuming once paused or finished; Reset to record again.
        self.pull.setEnabled(self.model is not None and self.return_worker is None
                             and not self.animation())
        self.pull_mode.setEnabled(self.drag is None and not self.animation())
        self.release_holds.setEnabled(self.worker is not None and self.return_worker is None and self.drag is None
                                      and not self.animation())
        recording_done = self.recorded is not None and self.recorded.complete()
        if self.animation() and self.worker is not None and not running and recording_done:
            self.run_button.setEnabled(False)                   # a finished recording is not resumed

    # ----- send -------------------------------------------------------------------
    def send(self):
        if not self.send_button.isEnabled():
            return
        if self.standalone is not None:
            self.export_files()
            return
        spec = dict(self.inputs["return_spec"])
        panel = self.water_panel
        if panel.enabled_box.isChecked() and panel.export_plane.isChecked() and self.water_source is None:
            # No water prop in the scene: return a plain water-height plane (as the Drape Tool does).
            spec["water_plane"] = dict(height=panel.fields["height"].value(), x=float(self.water_preview.center[0]),
                                       z=float(self.water_preview.center[2]), extent=self.water_preview.extent)
        frames = self.recorded if self.animation() else None
        if frames is not None:
            spec["animation_format"] = self.return_format.currentData()
        self.return_worker = ReturnWorker(self.inputs, self.vertices.copy(), self, spec=spec, frames=frames)
        self.return_worker.finished.connect(self.sent)
        self.status.set_state("sending")
        self.status.setText("Sending to DAZ…")
        self.update_buttons()
        self.return_worker.start()

    def export_files(self):
        """Standalone: write the hair to files of the user's choice (still OBJ/FBX, animation FBX)."""
        config = self.standalone.config
        animated = self.animation()
        output = config.get("output")
        if not output:
            output, _ = QFileDialog.getSaveFileName(self, "Export hair", "", "FBX animation (*.fbx)" if animated
                                                    else "Wavefront OBJ (*.obj);;FBX mesh (*.fbx)")
            if not output:
                return
            if not Path(output).suffix:
                output += ".fbx" if animated else ".obj"
        overwrite = bool(config.get("overwrite")) or not config.get("output")   # the save dialog already asked
        water = None
        panel = self.water_panel
        if panel.enabled_box.isChecked() and panel.export_plane.isChecked() and self.water_source is None:
            water = dict(height=panel.fields["height"].value(), x=float(self.water_preview.center[0]),
                         z=float(self.water_preview.center[2]), extent=self.water_preview.extent)
        frames = self.recorded if animated else None
        vertices, inputs = self.vertices.copy(), dict(self.inputs)
        from hair_standalone import export

        class Export(QThread):
            def run(worker):
                worker.error = worker.result = None
                try:
                    worker.result = export(self.standalone, inputs, output, vertices, frames, water, overwrite)
                except Exception as error:
                    worker.error = str(error)
        self.return_worker = Export(self)
        self.return_worker.finished.connect(self.exported)
        self.status.set_state("sending")
        self.status.setText("Exporting…")
        self.update_buttons()
        self.return_worker.start()

    @Slot()
    def exported(self):
        worker, self.return_worker = self.return_worker, None
        self.update_buttons()
        if worker.error:
            self.status.set_state("failed")
            self.status.setText("Export failed; the result is still here to retry.")
            QMessageBox.critical(self, APP_NAME, worker.error)
            return
        self.status.set_state("sent")
        self.status.setText("Exported: " + str(worker.result[1]))
        if self.close_after.isChecked() or self.close_pending:
            self.close()

    @Slot()
    def sent(self):
        error = self.return_worker.error
        self.return_worker = None
        self.update_buttons()
        if error:
            self.status.set_state("failed")
            self.status.setText("Send failed; the result is still here to retry.")
            QMessageBox.critical(self, APP_NAME, error)
            return
        self.status.set_state("sent")
        self.status.setText("The Spin Tools receive window imports it; adjust and send again to replace it.")
        if self.close_after.isChecked() or self.close_pending:
            self.close()
        else:
            self.showMinimized()

    def closeEvent(self, event):
        if self.return_worker is not None:
            self.close_pending = True
            self.status.setText("Finishing the send before closing…")
            event.ignore()
            return
        if self.worker is not None:
            self.close_pending = True
            self.stop_worker()
            self.status.setText("Stopping the simulation before closing…")
            event.ignore()
            return
        if self.builder is not None:
            self.builder.wait(30000)
        spin_ui.save_window(self, self.settings, self.splitter)
        super().closeEvent(event)
