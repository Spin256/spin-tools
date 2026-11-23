# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Two-role soft-prop preview with material/support controls and reliable return."""
import time
from pathlib import Path
import numpy as np
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
from PySide6.QtCore import Qt,QThread,Signal,QSettings,Slot,QTimer,QSignalBlocker
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox, QLineEdit,
                               QRadioButton, QButtonGroup, QDoubleSpinBox, QSpinBox, QCheckBox, QSplitter,
                               QMessageBox, QFileDialog, QSlider, QApplication)

from branding import APP_NAME, VERSION
import sys as _sys
from pathlib import Path as _Path
_SUITE_APP = str(_Path(__file__).resolve().parents[2] / "app")
if _SUITE_APP not in _sys.path:
    _sys.path.append(_SUITE_APP)          # suite app/: shared interface (spin_ui) and keybindings
import spin_ui
from body_material import PRESETS,JIGGLE_PRESETS
from body_geometry import adjacency,grow,shrink,brush_region,open_edge_vertices,border_band,connected_labels
from body_guides import pull_weights
from body_regions import NamedRegions, deformation_score, suggest_patches
NO_REGION = "-none-"
NO_TEMPLATE = "-none-"        # no tissue template / motion example on the region
HIGHLIGHT = "QPushButton { background-color: palette(highlight); color: palette(highlighted-text); }"
from body_solver import simulate,RunControl
from drape_animation import AnimationFrames
from drape_return import write_static_result,write_animation_result
from live_return import publish_revision
from standalone_export import export_result,output_paths


def polydata(points,faces):
    data=vtk.vtkPolyData();vertices=vtk.vtkPoints()
    vertices.SetData(numpy_to_vtk(points,deep=True));data.SetPoints(vertices)
    cells=vtk.vtkCellArray()
    for face in faces:
        cells.InsertNextCell(3)
        for i in face:cells.InsertCellPoint(int(i))
    data.SetPolys(cells)
    return data


def actor(renderer,data,color,opacity=1):
    normals=vtk.vtkPolyDataNormals();normals.SetInputData(data);normals.SplittingOff()
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(normals.GetOutputPort())
    prop=vtk.vtkActor();prop.SetMapper(mapper);prop.GetProperty().SetColor(*color);prop.GetProperty().SetOpacity(opacity)
    renderer.AddActor(prop)
    return prop


class SimulationWorker(QThread):
    frame=Signal(object);failed=Signal(str);paused=Signal()
    def __init__(self,inputs,settings,control,parent=None):
        super().__init__(parent)
        self.inputs,self.settings,self.control=inputs,settings,control
        self.animation=None;self.error=None;self.complete=False
        self.collider_animation = {}
        self.loop_status = ''
        self.loop_closed = False
    def run(self):
        pending=None;last=None
        def flush():
            nonlocal pending,last
            if pending is not None:self.frame.emit(pending);pending=None;last=time.monotonic()
        def paused():flush();self.paused.emit()
        motion=self.inputs.get('motion')
        try:
            if motion is not None:self.animation=AnimationFrames(motion,self.inputs['garment'][0]*100)
            def publish(payload):
                nonlocal pending
                if self.animation is not None and not payload.get('preview'):
                    sample = payload['sample']
                    if 'loop_status' in payload:
                        self.loop_status = payload['loop_status']
                        self.loop_closed = payload.get('loop_closed', False)
                    self.animation.put(sample, payload['vertices'])
                    for index, points in payload['colliders'].items():
                        if index not in self.collider_animation:
                            self.collider_animation[index] = AnimationFrames(motion, points)
                        self.collider_animation[index].put(sample, points)
                pending=payload
                if last is None or time.monotonic()-last>=1/30:flush()
            faces = self.settings.pop('collider_faces', None)
            colliders = self.inputs['colliders'] if faces is None else [
                (points, kept) for (points, _), kept in zip(self.inputs['colliders'], faces)]
            simulate(self.inputs['garment'],colliders,publish,self.isInterruptionRequested,
                     ranges=self.inputs['garment_ranges'],motion=motion,control=self.control,on_pause=paused,**self.settings)
            flush()
            completed = not self.isInterruptionRequested() and not self.control.stopped
            if completed and self.animation is not None:
                tracks = [self.animation, *self.collider_animation.values()]
                if any(len(track.seen) != track.count for track in tracks):
                    raise ValueError('The animation ended before all preview frames were recorded.')
            self.complete = completed
        except Exception as error:
            flush();self.error=str(error);self.failed.emit(self.error)


class ExportWorker(QThread):
    def __init__(self,window,output=None,overwrite=False):
        super().__init__(window)
        self.vertices=window.vertices.copy();self.ranges=window.inputs['garment_ranges']
        self.animation=window.animation;self.spec=window.inputs.get('return_spec');self.project=window.project
        if self.spec and self.animation is not None:     # DAZ return format
            self.spec = dict(self.spec, animation_format=window.return_format.currentData())
        self.live_session = window.inputs.get('live_session')
        self.output,self.overwrite=output,overwrite;self.error=None;self.result=None

    def write_daz_result(self, spec):
        if self.animation is not None:
            write_animation_result(spec, self.animation, self.ranges)
        else:
            write_static_result(spec, self.vertices, self.ranges)

    def run(self):
        try:
            if self.live_session:
                self.result = publish_revision(self.live_session, self.spec, self.write_daz_result)
            elif self.spec:
                self.write_daz_result(self.spec)
            else:self.result=export_result(self.project,self.output,self.vertices,self.animation,overwrite=self.overwrite)
        except Exception as error:self.error=str(error)


class SoftWindow(QWidget):
    def __init__(self,inputs,project=None):
        super().__init__()
        self.inputs,self.project=inputs,project
        self.vertices=inputs['garment'][0]*100
        self.start=self.vertices.copy();self.supports=np.zeros(len(self.vertices),dtype=bool)
        self.softness=np.zeros(len(self.vertices),dtype=np.float32);self.rigid=np.zeros(len(self.vertices),dtype=bool)
        self.ignore_collider = np.zeros(len(self.vertices), dtype=bool)
        self.suggested = np.full(len(self.vertices), -1)
        self.regions = NamedRegions(len(self.vertices))
        self.region_settings = {}          # name -> settings applied to that region (recalled on choosing it)
        self.rows=adjacency(len(self.vertices),inputs['garment'][1])
        self.open_border = open_edge_vertices(len(self.vertices), inputs['garment'][1])
        self.pinned_border = self.open_border
        self.worker=None;self.export_worker=None;self.control=None;self.animation=None
        self.loop_closed = False
        self.guides = {}
        self.motion_examples = []
        self.guides_dirty = False
        self.guide_drag = None
        self.collider_animation = {}
        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.play_timer.timeout.connect(self.advance_playback)
        self.close_pending=False;self.reset_pending=False;self.ready=False;self.error=False;self.painting=False;self.has_preview=False
        self.animated=inputs.get('motion') is not None
        self.settings=QSettings('Spin256',APP_NAME)
        self.setWindowTitle(APP_NAME+(' — DAZ bridge' if inputs.get('return_spec') else ' — Standalone'))
        self.resize(1200,800)
        layout=QHBoxLayout(self);self.splitter=QSplitter(Qt.Orientation.Horizontal);layout.addWidget(self.splitter)
        panel=QWidget();controls=QVBoxLayout(panel)
        label=QLabel('Soft bodies: '+inputs['garment_label']+'\nCollision objects: '+', '.join(inputs['collision_labels']))
        label.setWordWrap(True);label.setTextFormat(Qt.TextFormat.PlainText);controls.addWidget(label)
        mode=QLabel('Animated result (baked; no figure rig).' if self.animated else 'Static result for the current pose (no figure rig).')
        mode.setWordWrap(True);controls.addWidget(mode)
        self.edits=[]
        self.shared_sections=spin_ui.Sections(controls,self.settings)
        self.sections=self.shared_sections.items
        def group(title):
            return self.shared_sections.form(title,title in ('Region painting', 'Named regions'))
        def number(form,label,low,high,value,step=1):
            field=QDoubleSpinBox();field.setRange(low,high);field.setSingleStep(step);field.setValue(value)
            form.addRow(label,field);self.edits.append(field);return field
        def button(form,label,callback):
            field=QPushButton(label);field.clicked.connect(callback);form.addRow(field);self.edits.append(field);return field
        # Workflow: find and paint areas, save them as named regions, then choose a
        # named region below; every later section works on it and remembers its own settings for it.
        support=group('Region painting')
        self.suggest_frames = QLineEdit()
        self.suggest_frames.setPlaceholderText('Example frame numbers, separated by commas')
        self.suggest_frames.setToolTip('Frames where the mesh itself changes shape. They are compared with the '
                                       'first frame of the timeline; moving, turning or swinging limbs as a whole '
                                       'is ignored. DAZ motion examples use these frames when their own box is empty.')
        support.addRow('Example frames', self.suggest_frames)
        self.suggest_threshold = number(support, 'Suggest above (mm)', .1, 50, .5, .1)
        self.suggest_part = number(support, 'Rigid part size (cm)', 3, 50, 15, 1)
        self.suggest_part.setToolTip('Roughly the size of a body part that moves as one piece (forearm, thigh, '
                                     'chest). Smaller finds finer changes but misses smooth swelling.')
        self.suggest_button = button(support, 'Suggest moving areas', self.suggest_areas)
        self.pick_suggested = QCheckBox('Pick suggested area (click)')
        self.pick_suggested.setToolTip('Click an orange area to select it. Shift-click adds another.')
        support.addRow(self.pick_suggested)
        self.edits.append(self.pick_suggested)
        for widget in (self.suggest_frames, self.suggest_threshold, self.suggest_part, self.suggest_button, self.pick_suggested):
            widget.setVisible(self.animated)
            if support.labelForField(widget) is not None:
                support.labelForField(widget).setVisible(self.animated)
        self.paint=QCheckBox('Paint selection (left drag)');self.paint.setChecked(False);support.addRow(self.paint);self.edits.append(self.paint)
        self.pick_soft = QCheckBox('Select soft patch (click)')
        self.pick_soft.setToolTip('Click a connected soft patch to select it. Shift-click adds another patch.')
        support.addRow(self.pick_soft)
        self.edits.append(self.pick_soft)
        self.erase=QCheckBox('Remove from selection');support.addRow(self.erase);self.edits.append(self.erase)
        self.radius=number(support,'Brush radius (cm)',.1,100,3,.5)
        self.support_label=QLabel();self.support_label.setWordWrap(True);support.addRow(self.support_label)
        self.selection_controls = [
            self.paint, self.pick_soft, self.erase, self.radius,
            button(support,'Grow selection +',lambda:self.change_selection(grow(self.supports,self.rows))),
            button(support,'Shrink selection −',lambda:self.change_selection(shrink(self.supports,self.rows))),
            button(support,'Clear selection',self.clear_supports),
        ]
        self.selection_controls.append(button(support,'Select all',lambda:self.change_selection(np.ones(len(self.vertices),dtype=bool))))
        self.region_name = QLineEdit()
        self.region_name.setPlaceholderText('e.g. Left breast, Crotch back')
        support.addRow('Region name', self.region_name)
        button(support, 'Save selection as region', self.save_named_region)

        named = group('Named regions')
        named_note = QLabel('Choosing a region selects it and recalls its settings.')
        named_note.setWordWrap(True)
        named.addRow(named_note)
        self.named_regions = QComboBox()
        self.named_regions.addItem(NO_REGION)          # choosing it clears the selection
        named.addRow('Region', self.named_regions)
        self.edits.append(self.named_regions)
        self.named_regions.activated.connect(lambda _: self.select_named_region())
        button(named, 'Delete region', self.delete_named_region)
        button(named, 'Clear all regions', self.clear_regions)

        tissue = group('Soft tissue')
        # Applies to the chosen named region: grayed out while Region is -none-; choosing a
        # region recalls its template (-none- if it has none) and highlights Rigid / Protect / Ignore when they apply.
        tissue_note = QLabel('Applies to the named region. A template also sets the Deformation values.')
        tissue_note.setWordWrap(True)
        tissue.addRow(tissue_note)
        self.preset=QComboBox();self.preset.addItem(NO_TEMPLATE);self.preset.addItems(PRESETS)
        self.preset.setToolTip('-none- with Apply clears the region\'s template (its softness).')
        tissue.addRow('Tissue template',self.preset);self.edits.append(self.preset)
        first=next(iter(PRESETS.values()))
        self.fields={'softness':number(tissue,'Paint softness (%)',1,100,first['softness'],5)}
        self.region_controls = [self.preset, self.fields['softness']]
        self.region_controls.append(button(tissue,'Apply template to region',self.apply_soft_selection))
        self.rigid_button = button(tissue,'Make region rigid / ghost',self.apply_rigid_selection)
        self.protect_button = button(tissue,'Protect region',self.protect_selection)
        self.region_controls += [self.rigid_button, self.protect_button,
                                 button(tissue,'Protect everything else',self.protect_unselected)]
        self.ignore_button = button(tissue,'Ignore collider contact on region',lambda:self.set_collider_contact(False))
        self.ignore_button.setToolTip('External colliders can pass through this region. Softness, jiggle and self-contact still apply.')
        self.region_controls += [self.ignore_button,
                                 button(tissue,'Restore collider contact on region',lambda:self.set_collider_contact(True))]
        self.tissue_form = tissue
        examples_panel = group('DAZ motion examples')
        self.shared_sections.set_visible('DAZ motion examples',self.animated)
        examples_note = QLabel('Applies to the named region: choose a behavior and example frames, then Apply '
                               '(-none- removes its example). Frame one is the rest shape.')
        examples_note.setWordWrap(True)
        examples_panel.addRow(examples_note)
        self.example_kind = QComboBox()
        # Bounce: example shapes blended by the body's motion; Opening: driven by the collider.
        self.example_kind.addItems([NO_TEMPLATE, 'Bounce', 'Opening'])
        self.example_kind.setToolTip('Bounce: frames with the area pushed different ways, blended as the body moves. '
                                     'Opening: frames with the collider in different places, blended by where it is.')
        examples_panel.addRow('Behavior', self.example_kind)
        # The reference is always the timeline's first frame.
        self.example_frames = QLineEdit()
        self.example_frames.setPlaceholderText('Empty: the Region painting frames')
        self.example_kind.currentTextChanged.connect(self.example_frames.clear)
        self.example_kind.currentTextChanged.connect(lambda *_: self.update_region_controls())
        examples_panel.addRow('Example frames', self.example_frames)
        self.save_example_button = button(examples_panel, 'Apply', self.save_motion_example)
        self.save_example_button.setToolTip('Store this behavior on the chosen region; with -none-, remove its example.')
        # Examples belong to named regions: Apply stores one on the chosen region (-none- removes it),
        # choosing the region recalls it; there is no separate list.
        self.example_feedback = QLabel('No motion examples applied.')
        self.example_feedback.setWordWrap(True)
        examples_panel.addRow(self.example_feedback)
        self.edits.extend((self.example_kind, self.example_frames))
        self.bounce_blend = QSpinBox()
        self.bounce_blend.setRange(0, 10)
        self.bounce_blend.setValue(8)
        self.bounce_blend.setToolTip('Bounce fades into the unpainted skin over this many rings. Raise it if the edge '
                                     'creases; lower it for a small area.')
        examples_panel.addRow('Bounce edge blend (rings)', self.bounce_blend)
        self.edits.append(self.bounce_blend)
        def opening_slider(label, maximum, initial, unit, tip):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, maximum)
            slider.setValue(initial)
            slider.setToolTip(tip)
            slider.setAccessibleName(label)
            value = QLabel(f'{initial}{unit}')
            slider.valueChanged.connect(lambda amount: value.setText(f'{amount}{unit}'))
            slider.valueChanged.connect(self.opening_settings_changed)
            layout.addWidget(slider, 1)
            layout.addWidget(value)
            examples_panel.addRow(label, row)
            return slider
        self.fold_rounding = opening_slider('Fold rounding', 100, 25, '%',
                                           'Softens compressed folds. Set to 0% to keep tight creases.')
        self.lip_drag = opening_slider('Collider drag', 10, 3, ' mm',
                                       'Distance the lips aim to follow the collider along its surface. Contact can limit the result.')
        self.drag_reach = opening_slider('Drag reach', 50, 8, ' mm',
                                         'How far the tug fades along the painted mesh from contact. At 0 mm, only contact points pull.')

        # Collider no-contact painting: when the bodies are swapped for a second pass, the
        # solid part behind the lips is a collider; painted collider areas do not push (their faces are left out).
        self.collider_skip = [np.zeros(len(points), dtype=bool) for points, _ in inputs['colliders']]
        self.collider_rows = [adjacency(len(points), faces) for points, faces in inputs['colliders']]
        collider_panel = group('Collider contact')
        collider_note = QLabel('Painted collider areas (red) do not push the soft body.')
        collider_note.setWordWrap(True)
        collider_panel.addRow(collider_note)
        self.collider_paint = QCheckBox('Paint collider no-contact (left drag)')
        self.collider_erase = QCheckBox('Restore contact instead')
        collider_panel.addRow(self.collider_paint)
        collider_panel.addRow(self.collider_erase)
        self.edits.extend((self.collider_paint, self.collider_erase))
        self.collider_clear = button(collider_panel, 'Clear collider paint', self.clear_collider_paint)
        self.collider_label = QLabel()
        collider_panel.addRow(self.collider_label)
        self.shared_sections.set_visible('Collider contact', bool(inputs['colliders']))
        simulation=group('Deformation')
        deformation_note = QLabel('Applies to all soft areas.')
        deformation_note.setWordWrap(True)
        simulation.addRow(deformation_note)
        for key,title,lo,hi,step in [('depth','Maximum indentation (mm)',1,100,1),
                    ('spread','Spread / smoothing (%)',0,100,5),('recovery','Shape recovery (%)',0,100,5),('clearance','Contact clearance (mm)',0,10,.1)]:
            self.fields[key]=number(simulation,title,lo,hi,first[key],step)
        self.preset.currentTextChanged.connect(self.apply_material)
        self.opening_stretch = number(simulation,'Opening stretch (%)',50,300,150,10)
        self.opening_stretch.setToolTip('Allow more stretch near soft open edges and the rim of a painted pass-through area. 50% uses the normal strain limit.')
        self.part_soft_seal = QCheckBox('Part soft seal around moving collider')
        self.part_soft_seal.setToolTip('Paint the backing as pass-through and leave the lips soft with contact enabled. The moving collider can push the lips sideways as it enters.')
        simulation.addRow(self.part_soft_seal)
        self.part_soft_seal.setVisible(self.animated)
        self.steps=QSpinBox();self.steps.setRange(1,500);self.steps.setValue(40);simulation.addRow('Steps per run / resume',self.steps);self.edits.append(self.steps)
        self.pin_open_edges = QCheckBox('Pin open mesh edges')
        self.pin_open_edges.setToolTip('Keep cut borders and hole edges in their source pose while the painted interior deforms.')
        simulation.addRow(self.pin_open_edges)
        self.pin_edge_rings = QSpinBox()
        self.pin_edge_rings.setRange(1, 10)
        self.pin_edge_rings.setToolTip('One pins the edge itself. Higher values hold adjacent rows to preserve more of the seam.')
        simulation.addRow('Pinned border rings', self.pin_edge_rings)
        self.feather=QSpinBox();self.feather.setRange(0,10);self.feather.setValue(2);simulation.addRow('Protected boundary blend (rings)',self.feather);self.edits.append(self.feather)
        self.collision_accuracy=QSpinBox();self.collision_accuracy.setRange(1,8);self.collision_accuracy.setValue(2)
        simulation.addRow('Animation collision accuracy',self.collision_accuracy);self.edits.append(self.collision_accuracy)
        self.collision_accuracy.setVisible(self.animated);simulation.labelForField(self.collision_accuracy).setVisible(self.animated)
        simulation.labelForField(self.steps).setVisible(not self.animated);self.steps.setVisible(not self.animated)
        dynamics=group('Jiggle')
        jiggle_note = QLabel('Applies to all soft areas.' if self.animated else 'Jiggle needs an animated session.')
        jiggle_note.setWordWrap(True)
        dynamics.addRow(jiggle_note)
        self.jiggle_preset = QComboBox()
        self.jiggle_preset.addItems(['Custom', *JIGGLE_PRESETS])
        dynamics.addRow('Jiggle preset', self.jiggle_preset)
        self.edits.append(self.jiggle_preset)
        self.jiggle=QCheckBox('Enable motion-driven jiggle');self.jiggle.setEnabled(self.animated);dynamics.addRow(self.jiggle);self.edits.append(self.jiggle)
        self.preserve_volume = QCheckBox('Preserve painted volume')
        self.preserve_volume.setToolTip('Each connected painted area shares inertia and pushes back when compressed. Leave a fixed border around each area.')
        self.preserve_volume.setEnabled(False)
        dynamics.addRow(self.preserve_volume)
        self.vertical_bands = QCheckBox('Couple vertical motion in bands')
        self.vertical_bands.setToolTip('Give each horizontal slice of a painted area a shared vertical shift. Slices can stretch and compress slightly.')
        self.vertical_bands.setEnabled(False)
        dynamics.addRow(self.vertical_bands)
        self.directional_lag = QCheckBox('Favor lag in the direction of motion')
        self.directional_lag.setToolTip('Uses part of the damping to keep tissue trailing the motion of the body as it slows down.')
        self.directional_lag.setEnabled(False)
        dynamics.addRow(self.directional_lag)
        self.jiggle.toggled.connect(self.update_buttons)
        self.dynamic_fields={}
        for key,title,low,high,value,step in [('frequency','Stiffness / bounce frequency (Hz)',.25,12,3.,.25),
                    ('damping','Damping (/s)',0,60,12.,1.),('inertia','Movement inertia (%)',0,200,60.,5.)]:
            self.dynamic_fields[key]=number(dynamics,title,low,high,value,step)
        self.dynamic_fields['jiggle_gravity'] = number(dynamics,'Jiggle gravity (% of Earth)',0,100,0,5)
        self.dynamic_fields['jiggle_gravity'].setToolTip('Adds downward sag to procedural jiggle. Set to 0% to turn gravity off.')
        self.jiggle_preset.currentTextChanged.connect(self.apply_jiggle_preset)
        for key in ('frequency', 'damping', 'inertia'):
            self.dynamic_fields[key].valueChanged.connect(self.mark_jiggle_custom)
        for field in (self.preserve_volume, self.vertical_bands, self.directional_lag):
            field.toggled.connect(self.mark_jiggle_custom)
        guide_panel = group('Frame guides')
        self.shared_sections.set_visible('Frame guides',self.animated)
        guide_note = QLabel('After a run, pick a frame, select soft vertices and left-drag them, then run again.')
        guide_note.setWordWrap(True)
        guide_panel.addRow(guide_note)
        self.guide_move = QCheckBox('Left-drag selection to set guide')
        guide_panel.addRow(self.guide_move)
        self.guide_radius = number(guide_panel,'Guide pull radius (cm)',0,100,3,.5)
        self.guide_radius.setToolTip('Distance along the soft mesh beyond the selection. Set to 0 to pull only the selection.')
        self.guide_strength = number(guide_panel,'Guide strength (%)',0,100,65,5)
        self.guide_lead = QSpinBox()
        self.guide_lead.setRange(1,120)
        self.guide_lead.setValue(18)
        self.guide_lead.setToolTip('How many source frames before the first guide the mesh starts moving toward it.')
        guide_panel.addRow('Guide approach (frames)',self.guide_lead)
        self.guide_blend = QSpinBox()
        self.guide_blend.setRange(1,120)
        self.guide_blend.setValue(12)
        self.guide_blend.setToolTip('How many source frames after the last guide its influence takes to fade out.')
        guide_panel.addRow('Guide release (frames)',self.guide_blend)
        self.guide_label = QLabel()
        self.guide_label.setWordWrap(True)
        guide_panel.addRow(self.guide_label)
        self.remove_guide = button(guide_panel,'Remove guide at this frame',self.remove_current_guide)
        self.clear_guides_button = button(guide_panel,'Clear frame guides',self.clear_guides)
        loop = group('Animation loop')
        self.shared_sections.set_visible('Animation loop',self.animated)
        # Two choices: finish settled on the last frame's DAZ shape, or a seamless loop whose
        # last frame leads into frame 0. Radio buttons with "None".
        self.loop_none = QRadioButton('None')
        self.loop_none.setToolTip('The animation plays through as simulated.')
        loop.addRow(self.loop_none)
        self.close_loop = QRadioButton('Finish on last frame')
        self.close_loop.setToolTip('Over the last frames the painted areas ease onto their DAZ shape on the last frame and '
                                   'come to rest there, so the animation ends exactly as you keyed it.')
        loop.addRow(self.close_loop)
        self.loop_motion = QRadioButton('Loop animation (carry motion)')
        self.loop_motion.setToolTip('After the last frame the simulation steps into frame 0 and carries on until it meets '
                                    'the first run, so the animation repeats without restarting from rest. End the '
                                    "avatar's animation one pose before its first.")
        loop.addRow(self.loop_motion)
        self.loop_choice = QButtonGroup(self)
        for choice in (self.loop_none, self.close_loop, self.loop_motion):
            self.loop_choice.addButton(choice)
        (self.loop_motion if self.settings.value('loop/motion', False, type=bool) else self.loop_none).setChecked(True)
        self.loop_motion.toggled.connect(lambda value: self.settings.setValue('loop/motion', value))
        self.loop_frames = QSpinBox()
        self.loop_frames.setRange(2, 60)
        self.loop_frames.setValue(8)
        self.loop_frames.setToolTip('Finish: how many frames the painted areas take to settle onto the last frame. Loop: '
                                    'if the motion has not met the first run by the middle, it is steered onto the '
                                    'closest frame over this many frames.')
        loop.addRow('Blend frames', self.loop_frames)
        # DAZ return format for an animation: DUF by default, FBX selectable.
        self.return_format = spin_ui.return_format_box(self.settings)
        loop.addRow('DAZ return format', self.return_format)
        self.close_loop.toggled.connect(self.loop_settings_changed)
        self.loop_frames.valueChanged.connect(self.loop_settings_changed)
        self.loop_motion.toggled.connect(self.loop_settings_changed)
        view=group('View')
        self.view_controls={}
        for key,title,opacity in [('bodies','Soft bodies',100),('props','Props / collision objects',35),('ghosts','Ghost colliders',70)]:
            check=QCheckBox(title);check.setChecked(self.settings.value('view/'+key+'/visible',True,type=bool))
            slider=QSlider(Qt.Orientation.Horizontal);slider.setRange(0,100);slider.setValue(self.settings.value('view/'+key+'/opacity',opacity,type=int))
            view.addRow(check,slider);self.view_controls[key]=(check,slider)
            check.toggled.connect(self.update_visibility);slider.valueChanged.connect(self.update_visibility)
        self.flat_shading = QCheckBox('Flat shading')
        self.flat_shading.setChecked(self.settings.value('view/flat_shading', False, type=bool))
        view.addRow(self.flat_shading)
        self.mesh_edges = QCheckBox('Show mesh edges')
        self.mesh_edges.setChecked(self.settings.value('view/mesh_edges', False, type=bool))
        view.addRow(self.mesh_edges)
        reset_view=QPushButton('Reset view');reset_view.clicked.connect(self.reset_view);view.addRow(reset_view)
        self.view_form=view
        self.actions=spin_ui.ActionRow(self.settings,run_text='Run deformation',tool='soft_body',
                                       send_text=('Send animation to DAZ' if self.animated else 'Send to DAZ') if inputs.get('return_spec') else ('Export FBX animation…' if self.animated else 'Export static mesh…'),
                                       close_tip='Off: stay open after each send so you can inspect the result in DAZ and send again. On: close once the result is written.')
        self.run_button,self.reset_button,self.send_button,self.close_after=(self.actions.run_button,self.actions.reset_button,
                                                                            self.actions.send_button,self.actions.close_after)
        self.close_after.setVisible(bool(inputs.get('live_session')))
        self.run_button.clicked.connect(self.run_or_pause)
        self.reset_button.clicked.connect(self.reset_shape)
        self.reset_button.setToolTip('Reset to the starting shape. Region masks are kept.')
        playback = QWidget()
        playback_row = QHBoxLayout(playback)
        playback_row.setContentsMargins(0, 0, 0, 0)
        self.play_button = QPushButton('Replay')
        self.play_button.clicked.connect(self.toggle_playback)
        playback_row.addWidget(self.play_button)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        if self.animated:
            motion = inputs['motion']
            self.frame_slider.setRange(motion.start_frame, motion.end_frame)
            self.frame_slider.setValue(motion.start_frame)
        else:
            self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.show_animation_frame)
        self.frame_slider.sliderPressed.connect(self.stop_playback)
        playback_row.addWidget(self.frame_slider, 1)
        self.playback_label = QLabel('Frame')
        playback_row.addWidget(self.playback_label)
        self.frame_number = QSpinBox()
        self.frame_number.setRange(self.frame_slider.minimum(), self.frame_slider.maximum())
        self.frame_number.setValue(self.frame_slider.value())
        self.frame_number.setKeyboardTracking(False)
        self.frame_number.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.frame_number.setMinimumWidth(max(84, self.frame_number.fontMetrics().horizontalAdvance('−000000') + 16))
        self.frame_number.setAccessibleName('Go to frame')
        self.frame_number.setToolTip('Type a frame number and press Enter to jump to it.')
        self.frame_number.valueChanged.connect(self.seek_frame)
        self.frame_slider.valueChanged.connect(self.sync_frame_number)
        playback_row.addWidget(self.frame_number)
        playback.setVisible(self.animated)
        controls.addWidget(playback)
        self.send_button.clicked.connect(self.export)
        if inputs.get('live_session'):
            spin_ui.note('DAZ imports automatically while the receive prompt is open. Reopen Spin Soft Body Tool from the DAZ menu for another revision.',controls)
        spin_ui.note('DAZ return creates baked mesh copies and preserves the original figures; it does not create rigged morphs. Changing the original performance requires another solve.' if self.animated else 'Static surface deformation. Poses stay fixed. DAZ return creates mesh copies and preserves the original figures; it does not create rigged morphs.',controls)
        controls.addStretch()
        column,self.copyright_footer=spin_ui.control_column(panel, self.actions, 360, version=VERSION, tool_folder=Path(__file__).resolve().parent.parent)
        self.splitter.addWidget(column)
        self.viewport=QVTKRenderWindowInteractor(self)
        holder,self.status=spin_ui.viewport_with_status(self.viewport)
        self.status.setText('Paint a region and apply softness to begin.')
        self.splitter.addWidget(holder);self.splitter.setStretchFactor(1,1)
        self.splitter.setSizes([380,820])
        spin_ui.remember_window(self,self.settings,self.splitter)
        self.renderer=spin_ui.style_renderer(vtk.vtkRenderer());self.viewport.GetRenderWindow().AddRenderer(self.renderer)
        self.data=polydata(self.vertices,inputs['garment'][1]);self.surface_actor=actor(self.renderer,self.data,spin_ui.OBJECT_COLOR)
        self.ghost_data=polydata(self.start,np.empty((0,3),dtype=np.int32))
        self.ghost_actor=actor(self.renderer,self.ghost_data,(.9,.65,.2),.7)
        self.collider_data=[];self.collider_actors=[]
        for points,faces in inputs['colliders']:
            data=polydata(points*100,faces);self.collider_data.append(data)
            self.collider_actors.append(actor(self.renderer,data,(.65,.5,.75),.6))
        self.support_data=vtk.vtkPolyData();self.support_points=vtk.vtkPoints();self.support_data.SetPoints(self.support_points)
        glyph=vtk.vtkVertexGlyphFilter();glyph.SetInputData(self.support_data)
        mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(glyph.GetOutputPort())
        self.support_actor=vtk.vtkActor();self.support_actor.SetMapper(mapper);self.support_actor.GetProperty().SetColor(1,.5,.1)
        self.support_actor.GetProperty().SetPointSize(5);self.support_actor.PickableOff();self.renderer.AddActor(self.support_actor)
        self.picker=vtk.vtkCellPicker();self.picker.PickFromListOn();self.picker.AddPickList(self.surface_actor);self.picker.AddPickList(self.ghost_actor)
        self.collider_picker=vtk.vtkCellPicker();self.collider_picker.PickFromListOn()
        for prop in self.collider_actors:self.collider_picker.AddPickList(prop)
        self.style=vtk.vtkInteractorStyleTrackballCamera();self.viewport.SetInteractorStyle(self.style)
        self.style.AddObserver('LeftButtonPressEvent',self.mouse_down)
        self.style.AddObserver('MouseMoveEvent',self.mouse_move)
        self.style.AddObserver('LeftButtonReleaseEvent',self.mouse_up)
        self.camera=spin_ui.ViewportCamera(self.renderer,self.viewport.GetRenderWindow().Render,
                                           lambda:[self.surface_actor]+self.collider_actors)
        def cancel():
            for box in (self.paint,self.pick_soft,self.pick_suggested,self.guide_move):box.setChecked(False)
        def when(button,callback):
            return lambda:button.isEnabled() and callback()
        grow_button,shrink_button,clear_button=self.selection_controls[4:7]
        self.keys=spin_ui.Shortcuts(self,'soft_body',{
            'pause_resume':self.run_or_pause,'run_simulation':self.run_or_pause,'reset_view':self.reset_view,
            'send_to_daz':self.export,'reset_simulation':when(self.reset_button,self.reset_shape),
            'grow_selection':when(grow_button,grow_button.click),'shrink_selection':when(shrink_button,shrink_button.click),
            'clear_selection':when(clear_button,clear_button.click),'cancel_interaction':cancel,
            'show_avatar':lambda:self.view_controls['props'][0].toggle()},camera=self.camera)
        self.shortcuts=self.keys.shortcuts
        lighting_box=QWidget();lighting_layout=QVBoxLayout(lighting_box);lighting_layout.setContentsMargins(0,0,0,0)
        self.lighting=spin_ui.Lighting(lighting_layout,self.settings);self.view_form.addRow(lighting_box)
        self.lighting.attach(self.renderer,self.viewport.GetRenderWindow().Render)
        self.view_form.addRow(self.keys.help_button())
        self.view_bar = spin_ui.add_view_toolbar(self.keys, self.status)   # orbit/pan/zoom/reset buttons
        self.pin_open_edges.toggled.connect(self.refresh_regions)
        self.pin_edge_rings.valueChanged.connect(self.update_pin_width)
        self.guide_move.toggled.connect(self.guide_mode_changed)
        self.paint.toggled.connect(self.guide_paint_changed)
        self.pick_soft.toggled.connect(self.soft_patch_mode_changed)
        self.guide_strength.valueChanged.connect(self.guide_settings_changed)
        self.guide_lead.valueChanged.connect(self.guide_settings_changed)
        self.guide_blend.valueChanged.connect(self.guide_settings_changed)
        self.flat_shading.toggled.connect(self.update_surface_display)
        self.mesh_edges.toggled.connect(self.update_surface_display)
        self.update_surface_display()
        self.refresh_regions()
        if self.animated:
            self.show_animation_frame(self.frame_slider.value())
        self.reset_view();self.update_buttons()

    def apply_material(self,name):
        if name in PRESETS:
            for key,value in PRESETS[name].items():self.fields[key].setValue(value)
        self.update_region_controls()

    def update_region_controls(self):
        """Soft tissue and DAZ motion examples apply to the chosen named region: grayed out while Region is -none-;
        Rigid, Protect and Ignore are highlighted when the region has them."""
        if not hasattr(self, 'region_controls') or not hasattr(self, 'region_settings'):
            return
        editable = getattr(self, 'worker', None) is None and getattr(self, 'export_worker', None) is None
        name = self.current_region()
        on = editable and name is not None
        for widget in self.region_controls:
            widget.setEnabled(on)
        self.fields['softness'].setEnabled(on and self.preset.currentText() != NO_TEMPLATE)
        stored = self.region_settings.get(name, {}) if name is not None else {}
        for button, active in ((self.rigid_button, stored.get('rigid')), (self.protect_button, stored.get('protected')),
                               (self.ignore_button, stored.get('collider') is False)):
            button.setStyleSheet(HIGHLIGHT if active else '')
        self.example_kind.setEnabled(on and self.animated)
        self.example_frames.setEnabled(on and self.animated and self.example_kind.currentText() != NO_TEMPLATE)
        self.save_example_button.setEnabled(on and self.animated)

    def recall_region(self, name):
        """Show what is stored for the region: its template (or -none-), softness and motion example (or -none-)."""
        stored = self.region_settings.get(name, {})
        with QSignalBlocker(self.preset):
            self.preset.setCurrentText(stored['template'] if 'softness' in stored and 'template' in stored else NO_TEMPLATE)
        if 'softness' in stored:
            self.fields['softness'].setValue(stored['softness'])
        with QSignalBlocker(self.example_kind):
            if 'example' in stored:
                kind, reference, frames = stored['example']
                self.example_kind.setCurrentText(kind)
                self.example_frames.setText(frames)
            else:
                self.example_kind.setCurrentText(NO_TEMPLATE)
                self.example_frames.clear()
        self.update_region_controls()
        return stored

    def apply_jiggle_preset(self, name):
        if name in JIGGLE_PRESETS:
            frequency, damping, inertia, directional = JIGGLE_PRESETS[name]
            with QSignalBlocker(self.jiggle_preset):
                for key, value in zip(('frequency', 'damping', 'inertia'), (frequency, damping, inertia)):
                    self.dynamic_fields[key].setValue(value)
                self.preserve_volume.setChecked(True)
                self.vertical_bands.setChecked(False)
                self.directional_lag.setChecked(directional)
                self.jiggle.setChecked(True)

    def mark_jiggle_custom(self):
        if not self.jiggle_preset.signalsBlocked():
            self.jiggle_preset.setCurrentIndex(0)

    def brush_collider(self):
        x,y=self.viewport.GetEventPosition()
        if not self.collider_picker.Pick(x,y,0,self.renderer):
            return
        prop=self.collider_picker.GetActor()
        if prop not in self.collider_actors:
            return
        index=self.collider_actors.index(prop)
        seed=self.collider_picker.GetPointId()
        if seed<0:
            return
        points=np.asarray(self.collider_data[index].GetPoints().GetData())*.01
        selected=brush_region(points,self.collider_rows[index],seed,self.radius.value()*.01)
        self.collider_skip[index][selected]=not self.collider_erase.isChecked()
        self.refresh_collider_paint()

    def clear_collider_paint(self):
        for mask in self.collider_skip:mask[:]=False
        self.refresh_collider_paint()

    def collider_faces(self):
        """Each collider's faces without the painted no-contact ones (all three corners painted)."""
        return [faces[~np.all(skip[faces],axis=1)] for (_,faces),skip in zip(self.inputs['colliders'],self.collider_skip)]

    def refresh_collider_paint(self):
        for data,prop,skip in zip(self.collider_data,self.collider_actors,self.collider_skip):
            if skip.any():
                colors=np.tile(np.asarray([166,128,191],dtype=np.uint8),(len(skip),1))
                colors[skip]=[220,60,60]
                data.GetPointData().SetScalars(numpy_to_vtk(colors,deep=True))
                prop.GetMapper().ScalarVisibilityOn()
            else:
                data.GetPointData().SetScalars(None)
                prop.GetMapper().ScalarVisibilityOff()
            data.Modified()
        count=sum(int(np.count_nonzero(skip)) for skip in self.collider_skip)
        self.collider_label.setText(f'{count} collider vertices without contact.' if count else 'All collider areas push.')
        self.viewport.GetRenderWindow().Render()

    def clear_supports(self):self.supports[:]=False;self.refresh_supports()
    def soft_mask(self):
        painted = self.softness > 0
        return painted & ~self.pinned_border if self.pin_open_edges.isChecked() else painted

    def update_pin_width(self, rings):
        self.pinned_border = border_band(self.open_border, self.rows, rings)
        self.refresh_regions()

    def update_region_count(self):
        summary = f'{np.count_nonzero(self.supports)} selected; {np.count_nonzero(self.soft_mask())} soft; {np.count_nonzero(self.rigid)} rigid vertices'
        if self.pin_open_edges.isChecked():
            summary += f'; {np.count_nonzero(self.pinned_border)} on pinned edges'
        if self.ignore_collider.any():
            summary += f'; {np.count_nonzero(self.ignore_collider)} ignoring collider contact (purple)'
        self.support_label.setText(summary)

    def clean_view(self):
        """While simulating or playing the result the mesh is drawn without paint or selection; once it stops,
        both show again so regions can be edited."""
        return self.worker is not None or self.play_timer.isActive()

    def refresh_supports(self):
        # Painting, growing or picking leaves the saved region: show -none-.
        if hasattr(self, 'named_regions') and self.named_regions.currentIndex() > 0 and self.current_region() is None:
            with QSignalBlocker(self.named_regions):
                self.named_regions.setCurrentIndex(0)
        shown = self.supports & (not self.clean_view())
        self.support_points.SetData(numpy_to_vtk(self.vertices[shown],deep=True))
        self.update_region_count()
        self.update_region_controls()
        self.viewport.GetRenderWindow().Render()
    def change_selection(self,mask):self.supports[:]=mask;self.refresh_supports()
    def select_soft_patch(self):
        x, y = self.viewport.GetEventPosition()
        if self.picker.Pick(x, y, 0, self.renderer):
            seed = self.picker.GetPointId()
            soft = self.soft_mask()
            if 0 <= seed < len(soft) and soft[seed]:
                labels, _ = connected_labels(self.rows, soft)
                selected = labels == labels[seed]
                if self.viewport.GetShiftKey():
                    selected |= self.supports
                self.change_selection(selected)

    def suggest_areas(self):
        motion = self.inputs['motion']
        try:
            frames = sorted({int(value) for value in self.suggest_frames.text().split(',') if value.strip()})
            if not frames or any(not motion.start_frame <= frame <= motion.end_frame for frame in frames):
                raise ValueError()
        except ValueError:
            self.status.setText('Enter example frame numbers from the captured DAZ timeline to compare.')
            return
        reference = 0                    # the timeline's first frame; this section needs nothing from elsewhere
        score = deformation_score(motion.body_positions(reference),
                                  [motion.body_positions(frame - motion.start_frame) for frame in frames], self.rows,
                                  self.suggest_part.value() * .01)
        self.suggested, count = suggest_patches(score, self.rows, self.suggest_threshold.value() * .001)
        if count:
            self.pick_suggested.setChecked(True)
            self.status.setText(f'{count} suggested area(s) in orange. Click the ones you want, then save them '
                                'as named regions.')
        else:
            self.status.setText('Nothing changes shape more than the threshold in those frames.')
        self.refresh_regions()

    def select_suggested_area(self):
        x, y = self.viewport.GetEventPosition()
        if self.picker.Pick(x, y, 0, self.renderer):
            seed = self.picker.GetPointId()
            if 0 <= seed < len(self.suggested) and self.suggested[seed] >= 0:
                selected = self.suggested == self.suggested[seed]
                if self.viewport.GetShiftKey():
                    selected |= self.supports
                self.change_selection(selected)

    def save_named_region(self):
        if self.region_name.text().strip() == NO_REGION:
            self.status.setText('Choose another name for the region.')
            return
        try:
            name = self.regions.save(self.region_name.text(), self.supports)
        except ValueError as error:
            self.status.setText(str(error))
            return
        if self.named_regions.findText(name) < 0:
            self.named_regions.addItem(name)
        self.named_regions.setCurrentText(name)
        self.recall_region(name)
        self.region_name.clear()
        self.status.setText(f'Saved region "{name}" ({np.count_nonzero(self.supports)} vertices).')

    def current_region(self):
        """The chosen named region while the selection is exactly that region, else None."""
        name = self.named_regions.currentText()
        if name and name in self.regions.masks and np.array_equal(self.regions.masks[name], self.supports):
            return name
        return None

    def remember(self, **values):
        name = self.current_region()
        if name is not None:
            self.region_settings.setdefault(name, {}).update(values)

    def select_named_region(self):
        """Select the region's vertices and recall the settings stored for it."""
        name = self.named_regions.currentText()
        if not name or name == NO_REGION:
            self.change_selection(np.zeros(len(self.vertices), dtype=bool))
            self.status.setText('No region chosen; selection cleared.')
            return
        self.change_selection(self.regions.get(name))
        stored = self.recall_region(name)
        parts = []
        if 'softness' in stored:
            parts.append(f"{stored['template']} {stored['softness']:g}% soft")
        if stored.get('rigid'):
            parts.append('rigid')
        if stored.get('protected'):
            parts.append('protected')
        if 'collider' in stored:
            parts.append('collider contact ' + ('on' if stored['collider'] else 'ignored'))
        if 'example' in stored:
            parts.append(f"{stored['example'][0]} example")
        self.status.setText(f'Region "{name}" selected' + (': ' + ', '.join(parts) if parts else '') + '.')

    def delete_named_region(self):
        name = self.named_regions.currentText()
        if name and name != NO_REGION:
            mask = self.regions.get(name) & self.soft_mask()
            self.motion_examples[:] = [saved for saved in self.motion_examples if not np.array_equal(mask, saved[1])]
            self.regions.remove(name)
            self.region_settings.pop(name, None)
            self.named_regions.removeItem(self.named_regions.currentIndex())

    def brush(self):
        x,y=self.viewport.GetEventPosition()
        if self.picker.Pick(x,y,0,self.renderer):
            seed=self.picker.GetPointId()
            if seed>=0:
                selected=brush_region(self.vertices,self.rows,seed,self.radius.value())
                self.supports[selected]=not self.erase.isChecked();self.refresh_supports()
    def working(self, text):
        """Show what is being applied before the short pause it takes."""
        self.status.setText(text + '…')
        self.status.repaint()
        QApplication.processEvents()

    def applied(self, text):
        self.status.setText(f'{text} ({np.count_nonzero(self.supports)} vertices).')

    def apply_soft_selection(self):
        if self.preset.currentText() == NO_TEMPLATE:
            # -none- clears the region's template: its painted softness goes.
            self.working('Clearing the template')
            name = self.current_region()
            if name is not None:
                for key in ('template', 'softness', 'rigid', 'protected'):
                    self.region_settings.get(name, {}).pop(key, None)
            self.softness[self.supports]=0;self.rigid[self.supports]=False;self.refresh_regions()
            self.update_region_controls()
            self.applied('Template cleared')
            return
        self.working('Applying softness')
        self.remember(template=self.preset.currentText(), softness=self.fields['softness'].value(), rigid=False,
                      protected=False)
        self.softness[self.supports]=self.fields['softness'].value()/100.;self.rigid[self.supports]=False;self.refresh_regions()
        self.update_region_controls()
        self.applied(f"Softness {self.fields['softness'].value():g}% applied")
    def apply_rigid_selection(self):
        self.working('Making the selection rigid')
        self.remember(rigid=True, protected=False)
        self.region_settings.get(self.current_region() or '', {}).pop('softness', None)
        self.rigid[self.supports]=True;self.softness[self.supports]=0;self.refresh_regions()
        with QSignalBlocker(self.preset):
            self.preset.setCurrentText(NO_TEMPLATE)
        self.update_region_controls()
        self.applied('Region made rigid')
    def protect_unselected(self):
        self.working('Protecting unselected')
        self.softness[~self.supports]=0;self.refresh_regions()
        self.status.setText('Everything outside the selection is protected.')
    def protect_selection(self):
        self.working('Protecting the selection')
        self.remember(protected=True, rigid=False)
        self.region_settings.get(self.current_region() or '', {}).pop('softness', None)
        self._protect_selection()
        with QSignalBlocker(self.preset):
            self.preset.setCurrentText(NO_TEMPLATE)
        self.update_region_controls()
        self.applied('Region protected')
    def _protect_selection(self):self.softness[self.supports]=0;self.rigid[self.supports]=False;self.refresh_regions()
    def set_collider_contact(self, enabled):
        self.working('Restoring collider contact' if enabled else 'Ignoring collider contact')
        self.remember(collider=enabled)
        self.ignore_collider[self.supports] = not enabled
        self.refresh_regions()
        self.update_region_controls()
        self.applied('Collider contact restored' if enabled else 'Collider contact ignored')
    def clear_regions(self):
        self.motion_examples.clear()
        self.example_feedback.setText('No motion examples applied.')
        self.regions.masks.clear()
        self.region_settings.clear()
        self.named_regions.clear()
        self.named_regions.addItem(NO_REGION)
        self.softness[:]=0;self.rigid[:]=False;self.ignore_collider[:]=False;self.refresh_regions()
    def refresh_regions(self):
        faces=self.inputs['garment'][1];ghost=np.all(self.rigid[faces],axis=1)
        # A face is drawn exactly once, in either category. Keep all source point
        # indices, so picking and the returned geometry retain their correspondence.
        self.data.SetPolys(polydata(self.start,faces[~ghost]).GetPolys())
        self.ghost_data.SetPolys(polydata(self.start,faces[ghost]).GetPolys())
        colors=np.tile(np.asarray([105,120,145],dtype=np.uint8),(len(self.vertices),1))
        if self.clean_view():
            self.data.GetPointData().SetScalars(numpy_to_vtk(colors,deep=True))
            self.support_points.SetData(numpy_to_vtk(self.vertices[:0],deep=True))
            self.update_region_count()
            self.update_visibility()
            self.update_buttons()
            return
        colors[self.suggested >= 0] = [235, 150, 60]
        colors[self.soft_mask()]=[70,170,235]
        colors[self.ignore_collider] = [176, 124, 228]
        self.data.GetPointData().SetScalars(numpy_to_vtk(colors,deep=True))
        self.update_region_count()
        self.update_visibility()
        self.update_buttons()
    def update_visibility(self,*args):
        if not hasattr(self,'ghost_actor'):return
        for key,actors in [('bodies',[self.surface_actor]),('props',self.collider_actors),('ghosts',[self.ghost_actor])]:
            check,slider=self.view_controls[key]
            self.settings.setValue('view/'+key+'/visible',check.isChecked());self.settings.setValue('view/'+key+'/opacity',slider.value())
            for prop in actors:prop.SetVisibility(check.isChecked());prop.GetProperty().SetOpacity(slider.value()/100.)
        self.viewport.GetRenderWindow().Render()

    def update_surface_display(self):
        flat = self.flat_shading.isChecked()
        edges = self.mesh_edges.isChecked()
        for actor in (self.surface_actor, self.ghost_actor):
            style = actor.GetProperty()
            if flat:
                style.SetInterpolationToFlat()
            else:
                style.SetInterpolationToGouraud()
            style.SetEdgeVisibility(edges)
            style.SetEdgeColor(.20, .24, .31)
        self.settings.setValue('view/flat_shading', flat)
        self.settings.setValue('view/mesh_edges', edges)
        self.viewport.GetRenderWindow().Render()
    def mouse_down(self,style,event):
        if self.guide_move.isChecked() and self.guide_move.isEnabled():
            self.begin_guide_drag()
            return
        if self.pick_soft.isChecked() and self.pick_soft.isEnabled():
            self.select_soft_patch()
            return
        if self.pick_suggested.isChecked() and self.pick_suggested.isEnabled():
            self.select_suggested_area()
            return
        if self.collider_paint.isChecked() and self.worker is None and self.export_worker is None:
            self.collider_painting=True;self.brush_collider()
            return
        self.painting=self.paint.isChecked() and self.worker is None and self.export_worker is None
        if self.painting:self.brush()
        else:style.OnLeftButtonDown()
    def mouse_move(self,style,event):
        if self.guide_drag is not None:self.move_guide_drag()
        elif getattr(self,'collider_painting',False):self.brush_collider()
        elif self.painting:self.brush()
        else:style.OnMouseMove()
    def mouse_up(self,style,event):
        if self.guide_drag is not None:self.finish_guide_drag()
        elif getattr(self,'collider_painting',False):self.collider_painting=False
        elif self.painting:self.painting=False
        elif not ((self.pick_soft.isChecked() and self.pick_soft.isEnabled()) or
                  (self.pick_suggested.isChecked() and self.pick_suggested.isEnabled())):style.OnLeftButtonUp()

    def guide_mode_changed(self, enabled):
        if hasattr(self, 'support_points'):
            self.refresh_regions();self.refresh_supports()
        if enabled:
            self.stop_playback()
            self.paint.setChecked(False)
            self.pick_soft.setChecked(False)
            self.pick_suggested.setChecked(False)
        if self.animation is not None:
            self.show_animation_frame(self.frame_slider.value())

    def guide_paint_changed(self, enabled):
        if enabled:
            self.guide_move.setChecked(False)
            self.pick_soft.setChecked(False)
            self.pick_suggested.setChecked(False)

    def soft_patch_mode_changed(self, enabled):
        if enabled:
            self.paint.setChecked(False)
            self.guide_move.setChecked(False)

    def guide_sample(self):
        if self.animation is None:
            return None
        return self.frame_slider.value() - self.animation.start_frame

    def guide_mouse_point(self, depth):
        x, y = self.viewport.GetEventPosition()
        self.renderer.SetDisplayPoint(x, y, depth)
        self.renderer.DisplayToWorld()
        point = np.asarray(self.renderer.GetWorldPoint())
        return point[:3] / point[3] if abs(point[3]) > 1e-12 else None

    def begin_guide_drag(self):
        soft = self.soft_mask()
        for _, region, _, _ in self.motion_examples:
            soft &= ~region
        seeds = np.flatnonzero(self.supports & soft)
        if not len(seeds):
            self.guide_label.setText('Select some soft vertices before dragging.')
            return
        center = self.vertices[seeds].mean(axis=0)
        self.renderer.SetWorldPoint(*center, 1.)
        self.renderer.WorldToDisplay()
        depth = self.renderer.GetDisplayPoint()[2]
        origin = self.guide_mouse_point(depth)
        if origin is None:
            return
        indices, weights = pull_weights(self.vertices, self.rows, seeds, soft, self.guide_radius.value())
        self.guide_drag = (indices, weights, self.vertices.copy(), origin, depth)
        self.update_buttons()

    def move_guide_drag(self):
        indices, weights, original, origin, depth = self.guide_drag
        point = self.guide_mouse_point(depth)
        if point is None:
            return
        positions = original.copy()
        positions[indices] += weights[:, None] * (point - origin)
        self.render(positions)

    def finish_guide_drag(self):
        indices, _, original, _, _ = self.guide_drag
        self.guide_drag = None
        if np.max(np.abs(self.vertices[indices] - original[indices])) > 1e-4:
            key = self.guides.setdefault(self.guide_sample(), {})
            key.update(zip(indices.tolist(), (self.vertices[indices] * .01).tolist()))
            self.mark_guides_changed()
        self.update_buttons()

    def mark_guides_changed(self):
        if self.animation is not None:
            self.guides_dirty = True
            self.status.setText('Frame guides changed. Run the simulation again to update the result.')

    def guide_settings_changed(self):
        if self.guides:
            self.mark_guides_changed()
            self.update_buttons()

    def remove_current_guide(self):
        if self.guides.pop(self.guide_sample(), None) is not None:
            self.mark_guides_changed()
            self.show_animation_frame(self.frame_slider.value())
            self.update_buttons()

    def clear_guides(self):
        if self.guides:
            self.guides.clear()
            self.mark_guides_changed()
            self.show_animation_frame(self.frame_slider.value())
            self.update_buttons()

    def update_guide_controls(self):
        editable = self.animated and self.worker is None and self.export_worker is None and not self.close_pending
        self.frame_number.setEnabled(editable and self.guide_drag is None and not self.play_timer.isActive())
        can_edit_frame = editable and self.animation is not None and not self.play_timer.isActive() and self.guide_drag is None
        self.guide_move.setEnabled(can_edit_frame)
        self.guide_radius.setEnabled(can_edit_frame)
        self.guide_strength.setEnabled(editable and self.guide_drag is None)
        self.guide_lead.setEnabled(editable and self.guide_drag is None)
        self.guide_blend.setEnabled(editable and self.guide_drag is None)
        key = self.guides.get(self.guide_sample())
        self.remove_guide.setEnabled(can_edit_frame and bool(key))
        self.clear_guides_button.setEnabled(editable and bool(self.guides) and self.guide_drag is None)
        if self.animation is not None:
            for widget in self.selection_controls:
                widget.setEnabled(can_edit_frame)
        text = f'{len(self.guides)} guided frames'
        if self.guides:
            start_frame = self.inputs['motion'].start_frame
            frames = [str(start_frame + sample) for sample in sorted(self.guides)]
            text += ' (' + ', '.join(frames[:8]) + (', …' if len(frames) > 8 else '') + ')'
        if key:
            text += f'; {len(key)} vertices on this frame'
        self.guide_label.setText(text)

    def save_motion_example(self):
        self.working('Applying the motion example')
        motion = self.inputs['motion']
        mask = self.supports & self.soft_mask()
        reference = motion.start_frame
        kind = self.example_kind.currentText()
        if kind == NO_TEMPLATE:
            self.remove_motion_example()          # -none- clears the region's example
            return
        if not mask.any():
            self.example_feedback.setText('Select the region and apply softness before saving.')
            return
        try:
            # Example frames default to the Region painting frames, so they are typed once.
            typed = self.example_frames.text().strip() or self.suggest_frames.text()
            frames = sorted({int(value) for value in typed.split(',') if value.strip()})
            if any(not motion.start_frame <= frame <= motion.end_frame for frame in frames):
                raise ValueError()
        except ValueError:
            self.example_feedback.setText('Enter example frame numbers from the captured DAZ timeline.')
            return
        if kind == 'Opening' and not self.inputs['colliders']:
            self.example_feedback.setText('For Opening, include a collider. One wide frame opens by contact; '
                                          'several frames (collider in, out, left, right) are blended by where '
                                          'the collider is.')
            return
        if frames == [reference]:
            self.example_feedback.setText('Choose an example frame different from the reference.')
            return
        index = next((i for i, saved in enumerate(self.motion_examples) if np.array_equal(mask, saved[1])), None)
        if any(i != index and (mask & saved[1]).any() for i, saved in enumerate(self.motion_examples)):
            self.example_feedback.setText('This region overlaps another region with a motion example. Motion examples '
                                          'need separate areas.')
            return
        spec = (kind, mask, reference - motion.start_frame, tuple(frame - motion.start_frame for frame in frames))
        named = next((name for name in self.regions.names() if np.array_equal(self.regions.get(name), self.supports)), None)
        self.remember(example=(kind, reference, ', '.join(map(str, frames))))
        text = f'{named + " · " if named else ""}{kind} · {reference} → {", ".join(map(str, frames))}'
        if index is None:
            self.motion_examples.append(spec)
        else:
            self.motion_examples[index] = spec
        self.example_feedback.setText(f'Applied {text}. {len(self.motion_examples)} region(s) with motion examples.')
        # DAZ example poses often reshape skin beyond the painted area; those vertices follow the capture and
        # pop on the example frames.
        score = deformation_score(motion.body_positions(reference - motion.start_frame),
                                  [motion.body_positions(frame - motion.start_frame) for frame in frames],
                                  self.rows, self.suggest_part.value() * .01)
        if kind == 'Bounce' and len(frames) > 1:
            # Reshaping that is the same in every example is the body's pose (raised arms), not the example.
            poses = [motion.body_positions(frame - motion.start_frame) for frame in frames]
            score = np.minimum(score, deformation_score(poses[0], poses[1:], self.rows, self.suggest_part.value() * .01))
        # Only reshaping joined to the region counts, not other parts of the body.
        moving = score > .0005
        labels, _ = connected_labels(self.rows, moving | mask)
        outside = moving & ~mask & np.isin(labels, np.unique(labels[mask]))
        if outside.any():
            rings, grown = 0, mask.copy()
            while (outside & ~grown).any() and rings < 30:
                grown = grow(grown, self.rows); rings += 1
            self.example_feedback.setText(
                self.example_feedback.text() + f' Warning: these frames also reshape {int(outside.sum())} vertices '
                f'outside the region (up to {score[outside].max() * 1000:.1f} mm); they will jump on those frames. '
                f'Grow the region about {rings} time(s), save it again, apply softness and Apply.')
        self.status.setText('Motion example applied. Run with guides to preview the result.')
        self.update_buttons()

    def remove_motion_example(self):
        mask = self.supports & self.soft_mask()
        kept = [saved for saved in self.motion_examples if not np.array_equal(mask, saved[1])]
        name = self.current_region()
        if name is not None:
            self.region_settings.get(name, {}).pop('example', None)
        if len(kept) != len(self.motion_examples):
            self.motion_examples[:] = kept
            self.example_feedback.setText(f'Example removed. {len(kept)} region(s) with motion examples.')
        else:
            self.example_feedback.setText('This region has no motion example.')
        self.update_buttons()

    def update_example_save(self):
        self.update_region_controls()

    def opening_settings_changed(self):
        self.mark_guides_changed()
        self.update_buttons()

    def loop_settings_changed(self):
        if self.animation is not None:
            self.guides_dirty = True
            self.status.setText('Loop settings changed. Run again to update the animation.')
        self.update_buttons()

    def reset_view(self):
        self.renderer.ResetCamera();self.renderer.ResetCameraClippingRange();self.viewport.GetRenderWindow().Render()

    def update_buttons(self):
        busy=self.export_worker is not None
        dragging = self.guide_drag is not None
        paused=self.worker is not None and self.control is not None and self.control.paused and self.control.ack
        self.run_button.setEnabled(not busy and not dragging and not self.close_pending and (self.worker is not None or bool(self.soft_mask().any())))
        self.actions.run_text='Run with guides' if self.guides or self.motion_examples else 'Run deformation'
        self.actions.show_run_state(self.worker is not None,paused)
        self.send_button.setEnabled(not busy and not dragging and not self.guides_dirty and not self.close_pending and self.ready and (self.worker is None or paused) and
                                   (not self.animated or self.animation is not None))
        self.reset_button.setEnabled(not busy and not dragging and not self.close_pending)
        can_play = self.animation is not None and not busy and not dragging and not self.close_pending
        self.play_button.setEnabled(can_play)
        self.frame_slider.setEnabled(can_play or (self.animated and self.worker is None and not busy and not self.close_pending))
        # Regions and settings stay editable after a run; the next Run starts again from the first frame.
        editable=self.worker is None and not busy
        for widget in self.edits:widget.setEnabled(editable)
        self.update_example_save()
        can_refine_opening = self.worker is None and not busy and not dragging and not self.close_pending and any(
            kind == 'Opening' for kind, _, _, _ in self.motion_examples)
        self.fold_rounding.setEnabled(can_refine_opening)
        self.lip_drag.setEnabled(can_refine_opening)
        self.drag_reach.setEnabled(can_refine_opening)
        loop_editable = self.animated and self.worker is None and not busy and not dragging and not self.close_pending
        self.close_loop.setEnabled(loop_editable)
        self.loop_none.setEnabled(loop_editable)
        ending = self.close_loop.isChecked() or self.loop_motion.isChecked()
        self.loop_frames.setEnabled(loop_editable and ending)
        self.loop_motion.setEnabled(loop_editable)
        self.pin_open_edges.setEnabled(editable and bool(self.open_border.any()))
        self.opening_stretch.setEnabled(editable and bool(self.inputs['colliders']))
        self.part_soft_seal.setEnabled(editable and self.animated and
                                      bool(self.inputs['motion'].samples) and bool(self.ignore_collider.any()))
        self.pin_edge_rings.setEnabled(self.pin_open_edges.isEnabled() and self.pin_open_edges.isChecked())
        self.jiggle.setEnabled(editable and self.animated)
        self.jiggle_preset.setEnabled(editable and self.animated)
        self.preserve_volume.setEnabled(editable and self.animated and self.jiggle.isChecked())
        self.vertical_bands.setEnabled(editable and self.animated and self.jiggle.isChecked())
        self.directional_lag.setEnabled(editable and self.animated and self.jiggle.isChecked())
        for widget in self.dynamic_fields.values():widget.setEnabled(editable and self.animated)
        self.dynamic_fields['jiggle_gravity'].setEnabled(editable and self.animated and self.jiggle.isChecked())
        self.update_guide_controls()

    def run_or_pause(self):
        if not self.run_button.isEnabled():return
        if self.worker is not None:
            self.control.pause(not self.control.paused);self.ready=False
            self.status.set_state('running')
            self.status.setText('Pausing…' if self.control.paused else 'Simulating…');self.update_buttons();return
        self.clear_animation();self.error=False;self.ready=False;self.has_preview=False;self.paint.setChecked(False)
        self.pick_soft.setChecked(False)
        self.pick_suggested.setChecked(False)
        self.render(self.start)
        self.ghost_data.GetPoints().SetData(numpy_to_vtk(self.start,deep=True))
        for i,(points,faces) in enumerate(self.inputs['colliders']):self.collider_data[i].GetPoints().SetData(numpy_to_vtk(points*100,deep=True))
        self.viewport.GetRenderWindow().Render()
        config={key:field.value() for key,field in self.fields.items() if key!='softness'}
        config.update(steps=self.steps.value(),feather=self.feather.value(),softness=self.softness.copy(),rigid=self.rigid.copy())
        config['pin_open_edges'] = self.pin_open_edges.isChecked()
        config['opening_stretch'] = self.opening_stretch.value()
        config['part_soft_seal'] = self.part_soft_seal.isChecked()
        config['pin_edge_rings'] = self.pin_edge_rings.value()
        if any(skip.any() for skip in self.collider_skip):
            faces = self.collider_faces()
            if any(not len(kept) for kept in faces):
                self.status.setText('A collider is painted no-contact everywhere; clear some of its paint first.')
                return
            config['collider_faces'] = faces
        config['ignore_collider'] = self.ignore_collider.copy() if self.ignore_collider.any() else None
        config.update(jiggle=self.jiggle.isChecked(),**{key:field.value() for key,field in self.dynamic_fields.items()})
        config['preserve_volume'] = self.preserve_volume.isChecked()
        config['vertical_bands'] = self.vertical_bands.isChecked()
        config['directional_lag'] = self.directional_lag.isChecked()
        config['collision_passes']=self.collision_accuracy.value()
        config['guide_keys'] = {sample: targets.copy() for sample, targets in self.guides.items()}
        config['guide_strength'] = self.guide_strength.value()
        config['guide_lead'] = self.guide_lead.value()
        config['guide_blend'] = self.guide_blend.value()
        config['motion_examples'] = tuple(self.motion_examples)
        config['bounce_blend'] = self.bounce_blend.value()
        config['fold_rounding'] = self.fold_rounding.value()
        config['lip_drag'] = self.lip_drag.value()
        config['drag_reach'] = self.drag_reach.value()
        config['loop_frames'] = self.loop_frames.value() if (self.close_loop.isChecked() or self.loop_motion.isChecked()) else 0
        config['loop_motion'] = self.loop_motion.isChecked()
        self.control=RunControl();self.worker=SimulationWorker(self.inputs,config,self.control,self)
        self.worker.frame.connect(self.on_frame,Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.paused.connect(self.on_pause,Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(self.finished);self.worker.finished.connect(self.worker.deleteLater)
        self.status.set_state('running');self.status.set_frame()
        self.status.setText('Preparing surface contacts…');self.update_buttons();self.worker.start()
        self.refresh_regions();self.refresh_supports()

    def render(self,vertices):
        self.vertices=np.asarray(vertices).copy();self.data.GetPoints().SetData(numpy_to_vtk(self.vertices,deep=True))
        shown = self.supports & (not self.clean_view())
        self.support_points.SetData(numpy_to_vtk(self.vertices[shown],deep=True));self.viewport.GetRenderWindow().Render()

    def seek_frame(self, frame):
        self.stop_playback()
        self.frame_slider.setValue(frame)

    def sync_frame_number(self, frame):
        with QSignalBlocker(self.frame_number):
            self.frame_number.setValue(frame)

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.stop_playback()
            return
        if not self.play_button.isEnabled():
            return
        self.guide_move.setChecked(False)
        self.paint.setChecked(False)
        self.pick_soft.setChecked(False)
        self.pick_suggested.setChecked(False)
        if self.frame_slider.value() == self.animation.end_frame:
            self.frame_slider.setValue(self.animation.start_frame)
        self.play_start_frame = self.frame_slider.value()
        self.play_start_time = time.monotonic()
        self.play_button.setText('Pause replay')
        self.play_timer.start(max(1, min(16, int(1000 / self.animation.fps))))
        self.refresh_regions();self.refresh_supports()
        self.show_animation_frame(self.play_start_frame)
        self.update_guide_controls()

    def stop_playback(self):
        was_playing = self.play_timer.isActive()
        self.play_timer.stop()
        if was_playing and hasattr(self, 'support_points'):
            self.refresh_regions();self.refresh_supports()
        self.play_button.setText('Replay' if self.frame_slider.value() == self.frame_slider.maximum() else 'Play')
        self.update_guide_controls()

    def advance_playback(self):
        if not self.play_timer.isActive() or self.animation is None:
            return
        target = self.play_start_frame + int((time.monotonic() - self.play_start_time) * self.animation.fps)
        if self.loop_closed:
            start = self.animation.start_frame
            span = self.animation.end_frame - start + 1          # every frame plays; the last leads into frame 0
            self.frame_slider.setValue(start + (target - start) % span)
            return
        self.frame_slider.setValue(min(target, self.animation.end_frame))
        if target >= self.animation.end_frame:
            self.stop_playback()

    def show_animation_frame(self, frame):
        if not self.animated or self.worker is not None or self.export_worker is not None:
            return
        motion = self.inputs['motion']
        sample = frame - motion.start_frame
        source = self.animation is None
        try:
            points = motion.body_positions(sample) * 100. if source else self.animation.get(sample)
            colliders = motion.preview(sample) if source else {i: track.get(sample) for i, track in self.collider_animation.items()}
            for index, vertices in colliders.items():
                self.collider_data[index].GetPoints().SetData(numpy_to_vtk(vertices, deep=True))
        except (OSError, ValueError) as error:
            self.stop_playback()
            self.status.setText(f'Could not replay this frame: {error}')
            return
        show_guide = not source and not self.play_timer.isActive() and (self.guides_dirty or self.guide_move.isChecked())
        guide = self.guides.get(sample) if show_guide else None
        if guide:
            points = points.copy()
            indices = list(guide)
            points[indices] = np.asarray([guide[index] for index in indices]) * 100.
        # Ghost faces use rigid vertices, which already follow the recorded pose.
        self.ghost_data.GetPoints().SetData(numpy_to_vtk(points, deep=True))
        self.render(points)
        self.playback_label.setText('Source' if source else 'Frame')
        if not self.play_timer.isActive():
            self.stop_playback()
    @Slot(object)
    def on_frame(self,payload):
        self.has_preview=True
        for i,points in payload['colliders'].items():self.collider_data[i].GetPoints().SetData(numpy_to_vtk(points,deep=True))
        if payload.get('rest_vertices') is not None:self.ghost_data.GetPoints().SetData(numpy_to_vtk(payload['rest_vertices'],deep=True))
        self.render(payload['vertices'])
        self.saturated=payload.get('saturated',0)
        self.surface_limited=payload.get('surface_limited',False)
        if self.animated:
            motion=self.inputs['motion']
            self.status.set_frame(motion.start_frame+payload['sample'],motion.end_frame)
        prefix=payload.get('phase','Simulating…')
        progress=None
        if self.animated:
            motion=self.inputs['motion'];progress=payload['sample']/max(1,motion.end_frame-motion.start_frame)
        elif not self.animated and self.steps.value():
            progress=payload.get('step',0)/self.steps.value()
        self.status.setText(prefix,progress=progress)
    @Slot()
    def on_pause(self):
        self.status.set_state('paused')
        self.ready=self.has_preview and not self.error
        self.status.setText('Paused.')
        self.update_buttons()
    @Slot(str)
    def on_error(self,message):
        self.error=True;self.status.set_state('failed');self.status.setText('Simulation failed; last valid preview retained.')
        if not self.close_pending:QMessageBox.critical(self,APP_NAME,message)
    @Slot()
    def finished(self):
        worker=self.worker
        if worker.complete and not self.error:
            self.animation=worker.animation
            self.loop_closed = worker.loop_closed
            self.collider_animation=worker.collider_animation
            self.guides_dirty = False
            self.ready=True
            if self.animation is not None:
                self.frame_slider.blockSignals(True)
                self.frame_slider.setRange(self.animation.start_frame, self.animation.end_frame)
                self.frame_slider.setValue(self.animation.start_frame)   # a finished run shows its first frame
                self.frame_slider.blockSignals(False)
                self.sync_frame_number(self.animation.start_frame)
                self.playback_label.setText('Frame')
                self.stop_playback()
            self.status.set_state('finished')
            loop = worker.loop_status.split('. ')[0].rstrip('.')
            self.status.setText(('Animation complete.' if self.animation is not None else 'Simulation complete.') + (f' {loop}.' if loop else ''))
        else:
            if worker.animation is not None:worker.animation.close()
            for track in worker.collider_animation.values():track.close()
        self.worker=None;self.control=None;self.update_buttons()
        if self.animation is not None and worker.complete and not self.error:
            self.show_animation_frame(self.frame_slider.value())
        self.refresh_regions();self.refresh_supports()
        if self.close_pending:self.close()
        elif self.reset_pending:self.reset_pending=False;self.reset_shape()

    def reset_shape(self):
        if self.export_worker is not None:return
        if self.worker is not None:
            self.reset_pending=True;self.worker.requestInterruption();self.control.stop();return
        self.clear_animation();self.ready=False;self.render(self.start)
        self.ghost_data.GetPoints().SetData(numpy_to_vtk(self.start,deep=True))
        for i,(points,faces) in enumerate(self.inputs['colliders']):
            self.collider_data[i].GetPoints().SetData(numpy_to_vtk(points*100,deep=True))
        self.viewport.GetRenderWindow().Render()
        if self.inputs.get('motion'):self.inputs['motion'].reset()
        if self.animated:
            self.frame_slider.setValue(self.inputs['motion'].start_frame)
            self.show_animation_frame(self.frame_slider.value())
        self.status.set_state('idle');self.status.set_frame()
        self.has_preview=False;self.refresh_regions();self.refresh_supports()
        self.status.setText('Reset to starting shape; region masks retained.');self.update_buttons()

    def export(self):
        if not self.send_button.isEnabled():return
        self.stop_playback()
        output=None;overwrite=False
        if not self.inputs.get('return_spec'):
            output=self.project.config.get('output')
            if not output:
                filters='FBX animation (*.fbx)' if self.animated else 'Wavefront OBJ (*.obj);;FBX mesh (*.fbx)'
                output,selected=QFileDialog.getSaveFileName(self,'Export soft body','deformed.fbx' if self.animated else 'deformed.obj',filters,
                                                          options=QFileDialog.Option.DontConfirmOverwrite)
                if not output:return
                if not Path(output).suffix:output+='.fbx' if 'FBX' in selected else '.obj'
            try:
                paths,result,water=output_paths(self.project,output)
                existing=[p for p in paths+[result] if p.exists()]
                overwrite=bool(self.project.config.get('overwrite'))
                if existing and not overwrite:
                    answer=QMessageBox.question(self,'Replace exported files?','Replace these files?\n'+'\n'.join(map(str,existing)))
                    if answer!=QMessageBox.StandardButton.Yes:return
                    overwrite=True
            except Exception as error:QMessageBox.critical(self,APP_NAME,str(error));return
        self.export_worker=ExportWorker(self,output,overwrite)
        self.export_worker.finished.connect(self.exported);self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.status.set_state('sending' if self.inputs.get('return_spec') else 'exporting')
        self.status.setText('Sending to DAZ…' if self.inputs.get('return_spec') else 'Exporting…')
        self.update_buttons();self.export_worker.start()

    @Slot()
    def exported(self):
        worker=self.export_worker;self.export_worker=None
        if worker.error:
            self.close_pending=False;self.status.set_state('failed');self.status.setText('Export failed; the result remains available to retry.')
            QMessageBox.critical(self,APP_NAME,worker.error);self.update_buttons();return
        if self.close_pending or (self.inputs.get('live_session') and self.close_after.isChecked()):
            self.close();return
        if self.inputs.get('live_session'):
            self.status.set_state('sent')
            self.status.setText('Result sent. If the receive prompt is closed, reopen Spin Soft Body Tool in DAZ.')
            self.update_buttons()
            return
        if self.inputs.get('return_spec') or self.project.config.get('callback') or self.project.config.get('after_export'):
            self.close();return
        self.status.set_state('exported')
        self.status.setText('Exported: '+str(worker.result[1]));self.update_buttons()

    def clear_animation(self):
        self.stop_playback()
        self.loop_closed = False
        if self.animation is not None:self.animation.close();self.animation=None
        self.guide_move.setChecked(False)
        for track in self.collider_animation.values():track.close()
        self.collider_animation.clear()
        self.playback_label.setText('Frame')
    def closeEvent(self,event):
        if self.export_worker is not None:
            self.close_pending=True;self.status.setText('Finishing export before closing…');event.ignore();return
        if self.worker is not None:
            self.close_pending=True;self.worker.requestInterruption();self.control.stop()
            self.status.setText('Stopping simulation before closing…');event.ignore();return
        spin_ui.save_window(self,self.settings,self.splitter)
        self.clear_animation();super().closeEvent(event)
