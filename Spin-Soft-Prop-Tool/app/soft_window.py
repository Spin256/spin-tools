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
from PySide6.QtCore import Qt,QThread,Signal,QSettings,Slot
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox,
                               QDoubleSpinBox, QSpinBox, QCheckBox, QSplitter, QMessageBox, QFileDialog)

from branding import APP_NAME, VERSION
import sys as _sys
from pathlib import Path as _Path
_SUITE_APP = str(_Path(__file__).resolve().parents[2] / "app")
if _SUITE_APP not in _sys.path:
    _sys.path.append(_SUITE_APP)          # suite app/: shared interface (spin_ui) and keybindings
import spin_ui
from soft_material import PRESETS
from soft_solver import simulate,RunControl
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
        self.animation=None;self.error=None;self.complete=False;self.loop_status=''
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
                if payload.get('loop_status'):self.loop_status=payload['loop_status']
                if self.animation is not None and not payload.get('preview'):
                    self.animation.put(payload['sample'],payload['vertices'])
                pending=payload
                if last is None or time.monotonic()-last>=1/30:flush()
            simulate(self.inputs['garment'],self.inputs['colliders'],publish,self.isInterruptionRequested,
                     ranges=self.inputs['garment_ranges'],motion=motion,control=self.control,on_pause=paused,**self.settings)
            flush()
            self.complete=not self.isInterruptionRequested() and not self.control.stopped
        except Exception as error:
            flush();self.error=str(error);self.failed.emit(self.error)


class ExportWorker(QThread):
    def __init__(self,window,output=None,overwrite=False):
        super().__init__(window)
        self.vertices=window.vertices.copy();self.ranges=window.inputs['garment_ranges']
        self.animation=window.animation;self.spec=window.inputs.get('return_spec');self.project=window.project
        if self.spec and self.animation is not None:     # DAZ return format
            self.spec=dict(self.spec,animation_format=window.return_format.currentData())
        self.live_session=window.inputs.get('live_session')
        self.output,self.overwrite=output,overwrite;self.error=None;self.result=None
    def run(self):
        try:
            if self.spec:
                if self.live_session is None:
                    raise ValueError('This capture has no Spin Tools receive session. Recapture it from DAZ.')
                # The helper stays open: publish a complete revision for the Spin Tools receive window.
                publish_revision(self.live_session,self.spec,self.write)
            else:self.result=export_result(self.project,self.output,self.vertices,self.animation,overwrite=self.overwrite)
        except Exception as error:self.error=str(error)
    def write(self,spec):
        if self.animation is not None:write_animation_result(spec,self.animation,self.ranges)
        else:write_static_result(spec,self.vertices,self.ranges)


class SoftWindow(QWidget):
    def __init__(self,inputs,project=None):
        super().__init__()
        self.inputs,self.project=inputs,project
        self.vertices=inputs['garment'][0]*100
        self.start=self.vertices.copy();self.supports=np.zeros(len(self.vertices),dtype=bool)
        self.worker=None;self.export_worker=None;self.control=None;self.animation=None
        self.close_pending=False;self.reset_pending=False;self.ready=False;self.error=False;self.painting=False;self.has_preview=False
        self.animated=inputs.get('motion') is not None
        self.settings=QSettings('Spin256',APP_NAME)
        self.setWindowTitle(APP_NAME+(' — DAZ bridge' if inputs.get('return_spec') else ' — Standalone'))
        self.resize(1200,800)
        layout=QHBoxLayout(self);self.splitter=QSplitter(Qt.Orientation.Horizontal);layout.addWidget(self.splitter)
        panel=QWidget();controls=QVBoxLayout(panel)
        label=QLabel('Soft props: '+inputs['garment_label']+'\nCollision objects: '+', '.join(inputs['collision_labels']))
        label.setWordWrap(True);label.setTextFormat(Qt.TextFormat.PlainText);controls.addWidget(label)
        mode=QLabel('Animation output — follows the captured timeline' if self.animated else 'Static output — current pose')
        mode.setWordWrap(True);controls.addWidget(mode)
        self.edits=[]
        self.shared_sections=spin_ui.Sections(controls,self.settings);self.sections=self.shared_sections.items
        def group(title):
            return self.shared_sections.form(title,True)
        def number(form,label,low,high,value,step=1,decimals=2):
            field=QDoubleSpinBox();field.setRange(low,high);field.setDecimals(decimals);field.setSingleStep(step);field.setValue(value)
            form.addRow(label,field);self.edits.append(field);return field
        material=group('Material')
        self.preset=QComboBox();self.preset.addItems(PRESETS);material.addRow('Starting material',self.preset);self.edits.append(self.preset)
        self.fields={}
        for key,title,lo,hi,step in [('firmness','Shape firmness (%)',0,100,5),('volume','Compression firmness (%)',0,100,5),
                    ('tension','Surface tension (%)',0,100,5),('clearance','Contact clearance (mm)',0,10,.1),
                    ('damping','Damping (/s)',0,50,.5),('recovery','Shape recovery (Hz)',0,5,.1),('density','Density (kg/m³)',10,1500,10)]:
            self.fields[key]=number(material,title,lo,hi,PRESETS['Soft foam'][key],step)
        self.preset.currentTextChanged.connect(self.apply_material)
        note=QLabel('Compression firmness resists loss of volume. Surface tension smooths dents and pulls neighboring areas along; zero disables it. Clearance sets the contact gap. Low damping allows rebound. Presets are artistic starting points.')
        note.setWordWrap(True);controls.addWidget(note)
        support=group('Fixed supports')
        self.bottom=QCheckBox('Hold underside of each prop');self.bottom.setChecked(True);support.addRow(self.bottom);self.edits.append(self.bottom)
        self.paint=QCheckBox('Paint supports (left drag)');support.addRow(self.paint);self.edits.append(self.paint)
        self.erase=QCheckBox('Erase instead of add');support.addRow(self.erase);self.edits.append(self.erase)
        self.radius=number(support,'Brush radius (cm)',.2,100,5,.5)
        self.support_label=QLabel('0 painted support vertices');support.addRow(self.support_label)
        clear=QPushButton('Clear painted supports');clear.clicked.connect(self.clear_supports);support.addRow(clear);self.edits.append(clear)
        simulation=group('Simulation')
        self.duration=number(simulation,'Static settling (s)',.1,120,3,.5)
        self.duration.setVisible(not self.animated)
        simulation.labelForField(self.duration).setVisible(not self.animated)
        self.initial=number(simulation,'Initial pose settling (s)',0,10,.25,.1);self.initial.setVisible(self.animated)
        simulation.labelForField(self.initial).setVisible(self.animated)
        self.resolution=QSpinBox();self.resolution.setRange(4,24);self.resolution.setValue(10)
        simulation.addRow('Cage resolution',self.resolution);self.edits.append(self.resolution)
        self.iterations=QSpinBox();self.iterations.setRange(1,12);self.iterations.setValue(4)
        simulation.addRow('Solver iterations',self.iterations);self.edits.append(self.iterations)
        # Seamless loop: continue past the last frame until the motion meets the first run.
        self.loop=QCheckBox('Loop animation (carry motion)')
        self.loop.setToolTip('Keep simulating past the last frame into frames 1, 2, ... until the motion meets the first '
                             'run, so the animation repeats without restarting from rest. The last frame leads into '
                             'frame 0: end the animation one pose before its first.')
        self.loop.setChecked(self.settings.value('loop/enabled',False,type=bool))
        self.loop.toggled.connect(lambda value:self.settings.setValue('loop/enabled',value))
        self.edits.append(self.loop)
        self.loop_frames=QSpinBox();self.loop_frames.setRange(2,60);self.loop_frames.setValue(8)
        self.loop_frames.setToolTip('If the motion has not met the first run by the middle of the animation, it is steered '
                                    'onto the closest frame over this many frames.')
        self.edits.append(self.loop_frames)
        # Playback of a finished animation.
        self.playback=spin_ui.PlaybackBar(self.show_recorded_frame)
        # DAZ return format for an animation: DUF by default, FBX selectable.
        self.return_format=spin_ui.return_format_box(self.settings)
        view_box=self.shared_sections.add('View',True)
        self.view=spin_ui.ViewControls(view_box,self.settings,'soft props',self.update_view,avatar_name='collision objects',avatar_opacity=60,tool='soft_prop')
        self.show_colliders=self.view.show_avatar
        self.view.reset_view_button.clicked.connect(self.reset_view)
        self.view_box=view_box
        if self.animated:
            # Animation (as Drape): loop, playback and DAZ return format at the bottom.
            animation=group('Animation')
            animation.addRow(self.loop)
            animation.addRow('Loop landing frames',self.loop_frames)
            animation.addRow('Playback',self.playback)
            animation.addRow('DAZ return format',self.return_format)
        self.actions=spin_ui.ActionRow(self.settings,run_text='Run deformation',tool='soft_prop',
                                       send_text=('Send animation to DAZ' if self.animated else 'Send static result to DAZ')
                                       if inputs.get('return_spec') else ('Export animation…' if self.animated else 'Export static result…'))
        self.run_button,self.reset_button,self.send_button,self.close_after=(self.actions.run_button,self.actions.reset_button,
                                                                            self.actions.send_button,self.actions.close_after)
        self.run_button.clicked.connect(self.run_or_pause);self.reset_button.clicked.connect(self.reset_shape)
        self.reset_button.setToolTip('Reset to the starting shape. Painted supports are kept.')
        self.send_button.clicked.connect(self.export)
        self.close_after_key=None;self.actions.remember=self.remember_close_after
        config=project.config if project is not None else {}
        bridge=bool(config.get('callback') or config.get('after_export'))
        if inputs.get('return_spec'):
            context,text,default='daz','Close after sending',False
            tip='Off: minimize after each send so you can inspect the result in DAZ and send again. On: exit once the result is written.'
        elif bridge:
            context,text,default='bridge','Close after send',True
            tip='On: exit after the export and its bridge callback start. Off: minimize so you can send again; each export runs the callback again.'
        else:
            context,text,default='export','Close after export',False
            tip='On: close once the files are written.'
        override=config.get('close_after')
        self.close_after.setText(text);self.close_after.setToolTip(tip)
        self.close_after.setChecked(override if isinstance(override,bool) else
                                    self.settings.value('close_after/'+context,default,type=bool))
        self.close_after_key=None if isinstance(override,bool) else 'close_after/'+context
        self.bridge=bridge
        spin_ui.note('Best for solid cushions/mattresses. The internal box cage approximates the volume; hollows, thin shells and separate soft-prop mutual collisions are not modelled.',controls)
        controls.addStretch()
        column,self.copyright_footer=spin_ui.control_column(panel, self.actions, 340, version=VERSION, tool_folder=Path(__file__).resolve().parent.parent)
        self.splitter.addWidget(column)
        self.viewport=QVTKRenderWindowInteractor(self)
        holder,self.status=spin_ui.viewport_with_status(self.viewport)
        self.status.setText('Ready — material and support changes take effect on a new run.')
        self.splitter.addWidget(holder);self.splitter.setStretchFactor(1,1)
        self.splitter.setSizes([380,820])
        spin_ui.remember_window(self,self.settings,self.splitter)
        self.renderer=spin_ui.style_renderer(vtk.vtkRenderer());self.viewport.GetRenderWindow().AddRenderer(self.renderer)
        self.data=polydata(self.vertices,inputs['garment'][1]);self.surface_actor=actor(self.renderer,self.data,spin_ui.OBJECT_COLOR)
        self.collider_data=[];self.collider_actors=[]
        for points,faces in inputs['colliders']:
            data=polydata(points*100,faces);self.collider_data.append(data)
            self.collider_actors.append(actor(self.renderer,data,spin_ui.AVATAR_COLOR,.6))
        self.view.apply([self.surface_actor],self.collider_actors)
        self.view.lighting.attach(self.renderer,self.viewport.GetRenderWindow().Render)
        self.support_data=vtk.vtkPolyData();self.support_points=vtk.vtkPoints();self.support_data.SetPoints(self.support_points)
        glyph=vtk.vtkVertexGlyphFilter();glyph.SetInputData(self.support_data)
        mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(glyph.GetOutputPort())
        self.support_actor=vtk.vtkActor();self.support_actor.SetMapper(mapper);self.support_actor.GetProperty().SetColor(1,.5,.1)
        self.support_actor.GetProperty().SetPointSize(5);self.support_actor.PickableOff();self.renderer.AddActor(self.support_actor)
        self.picker=vtk.vtkCellPicker();self.picker.PickFromListOn();self.picker.AddPickList(self.surface_actor)
        self.style=vtk.vtkInteractorStyleTrackballCamera();self.viewport.SetInteractorStyle(self.style)
        self.style.AddObserver('LeftButtonPressEvent',self.mouse_down)
        self.style.AddObserver('MouseMoveEvent',self.mouse_move)
        self.style.AddObserver('LeftButtonReleaseEvent',self.mouse_up)
        self.camera=spin_ui.ViewportCamera(self.renderer,self.viewport.GetRenderWindow().Render,
                                           lambda:[self.surface_actor]+self.collider_actors)
        self.keys=spin_ui.Shortcuts(self,'soft_prop',{
            'pause_resume':self.run_or_pause,'run_simulation':self.run_or_pause,'reset_view':self.reset_view,
            'send_to_daz':self.export,'reset_simulation':lambda:self.reset_button.isEnabled() and self.reset_shape(),
            'cancel_interaction':lambda:self.paint.setChecked(False),
            'show_avatar':self.show_colliders.toggle},camera=self.camera)
        self.shortcuts=self.keys.shortcuts
        self.view_box.addWidget(self.keys.help_button())
        self.view_bar = spin_ui.add_view_toolbar(self.keys, self.status)   # orbit/pan/zoom/reset buttons
        self.reset_view();self.update_buttons()

    def apply_material(self,name):
        for key,value in PRESETS[name].items():self.fields[key].setValue(value)

    def clear_supports(self):self.supports[:]=False;self.refresh_supports()
    def refresh_supports(self):
        self.support_points.SetData(numpy_to_vtk(self.vertices[self.supports],deep=True))
        self.support_label.setText(f'{np.count_nonzero(self.supports)} painted support vertices')
        self.viewport.GetRenderWindow().Render()
    def brush(self):
        x,y=self.viewport.GetEventPosition()
        if self.picker.Pick(x,y,0,self.renderer):
            p=np.asarray(self.picker.GetPickPosition())
            selected=np.linalg.norm(self.vertices-p,axis=1)<=self.radius.value()
            self.supports[selected]=not self.erase.isChecked();self.refresh_supports()
    def mouse_down(self,style,event):
        self.painting=self.paint.isChecked() and self.worker is None and self.export_worker is None
        if self.painting:self.brush()
        else:style.OnLeftButtonDown()
    def mouse_move(self,style,event):
        if self.painting:self.brush()
        else:style.OnMouseMove()
    def mouse_up(self,style,event):
        if self.painting:self.painting=False
        else:style.OnLeftButtonUp()
    def update_view(self):
        if hasattr(self,'collider_actors'):
            self.view.apply([self.surface_actor],self.collider_actors);self.viewport.GetRenderWindow().Render()
    def reset_view(self):
        self.renderer.ResetCamera();self.renderer.ResetCameraClippingRange();self.viewport.GetRenderWindow().Render()

    def update_buttons(self):
        busy=self.export_worker is not None
        paused=self.worker is not None and self.control is not None and self.control.paused and self.control.ack
        self.run_button.setEnabled(not busy and not self.close_pending)
        self.actions.show_run_state(self.worker is not None,paused)
        self.send_button.setEnabled(not busy and self.ready and (self.worker is None or paused) and
                                   (not self.animated or self.animation is not None))
        self.reset_button.setEnabled(not busy and not self.close_pending)
        for widget in self.edits:widget.setEnabled(self.worker is None and not busy)

    def run_or_pause(self):
        if not self.run_button.isEnabled():return
        if self.worker is not None:
            self.control.pause(not self.control.paused);self.ready=False
            self.status.set_state('running')
            self.status.setText('Pausing…' if self.control.paused else 'Simulating…');self.update_buttons();return
        self.clear_animation();self.error=False;self.ready=False;self.has_preview=False;self.paint.setChecked(False)
        self.render(self.start)
        config={key:field.value() for key,field in self.fields.items()}
        config.update(duration=self.duration.value(),initial=self.initial.value(),resolution=self.resolution.value(),
                      iterations=self.iterations.value(),bottom=self.bottom.isChecked(),painted=self.supports.copy(),
                      loop=self.animated and self.loop.isChecked(),loop_frames=self.loop_frames.value())
        self.control=RunControl();self.worker=SimulationWorker(self.inputs,config,self.control,self)
        self.worker.frame.connect(self.on_frame,Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.paused.connect(self.on_pause,Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(self.finished);self.worker.finished.connect(self.worker.deleteLater)
        self.status.set_state('running');self.status.set_frame()
        self.status.setText('Preparing GPU cage…');self.update_buttons();self.worker.start()

    def show_recorded_frame(self,frame):
        """Draw one recorded DAZ frame of the finished animation, with the collision objects at that frame."""
        if self.animation is None:
            return
        motion=self.inputs['motion'];sample=frame-motion.start_frame
        try:vertices=np.asarray(self.animation.get(sample),dtype=float)
        except (OSError,ValueError):return
        for i,points in motion.preview(sample).items():self.collider_data[i].GetPoints().SetData(numpy_to_vtk(points,deep=True))
        self.render(vertices)
        self.status.set_frame(frame,motion.end_frame)

    def render(self,vertices):
        self.vertices=np.asarray(vertices).copy();self.data.GetPoints().SetData(numpy_to_vtk(self.vertices,deep=True))
        self.support_points.SetData(numpy_to_vtk(self.vertices[self.supports],deep=True));self.viewport.GetRenderWindow().Render()
    @Slot(object)
    def on_frame(self,payload):
        self.has_preview=True
        for i,points in payload['colliders'].items():self.collider_data[i].GetPoints().SetData(numpy_to_vtk(points,deep=True))
        self.render(payload['vertices'])
        if self.animated:
            motion=self.inputs['motion']
            self.status.set_frame(motion.start_frame+payload['sample'],motion.end_frame)
            self.status.setText(payload.get('phase','Simulating…'),
                                progress=payload['sample']/max(1,motion.end_frame-motion.start_frame))
        else:
            duration=self.duration.value()
            self.status.setText(f"Settling {payload['time']:.1f} s",progress=min(1.,payload['time']/max(duration,1e-6)))
    @Slot()
    def on_pause(self):
        self.ready=self.has_preview and not self.error
        self.status.set_state('paused')
        self.status.setText('Paused — resume to settle further, or export the displayed result.' if self.ready else 'Paused — resume to begin simulation.')
        self.update_buttons()
    @Slot(str)
    def on_error(self,message):
        self.error=True;self.status.set_state('failed');self.status.setText('Simulation failed; last valid preview retained.')
        if not self.close_pending:QMessageBox.critical(self,APP_NAME,message)
    @Slot()
    def finished(self):
        worker=self.worker
        if worker.complete and not self.error:
            self.animation=worker.animation;self.ready=True;self.status.set_state('finished')
            loop=getattr(worker,'loop_status','').split(' (')[0]
            if self.animation is not None:
                motion=self.inputs['motion']
                self.playback.set_animation(motion.start_frame,motion.end_frame,motion.fps,looped=bool(loop))
            self.status.setText('Animation complete.'+(f' {loop}.' if loop else ''))
        elif worker.animation is not None:worker.animation.close()
        self.worker=None;self.control=None;self.update_buttons()
        if self.close_pending:self.close()
        elif self.reset_pending:self.reset_pending=False;self.reset_shape()

    def reset_shape(self):
        if self.export_worker is not None:return
        if self.worker is not None:
            self.reset_pending=True;self.worker.requestInterruption();self.control.stop();return
        self.clear_animation();self.ready=False;self.render(self.start)
        for i,(points,faces) in enumerate(self.inputs['colliders']):
            self.collider_data[i].GetPoints().SetData(numpy_to_vtk(points*100,deep=True))
        self.viewport.GetRenderWindow().Render()
        if self.inputs.get('motion'):self.inputs['motion'].reset()
        self.status.set_state('idle');self.status.set_frame()
        self.status.setText('Reset to starting shape; painted supports retained.');self.update_buttons()

    def export(self):
        if not self.send_button.isEnabled():return
        output=None;overwrite=False
        if not self.inputs.get('return_spec'):
            output=self.project.config.get('output')
            if not output:
                filters='FBX animation (*.fbx)' if self.animated else 'Wavefront OBJ (*.obj);;FBX mesh (*.fbx)'
                output,selected=QFileDialog.getSaveFileName(self,'Export soft prop','deformed.fbx' if self.animated else 'deformed.obj',filters,
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

    def remember_close_after(self,checked):
        if self.close_after_key:self.settings.setValue(self.close_after_key,checked)

    @Slot()
    def exported(self):
        worker=self.export_worker;self.export_worker=None
        if worker.error:
            self.close_pending=False;self.status.set_state('failed');self.status.setText('Export failed; the result remains available to retry.')
            QMessageBox.critical(self,APP_NAME,worker.error);self.update_buttons();return
        if self.close_pending or self.close_after.isChecked():
            self.close();return
        if self.inputs.get('return_spec'):
            self.status.set_state('sent')
            self.status.setText('The Spin Tools receive window imports it; adjust and send again to replace it.')
        else:self.status.set_state('exported');self.status.setText('Exported: '+str(worker.result[1]))
        self.update_buttons()
        # Control passes to DAZ or the receiving application; restore to refine and send again.
        if self.inputs.get('return_spec') or self.bridge:self.showMinimized()

    def clear_animation(self):
        if hasattr(self,'playback'):self.playback.clear()
        if self.animation is not None:self.animation.close();self.animation=None
    def closeEvent(self,event):
        if self.export_worker is not None:
            self.close_pending=True;self.status.setText('Finishing export before closing…');event.ignore();return
        if self.worker is not None:
            self.close_pending=True;self.worker.requestInterruption();self.control.stop()
            self.status.setText('Stopping simulation before closing…');event.ignore();return
        spin_ui.save_window(self,self.settings,self.splitter)
        self.clear_animation();super().closeEvent(event)
