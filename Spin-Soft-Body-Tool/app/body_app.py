# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Standalone/two-role loader and DAZ bridge entry for Spin Soft Body Tool."""
import argparse
import json
from pathlib import Path
import sys
from PySide6.QtCore import QThread,Signal,QTimer,QSettings
from PySide6.QtWidgets import (QApplication,QDialog,QVBoxLayout,QHBoxLayout,QListWidget,QPushButton,
    QGroupBox,QFormLayout,QComboBox,QDoubleSpinBox,QLineEdit,QLabel,QFileDialog,QMessageBox)
from branding import APP_NAME,footer
from standalone_project import prepare_project
from drape_session import load_session
from body_window import SoftWindow
from live_return import release_capture


def normalize_config(config):
    config=dict(config)
    if config.get('output_mode','static') not in ('static','animation'):raise ValueError('Choose static or animation output.')
    config['avatars']=config.get('colliders',config.get('avatars',[]))
    config['garments']=config.get('soft_bodies',config.get('garments',[]))
    config['sequences']=config.get('collider_sequences',config.get('sequences',[]))
    config['props']=[]
    if config.get('sequences') and config.get('output_mode','static')!='animation':
        raise ValueError('An OBJ sequence requires animation output.')
    return config


def read_project(path):
    path=Path(path).resolve();data=json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict) or data.get('version',1)!=1:raise ValueError('Unsupported project configuration.')
    def resolve(value):
        p=Path(value)
        return str((path.parent/p).resolve()) if not p.is_absolute() else str(p)
    for key in ('colliders','soft_bodies','collider_sequences'):
        data[key]=[resolve(value) for value in data.get(key,[])]
    for key in ('output','result','after_export'):
        if data.get(key):data[key]=resolve(data[key])
    return data


class PrepareWorker(QThread):
    progress=Signal(str)
    def __init__(self,config,parent=None):
        super().__init__(parent);self.config=config;self.project=None;self.error=None
    def run(self):
        try:self.project=prepare_project(normalize_config(self.config),self.progress.emit,self.isInterruptionRequested)
        except Exception as error:self.error=str(error)


class LoadDialog(QDialog):
    def __init__(self,config=None):
        super().__init__();self.config=dict(config or {});self.worker=None;self.project=None;self.close_pending=False
        self.settings=QSettings('Spin256',APP_NAME+'/Loader')
        self.setWindowTitle('Open '+APP_NAME);self.resize(850,650)
        if self.settings.value('geometry'):self.restoreGeometry(self.settings.value('geometry'))
        layout=QVBoxLayout(self)
        text=QLabel('Choose the soft bodies and optional rigid collision objects. For self-contact, load just the body and paint receiving/ghost regions in the next window. OBJ uses Y-up world coordinates; FBX axes/units are converted automatically.')
        text.setWordWrap(True);layout.addWidget(text)
        row=QHBoxLayout();layout.addLayout(row);self.lists={}
        for key,title in [('colliders','Collision objects'),('soft_bodies','Soft bodies to deform')]:
            box=QGroupBox(title);column=QVBoxLayout(box);items=QListWidget();items.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
            for value in self.config.get(key,[]):items.addItem(value)
            column.addWidget(items);self.lists[key]=items
            add=QPushButton('Add OBJ/FBX…');add.clicked.connect(lambda checked=False,k=key:self.add(k));column.addWidget(add)
            remove=QPushButton('Remove selected');remove.clicked.connect(lambda checked=False,k=key:self.remove(k));column.addWidget(remove)
            row.addWidget(box)
        self.sequences=list(self.config.get('collider_sequences',[]))
        self.sequence_label=QLabel();self.sequence_label.setWordWrap(True);layout.addWidget(self.sequence_label)
        seqrow=QHBoxLayout();layout.addLayout(seqrow)
        seq=QPushButton('Add collision OBJ sequence folder…');seq.clicked.connect(self.add_sequence);seqrow.addWidget(seq)
        clear=QPushButton('Clear sequence folders');clear.clicked.connect(self.clear_sequences);seqrow.addWidget(clear)
        self.sequence_buttons=[seq,clear]
        form=QFormLayout();layout.addLayout(form)
        self.units=QComboBox()
        for label,value in [('Centimeters','cm'),('Meters','m'),('Millimeters','mm')]:self.units.addItem(label,value)
        self.units.setCurrentIndex(max(0,self.units.findData(self.config.get('units','cm'))));form.addRow('OBJ units',self.units)
        self.mode=QComboBox();self.mode.addItem('Static current pose','static');self.mode.addItem('Baked FBX animation','animation')
        self.mode.setCurrentIndex(1 if self.config.get('output_mode')=='animation' else 0);form.addRow('Output',self.mode)
        self.fps=QDoubleSpinBox();self.fps.setRange(.1,240);self.fps.setDecimals(3);self.fps.setValue(self.config.get('fps',30));form.addRow('Sample frame rate',self.fps)
        self.take=QLineEdit(self.config.get('take') or '');self.take.setPlaceholderText('First take (leave blank)');form.addRow('FBX take',self.take)
        self.status=QLabel('Animation uses sampled body/collider motion and outputs a baked FBX; paint regions in the next window.');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.load_button=QPushButton('Open scene');self.load_button.clicked.connect(self.load);layout.addWidget(self.load_button)
        layout.addWidget(footer(self));self.refresh_sequences()

    def add(self,key):
        paths,_=QFileDialog.getOpenFileNames(self,'Choose '+key,'','Meshes (*.obj *.fbx)')
        current={self.lists[key].item(i).text() for i in range(self.lists[key].count())}
        for path in paths:
            if path not in current:self.lists[key].addItem(path)
    def remove(self,key):
        for item in self.lists[key].selectedItems():self.lists[key].takeItem(self.lists[key].row(item))
    def add_sequence(self):
        folder=QFileDialog.getExistingDirectory(self,'Collision OBJ sequence folder')
        if folder and folder not in self.sequences:self.sequences.append(folder);self.refresh_sequences()
    def clear_sequences(self):self.sequences=[];self.refresh_sequences()
    def refresh_sequences(self):self.sequence_label.setText('Collision sequence folders: '+(', '.join(self.sequences) or 'none'))
    def load(self):
        if self.worker:return
        config=dict(self.config)
        for key,items in self.lists.items():config[key]=[items.item(i).text() for i in range(items.count())]
        config.update(collider_sequences=self.sequences,units=self.units.currentData(),output_mode=self.mode.currentData(),
                      fps=self.fps.value(),take=self.take.text().strip() or None)
        self.worker=PrepareWorker(config,self);self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self.loaded);self.worker.finished.connect(self.worker.deleteLater)
        for widget in self.findChildren(QGroupBox)+self.sequence_buttons+[self.load_button,self.units,self.mode,self.fps,self.take]:widget.setEnabled(False)
        self.status.setText('Preparing scene…');self.worker.start()
    def loaded(self):
        worker,self.worker=self.worker,None
        if self.close_pending:
            if worker.project:worker.project.close()
            self.reject();return
        if worker.error:
            QMessageBox.critical(self,APP_NAME,worker.error)
            for widget in self.findChildren(QGroupBox)+self.sequence_buttons+[self.load_button,self.units,self.mode,self.fps,self.take]:widget.setEnabled(True)
            self.status.setText('Could not load scene.');return
        self.project=worker.project
        self.settings.setValue('geometry',self.saveGeometry());self.accept()
    def reject(self):
        if self.worker:
            self.close_pending=True;self.worker.requestInterruption();self.status.setText('Finishing the current import before closing…');return
        super().reject()
    def closeEvent(self,event):
        if self.worker:
            self.close_pending=True;self.worker.requestInterruption();event.ignore();return
        super().closeEvent(event)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument('--session',help='DAZ bridge session')
    parser.add_argument('--project',help='Standalone/custom bridge JSON')
    parser.add_argument('--collider',action='append')
    parser.add_argument('--soft-body',action='append')
    parser.add_argument('--collider-sequence',action='append')
    parser.add_argument('--output-mode',choices=('static','animation'))
    parser.add_argument('--fps',type=float)
    parser.add_argument('--units',choices=('cm','m','mm'))
    parser.add_argument('--take');parser.add_argument('--output');parser.add_argument('--result');parser.add_argument('--after-export')
    parser.add_argument('--overwrite',action='store_true',default=None)
    args=parser.parse_args(argv)
    if args.session and any(value is not None for key,value in vars(args).items() if key!='session'):
        parser.error('A DAZ session cannot be combined with standalone inputs/options.')
    config=read_project(args.project) if args.project else {}
    for key,target in [('collider','colliders'),('soft_body','soft_bodies'),('collider_sequence','collider_sequences')]:
        if getattr(args,key) is not None:config[target]=[str(Path(p).resolve()) for p in getattr(args,key)]
    for key in ('output_mode','fps','units','take','overwrite'):
        if getattr(args,key) is not None:config[key]=getattr(args,key)
    for key in ('output','result','after_export'):
        if getattr(args,key):config[key]=str(Path(getattr(args,key)).resolve())
    return args.session,config


def main(argv=None):
    from standalone_runtime import ensure_gui_streams
    ensure_gui_streams()
    app=QApplication([]);app.setQuitOnLastWindowClosed(False);project=None;live_session=None
    suite_app = str(Path(__file__).resolve().parents[2] / 'app')   # suite app/: taskbar icon
    if suite_app not in sys.path:
        sys.path.append(suite_app)
    import spin_icon
    spin_icon.set_app_icon(app, 'soft_body')
    try:
        session,config=parse_args(argv)
        if session:
            inputs=load_session(session)
            if json.loads(Path(session).read_text(encoding='utf-8-sig')).get('live_return'):
                live_session = inputs['live_session'] = Path(session)
            if not inputs.get('return_spec'):raise ValueError('This DAZ session has no return information.')
            if (inputs.get('motion') is not None)!=(inputs.get('output_mode')=='animation'):
                raise ValueError('Timeline sessions require animation output; capture a current pose for static output.')
        else:
            loader=LoadDialog(config)
            if config.get('soft_bodies'):QTimer.singleShot(0,loader.load)
            if loader.exec()!=QDialog.DialogCode.Accepted:return 0
            project=loader.project;inputs=project.inputs
        window=SoftWindow(inputs,project)
        app.setQuitOnLastWindowClosed(True);window.show();window.viewport.Initialize()
        return app.exec()
    except Exception as error:
        QMessageBox.critical(None,APP_NAME,str(error));return 1
    finally:
        if project:project.close()
        if live_session:
            try:
                release_capture(live_session)
            except (OSError, ValueError):
                # A cleanup failure must not invalidate a completed revision.
                pass


if __name__=='__main__':sys.exit(main())
