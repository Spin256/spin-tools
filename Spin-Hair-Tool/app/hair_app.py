# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Spin Hair Tool entry point: a DAZ capture (--session) or standalone OBJ/FBX files."""
import argparse
import json
from pathlib import Path
import sys

from PySide6.QtCore import QSettings, QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QVBoxLayout)

from branding import APP_NAME, footer

SCALP_NOTE = ("Hair is card hair (strips of quads). A surface named scalp or cap is held still and the hair roots on "
              "it; without one, the hair roots on the avatar's skin.")


def read_project(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or data.get('version', 1) != 1:
        raise ValueError('Unsupported project configuration.')

    def resolve(value):
        value = Path(value)
        return str(value if value.is_absolute() else (path.parent / value).resolve())
    for key in ('hair', 'colliders'):
        data[key] = [resolve(value) for value in data.get(key, [])]
    for key in ('avatar', 'avatar_sequence', 'output', 'result', 'after_export'):
        if data.get(key):
            data[key] = resolve(data[key])
    return data


class PrepareWorker(QThread):
    progress = Signal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config, self.project, self.error = config, None, None

    def run(self):
        from hair_standalone import prepare
        try:
            self.project = prepare(self.config, self.progress.emit, self.isInterruptionRequested)
        except Exception as error:
            self.error = str(error)


class LoadDialog(QDialog):
    def __init__(self, config=None):
        super().__init__()
        self.config, self.worker, self.project, self.close_pending = dict(config or {}), None, None, False
        self.settings = QSettings('Spin256', APP_NAME + '/Loader')
        self.setWindowTitle('Open ' + APP_NAME)
        self.resize(850, 620)
        if self.settings.value('geometry'):
            self.restoreGeometry(self.settings.value('geometry'))
        layout = QVBoxLayout(self)
        text = QLabel('Choose the hair, the avatar it grows from and any other collision objects. OBJ uses Y-up world '
                      'coordinates in the units below; FBX axes and units are converted. ' + SCALP_NOTE)
        text.setWordWrap(True)
        layout.addWidget(text)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.lists = {}
        for key, title in (('hair', 'Hair'), ('colliders', 'Other collision objects')):
            box = QGroupBox(title)
            column = QVBoxLayout(box)
            items = QListWidget()
            items.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
            for value in self.config.get(key, []):
                items.addItem(value)
            column.addWidget(items)
            add = QPushButton('Add OBJ/FBX…')
            add.clicked.connect(lambda checked=False, k=key: self.add(k))
            remove = QPushButton('Remove selected')
            remove.clicked.connect(lambda checked=False, k=key: self.remove(k))
            column.addWidget(add)
            column.addWidget(remove)
            self.lists[key] = items
            row.addWidget(box)
        form = QFormLayout()
        layout.addLayout(form)
        avatar_row = QHBoxLayout()
        self.avatar = QLineEdit(self.config.get('avatar') or self.config.get('avatar_sequence') or '')
        self.avatar.setPlaceholderText('Avatar OBJ or FBX, or a folder of OBJ frames')
        pick = QPushButton('File…')
        pick.clicked.connect(self.pick_avatar)
        folder = QPushButton('Folder…')
        folder.clicked.connect(self.pick_sequence)
        for widget in (self.avatar, pick, folder):
            avatar_row.addWidget(widget)
        form.addRow('Avatar', avatar_row)
        self.units = QComboBox()
        for label, value in (('Centimeters', 'cm'), ('Meters', 'm'), ('Millimeters', 'mm')):
            self.units.addItem(label, value)
        self.units.setCurrentIndex(max(0, self.units.findData(self.config.get('units', 'cm'))))
        form.addRow('OBJ units', self.units)
        self.mode = QComboBox()
        self.mode.addItem('Still (one pose)', 'static')
        self.mode.addItem('Animation (follow the avatar)', 'animation')
        self.mode.setCurrentIndex(1 if self.config.get('output_mode') == 'animation' else 0)
        form.addRow('Mode', self.mode)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(.1, 240)
        self.fps.setDecimals(3)
        self.fps.setValue(float(self.config.get('fps', 30)))
        form.addRow('Frame rate', self.fps)
        self.take = QLineEdit(self.config.get('take') or '')
        self.take.setPlaceholderText('First take (leave blank)')
        form.addRow('FBX take', self.take)
        self.status = QLabel('Animation follows an animated FBX avatar or a folder of OBJ frames.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.load_button = QPushButton('Open scene')
        self.load_button.clicked.connect(self.load)
        layout.addWidget(self.load_button)
        layout.addWidget(footer(self))
        self.inputs = [*self.findChildren(QGroupBox), self.avatar, pick, folder, self.units, self.mode, self.fps,
                       self.take, self.load_button]

    def add(self, key):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Choose files', '', 'Meshes (*.obj *.fbx)')
        current = {self.lists[key].item(i).text() for i in range(self.lists[key].count())}
        for path in paths:
            if path not in current:
                self.lists[key].addItem(path)

    def remove(self, key):
        for item in self.lists[key].selectedItems():
            self.lists[key].takeItem(self.lists[key].row(item))

    def pick_avatar(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose the avatar', '', 'Meshes (*.obj *.fbx)')
        if path:
            self.avatar.setText(path)

    def pick_sequence(self):
        path = QFileDialog.getExistingDirectory(self, 'Folder of avatar OBJ frames')
        if path:
            self.avatar.setText(path)

    def load(self):
        if self.worker:
            return
        config = dict(self.config)
        for key, items in self.lists.items():
            config[key] = [items.item(i).text() for i in range(items.count())]
        avatar = self.avatar.text().strip()
        config.pop('avatar', None)
        config.pop('avatar_sequence', None)
        if avatar:
            config['avatar_sequence' if Path(avatar).is_dir() else 'avatar'] = avatar
        config.update(units=self.units.currentData(), output_mode=self.mode.currentData(), fps=self.fps.value(),
                      take=self.take.text().strip() or None)
        self.worker = PrepareWorker(config, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self.loaded)
        self.worker.finished.connect(self.worker.deleteLater)
        for widget in self.inputs:
            widget.setEnabled(False)
        self.status.setText('Preparing scene…')
        self.worker.start()

    def loaded(self):
        worker, self.worker = self.worker, None
        if self.close_pending:
            if worker.project:
                worker.project.close()
            self.reject()
            return
        if worker.error:
            QMessageBox.critical(self, APP_NAME, worker.error)
            for widget in self.inputs:
                widget.setEnabled(True)
            self.status.setText('Could not load the scene.')
            return
        self.project = worker.project
        self.settings.setValue('geometry', self.saveGeometry())
        self.accept()

    def reject(self):
        if self.worker:
            self.close_pending = True
            self.worker.requestInterruption()
            self.status.setText('Finishing the current import before closing…')
            return
        super().reject()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument('--session', help='DAZ capture session written by Hair Tool.dsa')
    parser.add_argument('--project', help='Standalone project JSON')
    parser.add_argument('--hair', action='append')
    parser.add_argument('--avatar')
    parser.add_argument('--avatar-sequence')
    parser.add_argument('--collider', action='append')
    parser.add_argument('--output-mode', choices=('static', 'animation'))
    parser.add_argument('--fps', type=float)
    parser.add_argument('--units', choices=('cm', 'm', 'mm'))
    parser.add_argument('--take')
    parser.add_argument('--output')
    parser.add_argument('--result')
    parser.add_argument('--after-export')
    parser.add_argument('--overwrite', action='store_true', default=None)
    parser.add_argument('--close-after', choices=('on', 'off'), help='Close after export (this run only)')
    args = parser.parse_args(argv)
    if args.session and any(value is not None for key, value in vars(args).items() if key != 'session'):
        parser.error('A DAZ session cannot be combined with standalone files or options.')
    config = read_project(args.project) if args.project else {}
    for key, target in (('hair', 'hair'), ('collider', 'colliders')):
        if getattr(args, key) is not None:
            config[target] = [str(Path(p).resolve()) for p in getattr(args, key)]
    for key in ('avatar', 'avatar_sequence', 'output', 'result', 'after_export'):
        if getattr(args, key):
            config[key] = str(Path(getattr(args, key)).resolve())
    for key in ('output_mode', 'fps', 'units', 'take', 'overwrite'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.close_after is not None:
        config['close_after'] = args.close_after == 'on'
    return args.session, config


def main(argv=None):
    from standalone_runtime import ensure_gui_streams
    ensure_gui_streams()
    app = QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)
    suite_app = str(Path(__file__).resolve().parents[2] / 'app')   # suite app/: taskbar icon
    if suite_app not in sys.path:
        sys.path.append(suite_app)
    import spin_icon
    spin_icon.set_app_icon(app, 'hair')
    live_session, project = None, None
    try:
        session, config = parse_args(argv)
        from hair_animation import load_any       # current-frame or timeline capture
        if session:
            from live_return import is_live
            if not is_live(session):
                raise ValueError("This capture predates the Spin Tools receive window. Recapture it from DAZ.")
            inputs = load_any(session)
            live_session = Path(session)
        else:
            loader = LoadDialog(config)
            if config.get('hair') and (config.get('avatar') or config.get('avatar_sequence')):
                QTimer.singleShot(0, loader.load)
            if loader.exec() != QDialog.DialogCode.Accepted:
                return 0
            project = loader.project
            inputs = load_any(project.session)
            inputs['standalone'] = project
        from hair_window import HairWindow
        window = HairWindow(inputs)
        app.setQuitOnLastWindowClosed(True)
        window.show()
        window.viewport.Initialize()
        return app.exec()
    except Exception as error:
        QMessageBox.critical(None, APP_NAME, str(error))
        return 1
    finally:
        if live_session is not None:
            try:
                from live_return import release_capture
                release_capture(live_session)
            except (OSError, ValueError):
                pass  # A cleanup failure must not invalidate a sent result.
        if project is not None:
            try:
                motion = inputs.get('motion') if 'inputs' in locals() else None
                if motion is not None:
                    motion.wait()
            finally:
                project.close()


if __name__ == "__main__":
    sys.exit(main())
