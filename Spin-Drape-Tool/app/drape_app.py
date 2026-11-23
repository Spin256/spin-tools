# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Standalone launcher, separate from drape_helper's DAZ entry point."""
import argparse
from pathlib import Path
import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QListWidget, QPushButton, QLabel, QFileDialog, QComboBox,
    QDoubleSpinBox, QLineEdit, QFormLayout, QMessageBox)

from drape_helper import DrapeWindow
import spin_ui  # suite app/ is on the path once drape_helper is imported
from standalone_project import prepare_project, read_config
from standalone_export import export_result, output_paths
from branding import APP_NAME, footer, window_settings


class PrepareWorker(QThread):
    progress = Signal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config, self.project, self.error = config, None, None

    def run(self):
        try:
            self.project = prepare_project(self.config, self.progress.emit, self.isInterruptionRequested)
        except Exception as error:
            self.error = str(error)


class ExportWorker(QThread):
    def __init__(self, project, output, vertices, animation, water, overwrite, parent):
        super().__init__(parent)
        self.args = project, output, vertices, animation, water, overwrite
        self.error, self.result = None, None

    def run(self):
        try:
            self.result = export_result(*self.args)
        except Exception as error:
            self.error = str(error)


class StandaloneWindow(DrapeWindow):
    closed = Signal()

    def __init__(self, project):
        self.project = project
        super().__init__(**project.inputs)
        self.setWindowTitle(APP_NAME + ' — Standalone')
        self.send_button.setToolTip('Export the displayed, paused result to OBJ or FBX' + spin_ui.key_hint('drape', 'send_to_daz') + '.')
        self.water_panel.export_plane.setText('Export a separate water-height OBJ plane')
        self.water_panel.export_plane.setToolTip('Include a separate OBJ reference when no Water prop was detected.')
        # Keep shared simulation controls; replace DAZ-specific explanatory text only here.
        for label in self.findChildren(QLabel):
            if label.text() == 'Return preference': label.setText('Output preference')
            if 'DAZ' in label.text():
                label.setText(label.text().replace('DAZ animation', 'imported animation').replace('DAZ frame', 'source frame')
                              .replace('DAZ water prop', 'source water prop').replace('to DAZ', 'with the export'))
        # The combined Simulation section keeps its name in the standalone window too.
        self.bridge = bool(project.config.get('after_export') or project.config.get('callback'))
        if self.bridge:
            self.configure_close_after('bridge', 'Close after send', True,
                                       'On: exit after the export and its bridge callback start. '
                                       'Off: minimize so you can send again; each export runs the callback again.',
                                       project.config.get('close_after'))
        else:
            self.configure_close_after('export', 'Close after export', False,
                                       'On: close the simulator once the files are written.',
                                       project.config.get('close_after'))
        self.update_output_note()

    def update_output_note(self):
        animated = self.output_mode.currentData() == 'animation'
        self.output_note.setText(
            'Export the settled garment at each source frame as FBX animation.' if animated and self.motion else
            'Load an animated FBX or an OBJ sequence to export animation.' if animated else
            'Pause the simulation to export the displayed result as OBJ or FBX.')
        if hasattr(self, 'send_button'): self.update_send_button()
        if hasattr(self, 'pause_simulation'): self.update_live_controls()

    def update_send_button(self):
        animated = self.output_mode.currentData() == 'animation'
        self.send_button.setText('Export animation…' if animated else 'Export static result…')
        self.send_button.setEnabled(self.result_ready and self.return_worker is None and not self.close_pending
            and (self.worker is None or (self.live is not None and self.live.paused_ack))
            and (not animated or self.animation is not None))

    def live_paused(self):
        super().live_paused()
        self.progress.setText(self.progress.activity.text().replace('send to DAZ', 'export the result'))

    def send_static_result(self):
        if not self.send_button.isEnabled(): return
        animation = self.animation if self.output_mode.currentData() == 'animation' else None
        output = self.project.config.get('output')
        if not output:
            filters = 'FBX animation (*.fbx)' if animation is not None else 'Wavefront OBJ (*.obj);;FBX mesh (*.fbx)'
            name = str(Path(self.project.entries[0]['source']).with_name('draped.fbx' if animation else 'draped.obj'))
            output, selected = QFileDialog.getSaveFileName(self, 'Export drape', name, filters,
                options=QFileDialog.Option.DontConfirmOverwrite)
            if not output: return
            if not Path(output).suffix: output += '.fbx' if 'FBX' in selected else '.obj'
        water = None
        if (self.water_panel.enabled_box.isChecked() and self.water_panel.export_plane.isChecked()
                and self.water_panel.water_source is None):
            water = dict(height=self.water_panel.fields['height'].value(), x=float(self.water_preview.center[0]),
                         z=float(self.water_preview.center[2]), extent=self.water_preview.extent)
        try:
            paths, result, plane = output_paths(self.project, output, water)
            existing = [p for p in paths + [result] + ([plane] if plane else []) if p.exists()]
            overwrite = bool(self.project.config.get('overwrite'))
            if existing and not overwrite:
                answer = QMessageBox.question(self, 'Replace exported files?',
                    'Replace these files?\n\n'+'\n'.join(str(p) for p in existing),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes: return
                overwrite = True
            self.return_worker = ExportWorker(self.project, output, self.vertices.copy(), animation, water, overwrite, self)
        except Exception as error:
            QMessageBox.critical(self, 'Export drape', str(error))
            return
        self.update_live_controls()
        for widget in self.edit_controls + [self.run_button, self.reset_button, self.output_mode, self.send_button]:
            widget.setEnabled(False)
        self.status.set_state('exporting')
        self.progress.setText('Exporting drape…')
        self.return_worker.finished.connect(self.finish_return)
        self.return_worker.finished.connect(self.return_worker.deleteLater)
        self.return_worker.start()

    def finish_return(self):
        worker = self.return_worker
        error, result = worker.error, worker.result
        self.return_worker = None
        if error:
            self.close_pending = False
            QMessageBox.critical(self, 'Export drape', error)
            self.status.set_state('failed')
            self.progress.setText('Export failed. The displayed result is available to retry.')
        else:
            self.status.set_state('exported')
            self.progress.setText('Exported: ' + str(result[1]))
            if self.close_pending or self.close_after.isChecked():
                self.close()
                return
            if self.bridge:
                # Control passes to the receiving application; restore to send again.
                self.showMinimized()
        for widget in self.edit_controls + [self.output_mode]: widget.setEnabled(self.worker is None)
        self.run_button.setEnabled(True)
        self.update_live_controls()
        self.update_send_button()

    def closeEvent(self, event):
        super().closeEvent(event)
        if event.isAccepted(): self.closed.emit()


class Launcher(QWidget):
    closed = Signal()
    def __init__(self, config=None, direct=False):
        super().__init__()
        self.worker, self.viewer, self.project = None, None, None
        self.close_pending, self.direct = False, direct
        self.handing_off = False
        self.base_config = dict(config or {})
        self.settings = window_settings('Standalone')
        self.setWindowTitle('Open ' + APP_NAME + ' — Standalone')
        self.resize(900, 680)
        if self.settings.value('geometry'): self.restoreGeometry(self.settings.value('geometry'))
        layout = QVBoxLayout(self)
        note = QLabel('Load meshes in their starting positions. OBJ uses Y-up world coordinates; FBX units and axes are converted automatically. '
                      'An avatar or prop can supply movement. Garments start at the first frame.')
        note.setWordWrap(True)
        layout.addWidget(note)
        grid = QGridLayout()
        self.lists = {}
        for n, (key, title) in enumerate((('props', 'Props / static collision objects'),
                ('avatars', 'Avatar / moving collision objects'), ('garments', 'Garments to drape'),
                ('sequences', 'Moving objects — OBJ sequence folders'))):
            box = QGroupBox(title)
            group = QVBoxLayout(box)
            items = QListWidget()
            items.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
            for path in self.base_config.get(key, []): items.addItem(path)
            self.lists[key] = items
            group.addWidget(items)
            row = QHBoxLayout()
            add = QPushButton('Add folder…' if key == 'sequences' else 'Add files…')
            remove = QPushButton('Remove selected')
            add.clicked.connect(lambda checked=False, k=key: self.add_files(k))
            remove.clicked.connect(lambda checked=False, k=key: self.remove_files(k))
            row.addWidget(add)
            row.addWidget(remove)
            group.addLayout(row)
            grid.addWidget(box, n//2, n%2)
        layout.addLayout(grid)
        form = QFormLayout()
        self.units = QComboBox()
        for label, key in (('Centimeters', 'cm'), ('Meters', 'm'), ('Millimeters', 'mm')): self.units.addItem(label, key)
        self.units.setCurrentIndex(max(0, self.units.findData(self.base_config.get('units', 'cm'))))
        self.fps = QDoubleSpinBox()
        self.fps.setRange(.1, 240)
        self.fps.setDecimals(3)
        self.fps.setValue(float(self.base_config.get('fps', 30)))
        self.take = QLineEdit(self.base_config.get('take') or '')
        self.take.setPlaceholderText('First animation take (leave blank)')
        self.mode = QComboBox()
        self.mode.addItem('Final static mesh', 'static')
        self.mode.addItem('Animation sequence (FBX)', 'animation')
        self.mode.setCurrentIndex(1 if self.base_config.get('output_mode') == 'animation' else 0)
        form.addRow('OBJ units', self.units)
        form.addRow('Sample frame rate', self.fps)
        form.addRow('FBX take name (optional)', self.take)
        form.addRow('Output preference', self.mode)
        layout.addLayout(form)
        self.status = QLabel('OBJ sequences: one object per folder, numbered filenames, matching topology.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.open_button = QPushButton('Open scene in ' + APP_NAME)
        self.open_button.clicked.connect(self.load)
        layout.addWidget(self.open_button)
        self.copyright_footer = footer(self)
        layout.addWidget(self.copyright_footer)

    def add_files(self, key):
        if key == 'sequences':
            folder = QFileDialog.getExistingDirectory(self, 'Choose OBJ sequence folder')
            paths = [folder] if folder else []
        else:
            paths, _ = QFileDialog.getOpenFileNames(self, 'Choose '+key, '', 'Meshes (*.obj *.fbx)')
        current = {self.lists[key].item(i).text() for i in range(self.lists[key].count())}
        for path in paths:
            if path not in current: self.lists[key].addItem(path)

    def remove_files(self, key):
        for item in self.lists[key].selectedItems(): self.lists[key].takeItem(self.lists[key].row(item))

    def load(self):
        if self.worker is not None: return
        config = dict(self.base_config)
        for key, items in self.lists.items(): config[key] = [items.item(i).text() for i in range(items.count())]
        config.update(units=self.units.currentData(), fps=self.fps.value(), take=self.take.text().strip() or None,
                      output_mode=self.mode.currentData())
        self.worker = PrepareWorker(config, self)
        for child in self.findChildren(QGroupBox): child.setEnabled(False)
        for child in (self.open_button, self.units, self.fps, self.take, self.mode): child.setEnabled(False)
        self.status.setText('Preparing scene…')
        self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self.loaded)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def loaded(self):
        worker, self.worker = self.worker, None
        if self.close_pending:
            if worker.project: worker.project.close()
            self.close()
            return
        for child in self.findChildren(QGroupBox): child.setEnabled(True)
        for child in (self.open_button, self.units, self.fps, self.take, self.mode): child.setEnabled(True)
        if worker.error:
            self.status.setText('Could not load scene.')
            QMessageBox.critical(self, APP_NAME, worker.error)
            return
        self.project = worker.project
        try:
            self.viewer = StandaloneWindow(self.project)
            self.viewer.closed.connect(self.viewer_closed)
            self.viewer.show()
            # Close the native selection window before initializing VTK, which
            # may process window events. This close transfers control rather
            # than requesting application exit; the simulator owns the scene.
            self.handing_off = True
            try:
                self.close()
            finally:
                self.handing_off = False
            self.viewer.viewport.Initialize()
        except Exception as error:
            if self.viewer is not None:
                self.viewer.closed.disconnect(self.viewer_closed)
                self.viewer.close()
                self.viewer.deleteLater()
                self.viewer = None
            if self.project is not None: self.project.close()
            self.project = None
            self.show()
            QMessageBox.critical(self, APP_NAME, str(error))
            return

    def viewer_closed(self):
        bridge = self.project.config.get('callback') or self.project.config.get('after_export')
        self.project.close()
        self.project = None
        self.viewer.deleteLater()
        self.viewer = None
        if self.direct or bridge:
            QApplication.instance().quit()
        else:
            self.status.setText('Choose files to open another scene.')
            self.show()

    def closeEvent(self, event):
        if self.worker is not None:
            self.close_pending = True
            self.worker.requestInterruption()
            self.status.setText('Finishing the current import before closing…')
            event.ignore()
            return
        self.settings.setValue('geometry', self.saveGeometry())
        super().closeEvent(event)
        if event.isAccepted() and not self.handing_off: self.closed.emit()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Standalone Drape Tool (does not use the DAZ bridge).')
    parser.add_argument('--project', help='Standalone/custom bridge JSON configuration')
    for flag in ('prop', 'avatar', 'garment', 'avatar-sequence'):
        parser.add_argument('--'+flag, action='append', help='Input path; repeat for multiple objects')
    parser.add_argument('--fps', type=float)
    parser.add_argument('--units', choices=('cm', 'm', 'mm'))
    parser.add_argument('--take', help='FBX animation take name; default first take')
    parser.add_argument('--output-mode', choices=('static', 'animation'))
    parser.add_argument('--output', help='Default export .obj or .fbx path')
    parser.add_argument('--result', help='Result JSON path in the export folder')
    parser.add_argument('--after-export', help='Python script to launch with the completed result JSON path')
    parser.add_argument('--overwrite', action='store_true', default=None, help='Allow replacing existing exports')
    parser.add_argument('--close-after', choices=('on', 'off'),
                        help='Close after export/send (overrides the remembered checkbox for this run)')
    args = parser.parse_args(argv)
    config = read_config(args.project) if args.project else {}
    values = vars(args)
    for key, role in (('prop', 'props'), ('avatar', 'avatars'), ('garment', 'garments'), ('avatar_sequence', 'sequences')):
        if values[key] is not None: config[role] = [str(Path(p).resolve()) for p in values[key]]
    for key in ('fps', 'units', 'take', 'output_mode', 'overwrite'):
        if values[key] is not None: config[key] = values[key]
    for key in ('output', 'result', 'after_export'):
        if values[key] is not None: config[key] = str(Path(values[key]).resolve())
    if values['close_after'] is not None: config['close_after'] = values['close_after'] == 'on'
    if 'close_after' in config and not isinstance(config['close_after'], bool):
        raise ValueError('close_after must be true or false.')
    return config, bool(config.get('garments'))


def main(argv=None):
    from standalone_runtime import ensure_gui_streams
    ensure_gui_streams()
    app = QApplication([])
    suite_app = str(Path(__file__).resolve().parents[2] / 'app')   # suite app/: taskbar icon
    if suite_app not in sys.path:
        sys.path.append(suite_app)
    import spin_icon
    spin_icon.set_app_icon(app, 'drape')
    app.setQuitOnLastWindowClosed(False)
    try:
        config, direct = parse_args(argv)
        launcher = Launcher(config, direct)
    except Exception as error:
        QMessageBox.critical(None, APP_NAME, str(error))
        return 1
    # A hidden import window must not keep a closed simulator process alive.
    launcher.closed.connect(app.quit)
    from PySide6.QtCore import QTimer
    launcher.show()
    if direct: QTimer.singleShot(0, launcher.load)
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
