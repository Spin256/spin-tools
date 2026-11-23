# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Shared interface pieces for the Spin Tools helper windows (Spin256 · https://www.spin256.com/tools).

Interface only: nothing here reads or writes DAZ transfer files. Each tool keeps its own
simulation, workers and send/return code and builds its window from these pieces:

- ``remember_window`` / ``save_window``: per-tool geometry and splitter; a tool opened for the
  first time uses the size/position of the most recently closed Spin tool (shared suite key).
- ``Sections``: collapsible left-panel sections whose open/closed state is remembered per tool.
- ``ActionRow``: Run/Pause, Reset, Send to DAZ and Close after sending, same order and wording.
- ``StatusBar``: below the viewport. Left: simulation state. Middle: frame x of y (timeline
  work). Right: what the tool is doing now, with a progress bar when the length is known.
- ``ViewControls``: show/hide and opacity for the simulated object and the avatar/colliders.
- ``ViewportCamera``: keyboard orbit/pan/zoom/reset shared by every tool.
- ``Shortcuts``: keys from keybindings.json (common + the tool's section), plus F1 / ? help.
- ``footer``, ``style_renderer``, ``report_error``: Spin256 footer, viewport look, error style.
"""
import math
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))      # appended: a tool's own app/ modules win

import keybindings  # noqa: E402  (suite app/ folder)

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QKeySequence, QShortcut, QDesktopServices  # noqa: E402
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog,  # noqa: E402
                               QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QTextBrowser, QToolButton, QVBoxLayout, QWidget)

ORGANISATION = "Spin256"
SUITE_SETTINGS = "Spin Tools"
SITE = "https://www.spin256.com/tools"
DONATE = "https://www.spin256.com/tools/donate"
FOOTER = (f'© Spin256 · <a href="{SITE}">spin256.com/tools</a> · <a href="{DONATE}">Donate</a>')
BACKGROUND = (.12, .14, .18)
AVATAR_COLOR = (.67, .67, .67)
MOVING_COLOR = (.71, .43, .90)
OBJECT_COLOR = (.22, .61, .96)

# Simulation states shown on the left of the status bar (one wording for every tool).
STATES = {
    "idle": "Simulation not started",
    "preparing": "Preparing simulation",
    "running": "Simulation running",
    "paused": "Simulation paused",
    "finished": "Simulation finished",
    "sending": "Sending to DAZ",
    "sent": "Sent to DAZ",
    "exporting": "Exporting",
    "exported": "Exported",
    "failed": "Failed",
    "ready": "Ready",
}

# Readable names for the F1 list.
ACTION_NAMES = {
    "pause_resume": "Run / pause / resume", "run_simulation": "Run simulation",
    "reset_simulation": "Reset simulation", "send_to_daz": "Send to DAZ",
    "orbit_up": "Orbit up", "orbit_down": "Orbit down", "orbit_left": "Orbit left", "orbit_right": "Orbit right",
    "pan_up": "Pan up", "pan_down": "Pan down", "pan_left": "Pan left", "pan_right": "Pan right",
    "zoom_in": "Zoom in", "zoom_out": "Zoom out", "reset_view": "Reset view",
    "cancel_interaction": "Cancel the current tool / mode",
    "grow_selection": "Grow selection", "shrink_selection": "Shrink selection",
    "freeze_selected": "Freeze selected", "freeze_unselected": "Freeze unselected",
    "unfreeze_selected": "Unfreeze selected", "unfreeze_all": "Unfreeze all",
    "clear_selection": "Clear selection", "show_avatar": "Show / hide avatar",
    "show_props": "Show / hide props", "hold_frame": "Hold frame", "grab_fabric": "Grab fabric",
    "paint_selection": "Paint selection", "paint_add": "Paint: add", "paint_remove": "Paint: remove",
    "edit_pins": "Edit pins", "edit_fasteners": "Edit fastenings", "pin_top": "Pin top",
    "clear_pins": "Clear pins", "clear_fasteners": "Clear fastenings",
    "pull_hair": "Pull hair", "pick_strands": "Pick strands", "ring_select": "Ring select",
    "toggle_xray": "X-ray (strands only)", "select_connected": "Select connected",
    "split_off": "Split off selection", "frame_mesh": "Frame mesh", "paint_supports": "Paint supports",
    "show_help": "Show this list",
}

CAMERA_ACTIONS = {
    "orbit_up": ("orbit", 0, 20), "orbit_down": ("orbit", 0, -20),
    "orbit_left": ("orbit", 20, 0), "orbit_right": ("orbit", -20, 0),
    "pan_up": ("pan", 0, 1), "pan_down": ("pan", 0, -1),
    "pan_left": ("pan", -1, 0), "pan_right": ("pan", 1, 0),
    "zoom_in": ("zoom", 1.2, 0), "zoom_out": ("zoom", 1 / 1.2, 0)}


def key_for(tool, action):
    """The first key bound to ``action`` for ``tool`` in keybindings.json ("" when none), so text
    that mentions a key follows the user's own bindings."""
    bindings, _ = keybindings.load(tool or "common", keybindings.PATH)
    keys = bindings.get(action) or []
    return keys[0] if keys else ""


def key_hint(tool, action, before=" (", after=")"):
    """" (Space)" for messages and tooltips, or "" when the action has no key."""
    key = key_for(tool, action)
    return f"{before}{key}{after}" if key else ""


def suite_settings():
    return QSettings(ORGANISATION, SUITE_SETTINGS)


# ----- window geometry ---------------------------------------------------------------
def remember_window(window, settings, splitter=None):
    """Restore the tool's own geometry; a first opening borrows the last closed Spin tool's."""
    geometry = settings.value("geometry")
    if geometry is None:
        geometry = suite_settings().value("last_geometry")
    if geometry is not None:
        window.restoreGeometry(geometry)
    if splitter is not None and settings.value("splitter") is not None:
        splitter.restoreState(settings.value("splitter"))


def save_window(window, settings, splitter=None):
    geometry = window.saveGeometry()
    settings.setValue("geometry", geometry)
    suite_settings().setValue("last_geometry", geometry)
    if splitter is not None:
        settings.setValue("splitter", splitter.saveState())


# ----- footer, viewport look, errors ---------------------------------------------------
def footer(parent=None, version=None, tool_folder=None):
    """``version``: the tool's own version (its branding.VERSION), shown left of the copyright."""
    short_version = ".".join(str(version).split(".")[:2]) if version else "About"
    about_link = f'<a href="about:spin">v{short_version}</a>' if version else '<a href="about:spin">About</a>'
    label = QLabel(about_link + " · " + FOOTER, parent)
    label.setToolTip(SITE + "  ·  Donate: " + DONATE)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setOpenExternalLinks(False)
    def activate(target):
        if target == 'about:spin':
            from spin_about import AboutDialog
            try:
                dialog = AboutDialog(label.window(), tool_folder, version)
            except (OSError, ValueError, KeyError, TypeError) as error:
                QMessageBox.warning(label.window(), 'About', f'Could not read About information.\n{error}')
                return
            dialog.exec()
        else:
            QDesktopServices.openUrl(QUrl(target))
    label.linkActivated.connect(activate)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setContentsMargins(4, 5, 4, 3)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    return label


LIGHTS = {}          # renderer address -> (key light, headlight); see style_renderer / Lighting


def light_direction(heading, elevation):
    """Unit vector from the scene center toward the light. Heading 0° = in front (+Z, DAZ's
    front), 90° = the figure's left (+X); elevation +90° = straight above (Y-up)."""
    h, e = math.radians(heading), math.radians(elevation)
    return (math.sin(h) * math.cos(e), math.sin(e), math.cos(h) * math.cos(e))


def style_renderer(renderer):
    """Same background and lighting in every tool: a key light that always aims at the scene
    center (direction and brightness from the View section's Lighting controls) plus a soft
    headlight so faces turned away from the key light never go black. Adding lights switches
    off VTK's automatic headlight, so both are created here."""
    import vtk
    renderer.SetBackground(*BACKGROUND)
    renderer.SetBackground2(.20, .23, .29)
    renderer.GradientBackgroundOn()
    key_name = renderer.GetAddressAsString("vtkRenderer")
    # Keyed by address, so a new renderer at a freed renderer's address must not reuse its lights.
    if key_name not in LIGHTS or not renderer.GetLights().IsItemPresent(LIGHTS[key_name][0]):
        key = vtk.vtkLight()
        key.SetLightTypeToSceneLight()
        key.PositionalOff()
        head = vtk.vtkLight()
        head.SetLightTypeToHeadlight()
        renderer.AddLight(key)
        renderer.AddLight(head)
        LIGHTS[key_name] = (key, head)
        aim_light(renderer, DEFAULT_LIGHT["heading"], DEFAULT_LIGHT["elevation"], DEFAULT_LIGHT["brightness"])
    return renderer


DEFAULT_LIGHT = dict(heading=30., elevation=35., brightness=100)


def aim_light(renderer, heading, elevation, brightness):
    """Point the key light at the center of the visible scene from (heading, elevation)."""
    lights = LIGHTS.get(renderer.GetAddressAsString("vtkRenderer"))
    if lights is None or not renderer.GetLights().IsItemPresent(lights[0]):
        return
    key, head = lights
    bounds = renderer.ComputeVisiblePropBounds()
    if bounds[0] > bounds[1]:                     # nothing visible yet
        bounds = (-1, 1, -1, 1, -1, 1)
    centre = [(bounds[i] + bounds[i + 1]) / 2 for i in (0, 2, 4)]
    size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1e-6)
    d = light_direction(heading, elevation)
    key.SetFocalPoint(*centre)
    key.SetPosition(*(c + v * size * 2 for c, v in zip(centre, d)))
    level = brightness / 100
    key.SetIntensity(.85 * level)
    head.SetIntensity(.45 * level)


class LightSphere(QWidget):
    """Drag to choose where the light comes from (as the wind's direction sphere). The arrow
    always points at the center: the light is aimed at the middle of the scene."""
    from PySide6.QtCore import Signal as _Signal
    aimed = _Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.heading, self.elevation, self.drag = DEFAULT_LIGHT["heading"], DEFAULT_LIGHT["elevation"], None
        self.setMinimumSize(150, 150)
        self.setMaximumHeight(170)
        self.setToolTip("Drag left/right to move the light around the scene, up/down to raise or lower it. "
                        "It always shines toward the center of the scene.")

    def set_angles(self, heading, elevation):
        self.heading, self.elevation = heading, elevation
        self.update()

    def paintEvent(self, event):
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QPainter, QPen, QRadialGradient
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = min(self.width() * .36, (self.height() - 30) * .43)
        centre = QPointF(self.width() / 2, (self.height() - 22) / 2)
        d = light_direction(self.heading, self.elevation)
        source = centre + QPointF(float(d[0] * .85 + d[2] * .4) * radius, float(-d[1] * .9 + d[2] * .3) * radius)
        gradient = QRadialGradient(source, radius * 1.6)
        gradient.setColorAt(0, QColor("#f4ead0"))
        gradient.setColorAt(1, QColor("#3a4654"))
        p.setBrush(gradient)
        p.setPen(QPen(QColor("#7e98ad"), 1))
        p.drawEllipse(centre, radius, radius)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#68859a"), 1))
        p.drawEllipse(centre, radius * .42, radius)
        p.drawEllipse(centre, radius, radius * .35)
        p.setPen(QPen(QColor("#ffd966"), 3))
        p.drawLine(source, centre)                 # from the light to the scene centre
        p.setBrush(QColor("#ffd966"))
        p.drawEllipse(source, 7, 7)
        p.drawEllipse(centre, 3, 3)
        p.setPen(self.palette().text().color())
        p.drawText(0, self.height() - 22, self.width(), 20, Qt.AlignmentFlag.AlignCenter,
                   f"{self.heading:.0f}° around · {self.elevation:+.0f}° up")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag = (event.position(), self.heading, self.elevation)

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            start, heading, elevation = self.drag
            delta = event.position() - start
            self.aimed.emit((heading + delta.x() * .8) % 360, max(-89, min(89, elevation - delta.y() * .8)))

    def mouseReleaseEvent(self, event):
        self.drag = None


class Lighting:
    """Brightness and light direction controls (remembered per tool). Call ``attach`` once the
    renderer exists; the key light then always aims at the center of the visible scene."""

    def __init__(self, layout, settings):
        self.settings, self.renderer, self.render = settings, None, None
        self.brightness = QSlider(Qt.Orientation.Horizontal)
        self.brightness.setRange(20, 250)
        self.brightness.setValue(settings.value("view/brightness", DEFAULT_LIGHT["brightness"], type=int))
        self.brightness.setToolTip("Viewport lighting only; the simulation and result are unchanged.")
        self.sphere = LightSphere()
        self.sphere.set_angles(settings.value("view/light_heading", DEFAULT_LIGHT["heading"], type=float),
                               settings.value("view/light_elevation", DEFAULT_LIGHT["elevation"], type=float))
        form = QFormLayout()
        form.addRow("Brightness", self.brightness)
        layout.addLayout(form)
        layout.addWidget(QLabel("Light direction (aims at the scene center)"))
        layout.addWidget(self.sphere)
        reset = QPushButton("Default lighting")
        reset.clicked.connect(self.reset)
        layout.addWidget(reset)
        self.brightness.valueChanged.connect(lambda *_: self.apply())
        self.sphere.aimed.connect(self.aim)

    def aim(self, heading, elevation):
        self.sphere.set_angles(heading, elevation)
        self.apply()

    def reset(self):
        self.brightness.setValue(DEFAULT_LIGHT["brightness"])
        self.aim(DEFAULT_LIGHT["heading"], DEFAULT_LIGHT["elevation"])

    def attach(self, renderer, render):
        self.renderer, self.render = renderer, render
        self.apply(draw=False)

    def apply(self, draw=True):
        s = self.settings
        s.setValue("view/brightness", self.brightness.value())
        s.setValue("view/light_heading", self.sphere.heading)
        s.setValue("view/light_elevation", self.sphere.elevation)
        if self.renderer is not None:
            aim_light(self.renderer, self.sphere.heading, self.sphere.elevation, self.brightness.value())
            if draw and self.render is not None:
                self.render()


def report_error(window, status, title, message, dialog=True):
    """Status message always; a dialog only for real failures (``dialog``)."""
    if status is not None:
        status.set_state("failed")
        status.setText(f"{title}: {message}" if message else title)
    if dialog:
        QMessageBox.critical(window, title, message or title)


# ----- left panel sections ---------------------------------------------------------------
class Sections:
    """Collapsible titled sections; open/closed state remembered per tool in ``settings``."""

    def __init__(self, layout, settings):
        self.layout, self.settings, self.items = layout, settings, {}

    def add(self, title, expanded=False):
        key = "sections/" + title.lower().replace(" ", "_")
        opened = self.settings.value(key, expanded, type=bool)
        button = QToolButton()
        button.setText(title)
        button.setCheckable(True)
        button.setChecked(opened)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setStyleSheet("QToolButton { font-weight: bold; text-align: left; }")
        body = QWidget()
        content = QVBoxLayout(body)
        content.setContentsMargins(4, 2, 4, 8)

        def toggle(on):
            body.setVisible(on)
            button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
            self.settings.setValue(key, on)
        button.toggled.connect(toggle)
        self.layout.addWidget(button)
        self.layout.addWidget(body)
        toggle(opened)
        self.items[title] = (button, body)
        return content

    def set_visible(self, title, visible):
        """Show or hide a whole section (heading and body), e.g. timeline-only sections."""
        button, body = self.items[title]
        button.setVisible(visible)
        body.setVisible(visible and button.isChecked())

    def set_busy(self, busy, keep=("View",)):
        """While the tool is busy (analyzing, building, sending) every section's controls are disabled except those
        in ``keep``. Headings still open and close; each control's own state is kept."""
        for title, (button, body) in self.items.items():
            body.setEnabled(not busy or title in keep)

    def form(self, title, expanded=False):
        """A section holding a QFormLayout (returned)."""
        content = self.add(title, expanded)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        content.addLayout(form)
        return form


def note(text, parent_layout=None):
    label = QLabel(text)
    label.setWordWrap(True)
    # Normal text color, set apart by italics.
    label.setStyleSheet("font-style: italic;")
    if parent_layout is not None:
        parent_layout.addWidget(label)
    return label


# ----- bottom button row -----------------------------------------------------------------------
class ActionRow(QWidget):
    """Run/Pause, Reset and Send to DAZ stacked full width, with Close after sending beneath."""
    RUN, PAUSE, RESUME = "Run simulation", "Pause simulation", "Resume simulation"

    def __init__(self, settings, run_text=None, reset_text="Reset", send_text="Send to DAZ",
                 close_context="daz", close_text="Close after sending", close_default=False,
                 close_tip="Off: minimize after each send so you can inspect the result in DAZ and send again. "
                           "On: close once the result is written.", close_override=None, parent=None, tool=None):
        super().__init__(parent)
        self.settings = settings
        self.run_text = run_text or self.RUN
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        row = QVBoxLayout()          # stacked: a row of three is too wide for the panel
        row.setSpacing(4)
        self.run_button = QPushButton(self.run_text)
        self.reset_button = QPushButton(reset_text)
        self.send_button = QPushButton(send_text)
        self.run_button.setToolTip("Run, pause or resume the simulation" + key_hint(tool, "pause_resume") + ".")
        self.reset_button.setToolTip("Return to the captured shape. Settings, selections and freezes are kept.")
        self.send_button.setToolTip("Send the displayed result" + key_hint(tool, "send_to_daz") + ".")
        self.send_button.setDefault(True)
        for button in (self.run_button, self.reset_button, self.send_button):
            button.setMinimumHeight(30)
            row.addWidget(button)
        layout.addLayout(row)
        self.close_after = QCheckBox(close_text)
        self.close_after_key = None
        self.remember = None
        self.close_after.toggled.connect(self._remember)
        layout.addWidget(self.close_after)
        self.configure_close_after(close_context, close_text, close_default, close_tip, close_override)

    def configure_close_after(self, context, text, default, tip, override=None):
        """Remembered per context; an explicit override (command line/project) is not saved."""
        self.close_after_key = None
        self.close_after.setText(text)
        self.close_after.setToolTip(tip)
        self.close_after.setChecked(override if isinstance(override, bool) else
                                    self.settings.value("close_after/" + context, default, type=bool))
        self.close_after_key = None if isinstance(override, bool) else "close_after/" + context

    def _remember(self, checked):
        if self.remember is not None:          # the tool keeps its own key (tests may clear it)
            self.remember(checked)
        elif self.close_after_key:
            self.settings.setValue(self.close_after_key, checked)

    def show_run_state(self, worker_exists, paused):
        self.run_button.setText(self.RESUME if (worker_exists and paused) else
                                self.PAUSE if worker_exists else self.run_text)


# ----- status bar ----------------------------------------------------------------------------
class ElidedLabel(QLabel):
    """One line that ends with "…" when it does not fit; the full text is in the tooltip and text()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.full = ""

    def setText(self, text):
        self.full = text or ""
        self.setToolTip(self.full)
        self._fit()

    def text(self):
        return self.full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        shown = self.fontMetrics().elidedText(self.full, Qt.TextElideMode.ElideRight, max(10, self.width()))
        QLabel.setText(self, shown)

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(40)
        return hint


class StatusBar(QWidget):
    """Left: simulation state · middle: frame x of y · right: current activity (+ progress)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        self.state_label, self.frame_label, self.activity = QLabel(), QLabel(), ElidedLabel()
        self.state_label.setMinimumWidth(170)
        self.state_label.setStyleSheet("font-weight: bold;")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.activity.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.activity.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(160)
        self.progress.setMaximumHeight(14)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.state_label)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.activity, 1)             # the message gets all the room that is free
        self.frame_label.hide()
        layout.addWidget(self.progress)
        self.state = None
        self.set_state("idle")

    def set_state(self, state):
        self.state = state
        self.state_label.setText(STATES.get(state, state))
        colour = {"preparing": "#9fb7d9", "running": "#7fd88f", "paused": "#f2c94c", "failed": "#ff6b6b", "sent": "#6fb6ff",
                  "finished": "#6fb6ff", "exported": "#6fb6ff"}.get(state, "")
        self.state_label.setStyleSheet("font-weight: bold;" + (f" color: {colour};" if colour else ""))

    def set_frame(self, current=None, total=None):
        self.frame_label.setText("" if current is None or total is None else f"Frame {current:.0f} of {total:.0f}")
        self.frame_label.setVisible(bool(self.frame_label.text()))   # hidden when empty: more room for the message

    def setText(self, text, progress=None):
        """What the tool is doing now; ``progress`` 0–1 shows the bar, None hides it."""
        # GPU/contact preparation before the first frame shows as its own state.
        preparing = (text or "").startswith("Preparing")
        if self.state == "running" and preparing:
            self.set_state("preparing")
        elif self.state == "preparing" and not preparing:
            self.set_state("running")
        self.activity.setText(text or "")              # elided to fit; full text in the tooltip
        self.set_progress(progress)

    def set_progress(self, fraction=None):
        if fraction is None:
            self.progress.hide()
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(max(0., min(1., fraction)) * 1000))
            self.progress.show()

    def text(self):
        """Everything shown, for tests and logs."""
        return " · ".join(part for part in (self.state_label.text(), self.frame_label.text(),
                                            self.activity.text()) if part)


class WheelGuard(QObject):
    """Scrolling the panel never changes a setting: a number box, dropdown or slider in the panel only takes the
    wheel once it has been clicked (has focus); otherwise the wheel scrolls the panel, so scrolling past a box
    cannot change its value."""

    TYPES = (QAbstractSpinBox, QComboBox, QSlider)

    def __init__(self, scroll):
        super().__init__(scroll)
        self.scroll = scroll
        scroll.setProperty("spin_wheel_guard", True)
        for widget in scroll.widget().findChildren(QWidget) if scroll.widget() else ():
            if isinstance(widget, self.TYPES) and widget.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        QApplication.instance().installEventFilter(self)
        scroll.destroyed.connect(lambda *_: QApplication.instance() and
                                 QApplication.instance().removeEventFilter(self))

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel or not isinstance(obj, QWidget):
            return False
        target = obj if isinstance(obj, self.TYPES) else obj.parentWidget()
        if not isinstance(target, self.TYPES) or not self.scroll.isAncestorOf(target):
            return False
        if target.focusPolicy() == Qt.FocusPolicy.WheelFocus:
            target.setFocusPolicy(Qt.FocusPolicy.StrongFocus)          # the wheel itself must not give it focus
        if target.hasFocus():
            return False
        bar = self.scroll.verticalScrollBar()
        steps = event.angleDelta().y() / 120.
        bar.setValue(bar.value() - int(round(steps * bar.singleStep() * 3)))
        return True


def control_column(panel, actions, min_width=380, version=None, tool_folder=None):
    """Left column: the scrolling ``panel`` of sections, then the ActionRow and the Spin256
    footer, always visible. Returns (column, footer_label)."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(panel)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.wheel_guard = WheelGuard(scroll)
    column = QWidget()
    column.setMinimumWidth(min_width)
    layout = QVBoxLayout(column)
    layout.setContentsMargins(0, 0, 4, 0)
    layout.addWidget(scroll, 1)
    layout.addWidget(actions)
    credit = footer(column, version, tool_folder)
    layout.addWidget(credit)
    return column, credit


def viewport_with_status(viewport):
    """A widget stacking the viewport above a StatusBar; returns (widget, status)."""
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(viewport, 1)
    status = StatusBar(holder)
    layout.addWidget(status)
    return holder, status


# ----- View section -------------------------------------------------------------------------
class ViewControls:
    """Show/hide + opacity for the simulated object and the avatar/colliders, remembered per tool."""

    def __init__(self, layout, settings, object_name, on_change, avatar_name="avatar / colliders",
                 object_opacity=100, avatar_opacity=55, tool=None):
        self.settings = settings
        self.show_object = QCheckBox("Show " + object_name)
        self.show_avatar = QCheckBox("Show " + avatar_name)
        for box in (self.show_object, self.show_avatar):
            box.setChecked(True)
            box.setToolTip("Preview only. Hidden objects still take part in the simulation.")
            layout.addWidget(box)
        form = QFormLayout()
        self.object_opacity = self._slider(settings.value("view/object_opacity", object_opacity, type=int))
        self.avatar_opacity = self._slider(settings.value("avatar_opacity", avatar_opacity, type=int))
        form.addRow(object_name[:1].upper() + object_name[1:] + " opacity", self.object_opacity)
        form.addRow(avatar_name[:1].upper() + avatar_name[1:] + " opacity", self.avatar_opacity)
        layout.addLayout(form)
        self.reset_view_button = QPushButton("Reset view")
        self.reset_view_button.setToolTip("Fit the view again" + key_hint(tool, "reset_view") + ". The simulation is unchanged.")
        layout.addWidget(self.reset_view_button)
        self.lighting = Lighting(layout, settings)
        for widget in (self.show_object, self.show_avatar):
            widget.toggled.connect(lambda *_: on_change())
        for widget in (self.object_opacity, self.avatar_opacity):
            widget.valueChanged.connect(lambda *_: on_change())
        self.object_opacity.valueChanged.connect(lambda v: settings.setValue("view/object_opacity", v))
        self.avatar_opacity.valueChanged.connect(lambda v: settings.setValue("avatar_opacity", v))

    @staticmethod
    def _slider(value):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(5, 100)
        slider.setValue(value)
        return slider

    def apply(self, object_actors, avatar_actors):
        for actor in object_actors:
            actor.SetVisibility(self.show_object.isChecked())
            actor.GetProperty().SetOpacity(self.object_opacity.value() / 100)
        for actor in avatar_actors:
            actor.SetVisibility(self.show_avatar.isChecked())
            actor.GetProperty().SetOpacity(self.avatar_opacity.value() / 100)


# ----- camera -----------------------------------------------------------------------------------
class ViewportCamera:
    """Keyboard (and shared) camera control for a VTK renderer: orbit about the surface at the
    screen center, pan, zoom to the screen center and reset to the starting view."""

    def __init__(self, renderer, render, actors=lambda: []):
        self.renderer, self.render, self.actors = renderer, render, actors
        self.start = None

    def remember_start(self):
        import vtk
        self.renderer.ResetCamera()
        self.start = vtk.vtkCamera()
        self.start.DeepCopy(self.renderer.GetActiveCamera())

    def reset(self):
        if self.start is not None:
            self.renderer.GetActiveCamera().DeepCopy(self.start)
        self.renderer.ResetCamera()
        # DeepCopy does not mark the camera modified: mark it, so the next draw uses the reset view.
        self.renderer.GetActiveCamera().Modified()
        self.renderer.ResetCameraClippingRange()
        self.render()

    def center_pivot(self):
        import vtk
        renderer = self.renderer
        origin, size = renderer.GetOrigin(), renderer.GetSize()
        x, y = origin[0] + size[0] * .5, origin[1] + size[1] * .5
        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.SetTolerance(.002)
        for actor in self.actors():
            if actor.GetVisibility():
                picker.AddPickList(actor)
        if picker.Pick(x, y, 0, renderer):
            return picker.GetPickPosition()
        return renderer.GetActiveCamera().GetFocalPoint()

    def pan(self, horizontal, vertical):
        import numpy as np
        camera = self.renderer.GetActiveCamera()
        direction = np.array(camera.GetDirectionOfProjection())
        up = np.array(camera.GetViewUp())
        right = np.cross(direction, up)
        right /= max(np.linalg.norm(right), 1e-12)
        up = np.cross(right, direction)
        span = 2 * camera.GetParallelScale() if camera.GetParallelProjection() else (
            2 * camera.GetDistance() * math.tan(math.radians(camera.GetViewAngle()) * .5))
        delta = (right * horizontal + up * vertical) * span * .05
        camera.SetPosition(*(np.array(camera.GetPosition()) + delta))
        camera.SetFocalPoint(*(np.array(camera.GetFocalPoint()) + delta))

    def move(self, mode, horizontal, vertical):
        import vtk
        camera = self.renderer.GetActiveCamera()
        if mode == "orbit":
            camera.SetFocalPoint(*self.center_pivot())
            camera.Azimuth(-horizontal * .25)
            camera.Elevation(vertical * .25)
            camera.OrthogonalizeViewUp()
        elif mode == "pan":
            self.pan(horizontal, vertical)
        else:
            origin, size = self.renderer.GetOrigin(), self.renderer.GetSize()
            position = [int(origin[0] + size[0] / 2), int(origin[1] + size[1] / 2)]
            vtk.vtkInteractorStyleTrackballCamera.DollyToPosition(horizontal, position, self.renderer)
        self.renderer.ResetCameraClippingRange()
        self.render()

    def action(self, name):
        if name in CAMERA_ACTIONS:
            self.move(*CAMERA_ACTIONS[name])
            return True
        if name == "reset_view":
            self.reset()
            return True
        return False


# ----- keyboard shortcuts --------------------------------------------------------------------
def typing_focus():
    return isinstance(QApplication.focusWidget(), (QAbstractSpinBox, QLineEdit, QComboBox))


class Shortcuts:
    """Bind keybindings.json ("common" + ``tool``) to ``actions`` ({name: callable}); camera
    actions go to ``camera`` (a ViewportCamera). F1 and ? show the list. Unknown actions are
    ignored (they belong to other tools); bad or duplicate keys are reported in ``errors``."""

    def __init__(self, window, tool, actions, camera=None, path=None, allowed=None):
        self.window, self.tool, self.actions, self.camera = window, tool, dict(actions), camera
        self.allowed = allowed
        bindings, self.errors = keybindings.load(tool, path or keybindings.PATH)
        self.bindings, self.shortcuts, assigned = {}, [], set()
        for help_key in ("F1", "?"):
            self._add(help_key, "show_help", assigned)
        for action, keys in bindings.items():
            known = action in self.actions or (camera is not None and (action in CAMERA_ACTIONS
                                                                       or action == "reset_view"))
            if not known:
                continue
            for key in keys:
                if not self._add(key, action, assigned):
                    self.errors.append(f"{action}: {key}")

    def _add(self, key, action, assigned):
        sequence = QKeySequence(key)
        canonical = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        if sequence.isEmpty() or not canonical or canonical in assigned:
            return False
        assigned.add(canonical)
        shortcut = QShortcut(sequence, self.window)
        shortcut.setAutoRepeat(action in CAMERA_ACTIONS)
        shortcut.activated.connect(lambda action=action: self.invoke(action))
        self.shortcuts.append(shortcut)
        self.bindings.setdefault(action, []).append(canonical)
        return True

    def invoke(self, action, from_button=False):
        if action == "show_help":
            self.show_help()
            return
        if typing_focus() and not from_button:          # keys typed into a number box are not commands
            return
        if self.allowed is not None and not self.allowed(action):
            return
        if action in self.actions:
            self.actions[action]()
        elif self.camera is not None:
            self.camera.action(action)

    def key_for(self, action):
        keys = self.bindings.get(action)
        return keys[0] if keys else ""

    def help_rows(self):
        return [(ACTION_NAMES.get(action, action.replace("_", " ").capitalize()), ", ".join(keys))
                for action, keys in self.bindings.items()]

    def help_html(self):
        rows = "".join(f"<tr><td>{name}</td><td><b>{keys}</b></td></tr>" for name, keys in self.help_rows())
        mouse = ("<p>Mouse: left-drag orbit · middle-drag pan · wheel zoom (left-drag edits when an editing "
                 "mode is on).</p>")
        problems = ("<p style='color:#ff6b6b'>Could not load: " + ", ".join(self.errors) + "</p>") if self.errors else ""
        return (f"<table cellspacing=6>{rows}</table>{mouse}{problems}"
                f"<p>Keys come from <code>{keybindings.PATH}</code> (\"common\" plus \"{self.tool}\"). "
                "Edit it and reopen the tool to change them.</p>")

    def show_help(self):
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Keyboard shortcuts")
        layout = QVBoxLayout(dialog)
        text = QTextBrowser()
        text.setHtml(self.help_html())
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(460, 520)
        dialog.open()
        self.dialog = dialog
        return dialog

    def help_button(self):
        button = QPushButton("Keyboard shortcuts (F1)")
        button.setToolTip("\n".join(f"{keys}: {name}" for name, keys in self.help_rows()))
        button.clicked.connect(self.show_help)
        return button


# ----- viewport control bar ------------------------------------------------------------------
VIEW_BUTTONS = (("Orbit", (("◀", "orbit_left"), ("▶", "orbit_right"), ("▲", "orbit_up"), ("▼", "orbit_down"))),
                ("Pan", (("⇦", "pan_left"), ("⇨", "pan_right"), ("⇧", "pan_up"), ("⇩", "pan_down"))),
                ("Zoom", (("−", "zoom_out"), ("+", "zoom_in"))),
                ("", (("⌂", "reset_view"),)))


def add_view_toolbar(shortcuts, status):
    """Orbit/pan/zoom/reset buttons in a strip above the viewport, right-aligned. Buttons call the same
    actions as the keys (``shortcuts``: a Shortcuts); holding a button repeats it."""
    holder = status.parentWidget()
    bar = QWidget(holder)
    row = QHBoxLayout(bar)
    row.setContentsMargins(6, 2, 6, 2)
    row.setSpacing(2)
    row.addStretch()
    buttons = {}
    for title, items in VIEW_BUTTONS:
        if title:
            label = QLabel(title)
            label.setStyleSheet("font-style: italic; margin-left: 8px;")
            row.addWidget(label)
        for text, action in items:
            button = QToolButton(bar)
            button.setText(text)
            button.setAutoRaise(True)
            button.setAutoRepeat(action != "reset_view")
            button.setAutoRepeatDelay(250)
            button.setAutoRepeatInterval(40)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)        # keep keyboard focus where it was
            key = shortcuts.key_for(action)
            button.setToolTip(ACTION_NAMES.get(action, action) + (f" ({key})" if key else ""))
            button.clicked.connect(lambda checked=False, action=action: shortcuts.invoke(action, from_button=True))
            row.addWidget(button)
            buttons[action] = button
    holder.layout().insertWidget(0, bar)
    bar.buttons = buttons
    return bar


# ----- Playback of a recorded animation --------------------------------------------------------
class PlaybackBar(QWidget):
    """Play / Pause, a frame slider and the frame number for a finished animation.
    ``show(frame)`` draws one DAZ frame. Playback shows every frame in order at up to the animation's frame
    rate, without skipping frames; a looped
    result repeats without its last frame (it is the same moment as the first)."""

    def __init__(self, show, parent=None):
        super().__init__(parent)
        from PySide6.QtCore import QTimer
        self.show_frame = show
        self.fps, self.looped = 30., False
        # Play and the frame number on one line, the slider on its own full-width line below.
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 2, 0, 2)
        row = QHBoxLayout()
        self.button = QPushButton("Play")
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._changed)
        self.slider.sliderPressed.connect(self.stop)
        self.label = QLabel()
        self.label.setMinimumWidth(70)
        row.addWidget(self.button)
        row.addWidget(self.label, 1)
        column.addLayout(row)
        column.addWidget(self.slider)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.clear()

    def set_animation(self, start, end, fps, looped=False):
        """Enable playback of DAZ frames start..end and show the first frame."""
        self.stop()
        self.fps, self.looped = float(fps), bool(looped)
        self.slider.blockSignals(True)
        self.slider.setRange(int(start), int(end))
        self.slider.setValue(int(start))
        self.slider.blockSignals(False)
        self.setEnabled(True)
        self._changed(int(start))

    def clear(self):
        self.stop()
        self.setEnabled(False)
        self.label.setText("")

    def playing(self):
        return self.timer.isActive()

    def toggle(self):
        if self.playing():
            self.stop()
            return
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(self.slider.minimum())
        self.button.setText("Pause")
        self.timer.start(max(1, int(round(1000 / self.fps))))

    def stop(self):
        self.timer.stop()
        self.button.setText("Play")

    def _tick(self):
        start, end = self.slider.minimum(), self.slider.maximum()
        frame = self.slider.value() + 1
        if self.looped and frame > end:
            frame = start              # every frame plays: the last one leads into frame 0
        self.slider.setValue(min(frame, end))
        if frame >= end and not self.looped:
            self.stop()

    def _changed(self, frame):
        self.label.setText(f"Frame {frame}")
        self.show_frame(frame)


def return_format_box(settings, key="return/animation_format"):
    """"DAZ return format" for an animation (Hair has its own): DAZ native
    (DUF) or FBX. Remembered in ``settings``. Only
    animations sent to DAZ use it; standalone export keeps OBJ/FBX."""
    box = QComboBox()
    box.addItem("DAZ native (DUF)", "duf")
    box.addItem("FBX (compatibility)", "fbx")
    box.setCurrentIndex(max(0, box.findData(settings.value(key, "duf"))))
    box.currentIndexChanged.connect(lambda *_: settings.setValue(key, box.currentData()))
    box.setToolTip("How an animation is sent back to DAZ. DAZ native loads quickly; FBX is much slower "
                   "for large meshes.")
    return box
