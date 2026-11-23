# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""This PC's measured Drape speed, used to advise on heavy garments (Spin256 · spin256.com/tools).

Every run records how long preparation took and how fast frames ran, per simulated vertex, in
the suite settings (QSettings "Spin256/Spin Tools", group drape_performance), averaged over runs.
Before any run is measured, garments over FALLBACK_VERTICES are flagged.
"""
from PySide6.QtCore import QSettings

FALLBACK_VERTICES = 100_000
SLOW_PREPARATION = 20.0      # seconds before the first frame
SLOW_FRAMES = 10.0           # real seconds per simulated second
WEIGHT = .3                  # share of a new measurement in the running average
FPS = 60.0
MIN_VERTICES = 20_000        # smaller runs are dominated by fixed costs; not recorded
SETTINGS_FILE = None         # tests point this at a scratch .ini instead of the user's settings


def _settings():
    settings = (QSettings(str(SETTINGS_FILE), QSettings.Format.IniFormat) if SETTINGS_FILE
                else QSettings("Spin256", "Spin Tools"))
    settings.beginGroup("drape_performance")
    return settings


def _update(key, value):
    settings = _settings()
    old = settings.value(key, None, type=float)
    settings.setValue(key, value if not old else old * (1 - WEIGHT) + value * WEIGHT)


def record_preparation(vertices, seconds, proxy_full=0):
    """Seconds from Run to the first frame. With a proxy the build cost scales with the full mesh."""
    if vertices < MIN_VERTICES or seconds <= 0:
        return
    _update("proxy_preparation_per_full_vertex" if proxy_full else "preparation_per_vertex",
            seconds / (proxy_full or vertices))


def record_frames(vertices, frames, seconds):
    if vertices >= MIN_VERTICES and frames > 0 and seconds > 0:
        _update("frame_seconds_per_vertex", seconds / frames / vertices)


def predict(full, simulated, proxy):
    """(preparation seconds, real seconds per simulated second) or None per unknown value."""
    settings = _settings()
    per = settings.value("proxy_preparation_per_full_vertex" if proxy else "preparation_per_vertex", None, type=float)
    if proxy and not per:
        plain = settings.value("preparation_per_vertex", None, type=float)
        per = plain * .2 if plain else None             # rough until a proxy run is measured
    frame = settings.value("frame_seconds_per_vertex", None, type=float)
    preparation = per * (full if proxy else simulated) if per else None
    slowdown = frame * simulated * FPS if frame else None
    return preparation, slowdown


def short_advice(full, simulated, proxy):
    """A status-bar length version of advice() ("" when the garment should run comfortably)."""
    preparation, slowdown = predict(full, simulated, proxy)
    if not advice(full, simulated, proxy):
        return ""
    timing = ", ".join(([f"~{preparation:.0f} s to prepare"] if preparation is not None else []) +
                       ([f"{slowdown:.0f}× slower than real time" if slowdown >= 1 else "faster than real time"]
                        if slowdown is not None else []))
    if proxy:
        return f"Proxy {simulated:,} of {full:,} vertices" + (f": {timing}." if timing else ".")
    return (f"Large garment ({full:,} vertices)" + (f": {timing}." if timing else ": slow to simulate.") +
            " Try Proxy simulation.")


def advice(full, simulated, proxy):
    """Status-bar text for a heavy garment, or "" when it should run comfortably."""
    preparation, slowdown = predict(full, simulated, proxy)
    if proxy:
        if full <= simulated:
            return ""
        parts = [f"Proxy simulation: {simulated:,} of {full:,} vertices"]
        if preparation is not None:
            parts.append(f"about {preparation:.0f} s to prepare")
        if slowdown is not None:
            parts.append(f"{slowdown:.1f}× real time" if slowdown >= 1 else "faster than real time")
        return " · ".join(parts) + "."
    if preparation is None and slowdown is None:
        if full <= FALLBACK_VERTICES:
            return ""
        return (f"Large garment: {full:,} vertices — preparing and simulating will be slow. "
                "Proxy simulation (Simulation section) is much faster.")
    slow_preparation = preparation is not None and preparation > SLOW_PREPARATION
    slow_frames = slowdown is not None and slowdown > SLOW_FRAMES
    if not (slow_preparation or slow_frames):
        return ""
    parts = [f"Large garment: {full:,} vertices"]
    if preparation is not None:
        parts.append(f"about {preparation:.0f} s to prepare")
    if slowdown is not None:
        parts.append(f"{slowdown:.0f}× slower than real time on this PC")
    return " — ".join(parts[:1]) + " — " + ", ".join(parts[1:]) + ". Proxy simulation (Simulation section) is much faster."
