# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Surface colors and textures for the selector's View (Spin256 · spin256.com/tools).

The DAZ capture records each surface's base color and base color texture in the session ("surfaces": token, name,
color 0-255 or null, texture path or null). The OBJ gives each polygon's surface (usemtl) and UVs. Preview only:
nothing here changes what is split or sent to DAZ."""
from pathlib import Path

import numpy as np

FLAT, COLORS, TEXTURES = "flat", "colors", "textures"
UNKNOWN_COLOR = (170, 170, 170)        # a surface without a recorded color
PIECE_TINT = .45                       # how much of the piece color shows over a surface color or texture


class Surfaces:
    """Per-polygon surface index and UVs of an OBJ, with each surface's recorded look."""

    def __init__(self, obj_path, looks):
        tokens, face_surface, face_uvs, uvs = [], [], [], []
        index = {}
        current = -1
        with Path(obj_path).open(encoding="utf-8-sig") as source:
            for line in source:
                fields = line.partition("#")[0].split()
                if not fields:
                    continue
                if fields[0] == "vt":
                    uvs.append((float(fields[1]), float(fields[2])))
                elif fields[0] == "usemtl" and len(fields) > 1:
                    current = index.setdefault(fields[1], len(tokens))
                    if current == len(tokens):
                        tokens.append(fields[1])
                elif fields[0] == "f":
                    corners = [token.split("/") for token in fields[1:]]
                    face_surface.append(current)
                    face_uvs.append([int(c[1]) - 1 for c in corners]
                                    if all(len(c) > 1 and c[1] for c in corners) else None)
        by_token = {look.get("token"): look for look in looks or ()}
        self.tokens = tokens
        self.face_surface = np.array(face_surface, dtype=np.int64)
        self.face_uvs = face_uvs
        self.uvs = np.array(uvs, dtype=np.float64).reshape(-1, 2)
        self.colors = np.array([tuple(by_token.get(t, {}).get("color") or UNKNOWN_COLOR) for t in tokens] or
                               [UNKNOWN_COLOR], dtype=np.float64)
        self.textures = [by_token.get(t, {}).get("texture") or None for t in tokens]

    @classmethod
    def from_session(cls, obj_path, session):
        looks = session.get("surfaces") if isinstance(session, dict) else None
        return cls(obj_path, looks) if looks else None

    def has_textures(self):
        return any(path and Path(path).is_file() for path in self.textures)

    def face_colors(self, faces):
        """Surface color of each face (n x 3, 0-255); faces without a surface get the unknown color."""
        surface = self.face_surface[np.asarray(faces, dtype=np.int64)]
        colors = np.empty((len(surface), 3))
        colors[:] = UNKNOWN_COLOR
        known = surface >= 0
        colors[known] = self.colors[surface[known]]
        return colors


def tinted(base, tint, amount=PIECE_TINT):
    """``base`` (n x 3) mixed with ``tint`` (n x 3, or None rows kept as base)."""
    return np.clip(base * (1. - amount) + np.asarray(tint, dtype=np.float64) * amount, 0, 255)


def textured_meshes(mesh, surfaces, faces):
    """One textured mesh per surface with a readable texture, for the given faces.

    Returns [(texture path, points (k x 3), tcoords (k x 2), polygons [corner index lists], face ids)].
    Corners are separate points so UV seams stay sharp. Faces whose surface has no texture or no UVs are left out
    (the caller draws them with their surface color)."""
    groups = {}
    for face in faces:
        surface = int(surfaces.face_surface[face])
        uv = surfaces.face_uvs[face] if face < len(surfaces.face_uvs) else None
        path = surfaces.textures[surface] if surface >= 0 else None
        if path is None or uv is None:
            continue
        groups.setdefault(path, []).append(face)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    out = []
    for path, ids in groups.items():
        if not Path(path).is_file():
            continue
        corners = [mesh.faces[f] for f in ids]
        uv_corners = [surfaces.face_uvs[f] for f in ids]
        flat = np.fromiter((v for face in corners for v in face), dtype=np.int64)
        flat_uv = np.fromiter((u for face in uv_corners for u in face), dtype=np.int64)
        polygons, start = [], 0
        for face in corners:
            polygons.append(list(range(start, start + len(face))))
            start += len(face)
        out.append((path, vertices[flat], surfaces.uvs[flat_uv], polygons, ids))
    return out


UV_MAP_SIZES = (512, 1024, 2048, 4096, 8192)


def uv_map_image(surfaces, faces, size, color=(0, 0, 0), width=1.):
    """A line drawing of the faces' UVs over a transparent background, as a QImage.

    UV (0, 0) is the image's bottom-left corner, as textures are mapped. A face in another UV tile (UDIM) is drawn in
    its own 0-1 square. Faces without UVs are left out. Returns (image, faces drawn)."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(*color))
    pen.setWidthF(width)
    pen.setCosmetic(True)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    drawn = 0
    try:
        for face in faces:
            indices = surfaces.face_uvs[face] if face < len(surfaces.face_uvs) else None
            if indices is None:
                continue
            uv = surfaces.uvs[indices]
            uv = uv - np.floor(uv.min(axis=0) + 1e-9)          # its own 0-1 tile
            painter.drawPolygon(QPolygonF([QPointF(u * size, (1. - v) * size) for u, v in uv]))
            drawn += 1
    finally:
        painter.end()
    return image, drawn
