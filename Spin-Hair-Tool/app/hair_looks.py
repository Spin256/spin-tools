# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Display looks for the Hair window: surface colors and textures (Spin256 · spin256.com/tools).

The capture records each surface's base color and color texture for the avatar and the hair (Hair_ExportMesh.dsa
"looks"), and writes the avatar once more with its UVs (look_path). A LookMesh draws a mesh in those colors or textures
with its corners split (so UV seams stay sharp) and follows the mesh as it moves: update(points). Preview only."""
import json
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

FLAT, COLORS, TEXTURES = "flat", "colors", "textures"
PLAIN = (170, 170, 170)


def session_looks(session_path):
    """(avatar entry or None, [hair entries]) with their looks, from the capture's session file."""
    try:
        data = json.loads(Path(session_path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return None, []
    meshes = data.get("meshes") or []
    avatar = next((m for m in meshes if m.get("role") == "avatar" and m.get("looks")), None)
    hair = [m for m in meshes if m.get("role") == "garment"]
    return avatar, hair


def read_look_obj(path):
    """Corners of an OBJ with UVs and surfaces: (vertex per corner, uv per corner (or -1), sizes, face token, uvs)."""
    corners, corner_uv, sizes, tokens, uvs = [], [], [], [], []
    token = ""
    with Path(path).open(encoding="utf-8-sig") as source:
        for line in source:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "vt":
                uvs.append((float(fields[1]), float(fields[2])))
            elif fields[0] == "usemtl":
                token = fields[1] if len(fields) > 1 else ""
            elif fields[0] == "f":
                parts = [f.split("/") for f in fields[1:]]
                corners.extend(int(p[0]) - 1 for p in parts)
                corner_uv.extend(int(p[1]) - 1 if len(p) > 1 and p[1] else -1 for p in parts)
                sizes.append(len(parts))
                tokens.append(token)
    return (np.array(corners, np.int64), np.array(corner_uv, np.int64), np.array(sizes, np.int64), tokens,
            np.array(uvs, np.float64).reshape(-1, 2))


class Textures:
    """VTK textures by file, read once (None when unreadable)."""

    def __init__(self):
        self.cache = {}

    def get(self, path):
        if not path:
            return None
        if path not in self.cache:
            texture = None
            if Path(path).is_file():
                reader = vtk.vtkImageReader2Factory.CreateImageReader2(str(path))
                if reader is not None:
                    reader.SetFileName(str(path))
                    reader.Update()
                    if reader.GetOutput().GetNumberOfPoints() > 0:
                        texture = vtk.vtkTexture()
                        texture.SetInputConnection(reader.GetOutputPort())
                        texture.InterpolateOn(); texture.RepeatOn(); texture.MipmapOn()
            self.cache[path] = texture
        return self.cache[path]


class LookMesh:
    """A mesh drawn in surface colors or textures, one actor per texture (and one for the plain-colored faces).

    corner_vertex: mesh vertex of each corner; sizes: corners per face; face_group: each face's group; groups:
    [(color, texture path or None)]; corner_uvs: (C, 2) or None."""

    def __init__(self, renderer, corner_vertex, sizes, face_group, groups, corner_uvs, textures):
        self.renderer, self.textures = renderer, textures
        self.corner_vertex = np.asarray(corner_vertex, np.int64)
        self.sizes = np.asarray(sizes, np.int64)
        self.face_group = np.asarray(face_group, np.int64)
        self.groups = list(groups)
        self.corner_uvs = None if corner_uvs is None else np.asarray(corner_uvs, np.float64)
        self.points = vtk.vtkPoints()
        self.points.SetNumberOfPoints(len(self.corner_vertex))
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
        self.actors = []            # (actor, face ids)
        self.mode = None
        self.smooth = True          # smooth shading from the mesh's own vertex normals
        self.normals = vtk.vtkFloatArray()
        self.normals.SetNumberOfComponents(3)

    def update(self, points, normals=None):
        """Follow the mesh: each corner at its vertex's position, with its vertex's normal (smooth shading; corners
        are split for sharp UV seams, so without the mesh's normals every polygon would be lit flat)."""
        self.points.SetData(numpy_to_vtk(np.asarray(points, np.float32)[self.corner_vertex], deep=True))
        self.points.Modified()
        if normals is not None:
            self.normals.DeepCopy(numpy_to_vtk(np.asarray(normals, np.float32)[self.corner_vertex], deep=True))
            self.normals.Modified()
        for actor, _ in self.actors:
            data = actor.GetMapper().GetInput()
            data.GetPointData().SetNormals(self.normals if self.smooth and self.normals.GetNumberOfTuples() else None)
            data.Modified()
            if self.smooth:
                actor.GetProperty().SetInterpolationToGouraud()
            else:
                actor.GetProperty().SetInterpolationToFlat()

    def set_smooth(self, on):
        self.smooth = bool(on)
        for actor, _ in self.actors:
            data = actor.GetMapper().GetInput()
            data.GetPointData().SetNormals(self.normals if self.smooth and self.normals.GetNumberOfTuples() else None)
            data.Modified()
            if self.smooth:
                actor.GetProperty().SetInterpolationToGouraud()
            else:
                actor.GetProperty().SetInterpolationToFlat()

    def clear(self):
        for actor, _ in self.actors:
            self.renderer.RemoveActor(actor)
        self.actors = []
        self.mode = None

    def _polydata(self, faces, colors=None, uv=False):
        data = vtk.vtkPolyData()
        data.SetPoints(self.points)
        starts, sizes = self.offsets[faces], self.sizes[faces]
        connectivity = np.concatenate([np.arange(a, a + n) for a, n in zip(starts, sizes)]) if len(faces) \
            else np.zeros(0, np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        cells = vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(connectivity, deep=True))
        data.SetPolys(cells)
        if uv and self.corner_uvs is not None:
            data.GetPointData().SetTCoords(numpy_to_vtk(self.corner_uvs, deep=True))
        if colors is not None:
            data.GetCellData().SetScalars(numpy_to_vtk(np.asarray(colors, np.uint8), deep=True,
                                                       array_type=vtk.VTK_UNSIGNED_CHAR))
        return data

    def _actor(self, data, texture=None, color=None):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        if data.GetCellData().GetScalars() is not None:
            mapper.SetScalarModeToUseCellData(); mapper.SetColorModeToDirectScalars()
        else:
            mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if texture is not None:
            actor.SetTexture(texture)
        if color is not None:
            actor.GetProperty().SetColor(*[c / 255. for c in color])
        actor.GetProperty().BackfaceCullingOff()
        self.renderer.AddActor(actor)
        return actor

    def build(self, mode):
        """Actors for ``mode`` (COLORS or TEXTURES)."""
        self.clear()
        self.mode = mode
        colors = np.array([g[0] or PLAIN for g in self.groups] or [PLAIN], np.float64)
        faces = np.arange(len(self.sizes))
        if mode == COLORS or self.corner_uvs is None:
            group = np.clip(self.face_group, 0, len(colors) - 1)
            actor = self._actor(self._polydata(faces, colors[group]))
            self.actors.append((actor, faces))
            return
        textured = np.zeros(len(faces), bool)
        for index, (color, path) in enumerate(self.groups):
            texture = self.textures.get(path)
            mine = faces[self.face_group == index]
            if texture is None or not len(mine):
                continue
            textured[mine] = True
            self.actors.append((self._actor(self._polydata(mine, uv=True), texture=texture), mine))
        plain = faces[~textured]
        if len(plain):
            group = np.clip(self.face_group[plain], 0, len(colors) - 1)
            self.actors.append((self._actor(self._polydata(plain, colors[group])), plain))

    def face_of(self, actor, cell):
        """The mesh face under ``cell`` of one of this mesh's actors, or None."""
        for own, faces in self.actors:
            if own is actor:
                return int(faces[cell]) if 0 <= cell < len(faces) else None
        return None


def avatar_look(renderer, avatar_entry, vertex_count, textures):
    """A LookMesh of the avatar from its look OBJ, or None (missing or not matching the captured avatar)."""
    try:
        corners, corner_uv, sizes, tokens, uvs = read_look_obj(avatar_entry["look_path"])
    except (OSError, KeyError, ValueError, TypeError):
        return None
    if not len(corners) or corners.max() >= vertex_count:
        return None
    looks = avatar_entry.get("looks") or []
    index = {look.get("token"): k for k, look in enumerate(looks)}
    groups = [(tuple(look["color"]) if look.get("color") else None, look.get("texture")) for look in looks]
    face_group = np.array([index.get(t, -1) for t in tokens], np.int64)
    corner_uvs = uvs[np.clip(corner_uv, 0, max(len(uvs) - 1, 0))] if len(uvs) and (corner_uv >= 0).all() else None
    return LookMesh(renderer, corners, sizes, face_group, groups, corner_uvs, textures)


def hair_groups(surface_names, hair_entries, grown_parts=()):
    """Per hair surface name: (color, texture). Captured surfaces from their looks (names may carry "Object: "
    prefixes); grown surfaces ("Grown n", "Grown n secondary") from their materials."""
    by_name = {}
    for entry in hair_entries:
        label = entry.get("label", "Hair")
        for look in entry.get("looks") or []:
            value = (tuple(look["color"]) if look.get("color") else None, look.get("texture"))
            by_name.setdefault(look.get("name"), value)
            by_name[f"{label}: {look.get('name')}"] = value
    for index, part in enumerate(grown_parts):
        by_name[f"Grown {index + 1}"] = (tuple(part["color"]), None)
        by_name[f"Grown {index + 1} secondary"] = (tuple(part.get("color2", part["color"])), None)
    return [by_name.get(name, (None, None)) for name in surface_names]


def hair_look(renderer, inputs, hair_entries, textures):
    """A LookMesh of the hair (captured and grown) from the session inputs."""
    faces = np.asarray(inputs["faces"])
    if not len(faces):
        return None
    sizes = (faces >= 0).sum(axis=1)
    corners = faces[faces >= 0]
    uvs = inputs.get("vertex_uvs")
    corner_uvs = None
    if uvs is not None and len(uvs) == len(inputs["vertices"]) and np.isfinite(uvs).all():
        corner_uvs = np.asarray(uvs)[corners]
    groups = hair_groups(inputs["surface_names"], hair_entries, inputs.get("grown") or ())
    return LookMesh(renderer, corners, sizes, np.asarray(inputs["face_surface"]), groups, corner_uvs, textures)
