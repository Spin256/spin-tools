# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Write compressed binary FBX blend-shape animation, in DAZ world centimetres.

One base mesh and one shape per subsequent DAZ frame. Linear one-hot weights
reproduce the samples and interpolate between them without cubic overshoot.
"""
from collections import Counter
from contextlib import contextmanager
import struct
import zlib
import numpy as np


class I64(int):
    pass


def prop(value):
    if isinstance(value, bytes):
        return b"R" + struct.pack("<I", len(value)) + value
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        code = {"<i4": b"i", "<i8": b"l", "<f4": b"f", "<f8": b"d"}[a.dtype.str]
        raw = a.tobytes()
        encoded = zlib.compress(raw, 6) if len(raw) > 128 else raw
        return code + struct.pack("<III", a.size, int(len(raw) > 128), len(encoded)) + encoded
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + struct.pack("<I", len(raw)) + raw
    if isinstance(value, float):
        return b"D" + struct.pack("<d", value)
    if isinstance(value, bool):
        return b"C" + struct.pack("?", value)
    if isinstance(value, I64) or not -(2**31) <= value < 2**31:
        return b"L" + struct.pack("<q", value)
    return b"I" + struct.pack("<i", value)


class BinaryFBX:
    def __init__(self, stream):
        self.f = stream
        self.f.write(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400))

    @contextmanager
    def node(self, name, *values):
        start = self.f.tell()
        raw_name = name.encode("ascii")
        properties = b"".join(prop(v) for v in values)
        self.f.write(struct.pack("<IIIB", 0, len(values), len(properties), len(raw_name)))
        self.f.write(raw_name)
        self.f.write(properties)
        child_start = self.f.tell()
        yield
        if self.f.tell() != child_start or not values or name in ("AnimationStack", "AnimationLayer"):
            self.f.write(bytes(13))
        end = self.f.tell()
        if end >= 2**32:
            raise ValueError("This animation exceeds the FBX 4 GB file limit. Return a shorter range.")
        self.f.seek(start)
        self.f.write(struct.pack("<I", end))
        self.f.seek(end)

    def leaf(self, name, *values):
        with self.node(name, *values):
            pass

    def finish(self):
        self.f.write(bytes(13))
        self.f.write(bytes.fromhex("fabcab09d0c8d466b176fb831cf7267e"))
        self.f.write(bytes(4))
        self.f.write(bytes((-self.f.tell()) % 16 or 16))
        self.f.write(struct.pack("<I", 7400) + bytes(120))
        self.f.write(bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b"))


def read_obj(path):
    points, uvs, faces, materials = [], [], [], []
    material = "drape_surface_0"
    with open(path, encoding="utf-8-sig") as source:
        for line in source:
            f = line.split()
            if not f:
                continue
            if f[0] == "v":
                points.append([float(v) for v in f[1:4]])
            elif f[0] == "vt":
                uvs.append([float(v) for v in f[1:3]])
            elif f[0] == "usemtl":
                material = f[1]
            elif f[0] == "f":
                if material not in materials:
                    materials.append(material)
                corners = [v.split("/") for v in f[1:]]
                faces.append(([int(v[0]) - 1 for v in corners],
                              [int(v[1]) - 1 if len(v) > 1 and v[1] else -1 for v in corners],
                              materials.index(material)))
    return np.asarray(points, dtype=np.float64), uvs, faces, materials


def write_fbx(path, entry, frames, vertex_range):
    start, end = vertex_range
    source, uvs, faces, materials = read_obj(entry["path"])
    base = frames.get(0)[start:end].astype(np.float64)
    if source.shape != base.shape or len(base) != entry["vertex_count"] or not faces:
        raise ValueError("The animation does not match its captured garment.")
    if frames.seen != set(range(frames.count)):
        raise ValueError("The garment animation is incomplete.")
    # Stable identifiers and names avoid dependence on source scene names/Unicode labels.
    model, geom, blend, stack, layer, root = 1000, 1001, 1002, 1003, 1004, 1005
    material_ids = [2000 + i for i in range(len(materials))]
    ids = [(10000 + i * 4, 10001 + i * 4, 10002 + i * 4, 10003 + i * 4)
           for i in range(1, frames.count)]
    ticks = lambda frame: int(round(frame * 46186158000 / frames.fps))
    first_tick, last_tick = ticks(frames.start_frame), ticks(frames.end_frame)
    connections = [("OO", root, 0), ("OO", model, root), ("OO", geom, model), ("OO", blend, geom), ("OO", layer, stack)]
    with open(path, "w+b") as stream:
        w = BinaryFBX(stream)
        def p(name, kind, *values, flags="", subtype=""):
            if kind == "KTime":
                values = tuple(I64(v) for v in values)
            w.leaf("P", name, kind, subtype, flags, *values)
        def obj(kind, ident, name, subtype):
            # Binary object names encode name followed by NUL/SOH and class.
            classname = {"AnimationStack": "AnimStack", "AnimationLayer": "AnimLayer",
                         "AnimationCurveNode": "AnimCurveNode", "AnimationCurve": "AnimCurve"}.get(kind, kind)
            if subtype == "BlendShapeChannel":
                classname = "SubDeformer"
            return w.node(kind, I64(ident), name + "\x00\x01" + classname, subtype)
        with w.node("FBXHeaderExtension"):
            w.leaf("FBXHeaderVersion", 1003)
            w.leaf("FBXVersion", 7400)
            with w.node("CreationTimeStamp"):
                for key, value in (("Version", 1000), ("Year", 1970), ("Month", 1), ("Day", 1),
                                   ("Hour", 10), ("Minute", 0), ("Second", 0), ("Millisecond", 0)):
                    w.leaf(key, value)
            w.leaf("Creator", "DAZ Drape Tool")
        w.leaf("FileId", bytes.fromhex("28b32aebb624ccc2bfc8b02aa92bfcf1"))
        w.leaf("CreationTime", "1970-01-01 10:00:00:000")
        w.leaf("Creator", "DAZ Drape Tool")
        with w.node("GlobalSettings"):
            w.leaf("Version", 1000)
            with w.node("Properties70"):
                for key, value in (("UpAxis", 1), ("UpAxisSign", 1), ("FrontAxis", 2),
                                   ("FrontAxisSign", 1), ("CoordAxis", 0), ("CoordAxisSign", 1)):
                    p(key, "int", value, subtype="Integer")
                p("UnitScaleFactor", "double", 1.0, subtype="Number")
                p("TimeMode", "enum", 14)
                p("CustomFrameRate", "double", float(frames.fps), subtype="Number")
                p("TimeSpanStart", "KTime", first_tick, subtype="Time")
                p("TimeSpanStop", "KTime", last_tick, subtype="Time")
        with w.node("Documents"):
            w.leaf("Count", 1)
            with w.node("Document", I64(1), "", "Scene"):
                with w.node("Properties70"):
                    p("SourceObject", "object")
                    p("ActiveAnimStackName", "KString", "DrapeAnimation")
                w.leaf("RootNode", I64(0))
        with w.node("References"):
            pass
        counts = Counter(Model=2, Geometry=frames.count, Deformer=frames.count,
                         Material=len(materials), AnimationStack=1, AnimationLayer=1,
                         AnimationCurveNode=frames.count - 1, AnimationCurve=frames.count - 1)
        with w.node("Definitions"):
            w.leaf("Version", 100)
            w.leaf("Count", sum(counts.values()))
            for name, count in counts.items():
                with w.node("ObjectType", name):
                    w.leaf("Count", count)
                    template = {"Model": "FbxNode", "Geometry": "FbxMesh", "Deformer": "FbxBlendShapeChannel",
                                "Material": "FbxSurfacePhong", "AnimationStack": "FbxAnimStack",
                                "AnimationLayer": "FbxAnimLayer", "AnimationCurveNode": "FbxAnimCurveNode"}.get(name)
                    if template:
                        with w.node("PropertyTemplate", template):
                            with w.node("Properties70"):
                                if name == "Model":
                                    p("Lcl Translation", "Lcl Translation", 0.0, 0.0, 0.0, flags="A")
                                    p("Lcl Rotation", "Lcl Rotation", 0.0, 0.0, 0.0, flags="A")
                                    p("Lcl Scaling", "Lcl Scaling", 1.0, 1.0, 1.0, flags="A")
        with w.node("Objects"):
            with obj("Model", root, "DrapeGarment", "Null"):
                w.leaf("Version", 232)
                with w.node("Properties70"):
                    p("Lcl Translation", "Lcl Translation", 0.0, 0.0, 0.0, flags="A")
                    p("Lcl Rotation", "Lcl Rotation", 0.0, 0.0, 0.0, flags="A")
                    p("Lcl Scaling", "Lcl Scaling", 1.0, 1.0, 1.0, flags="A")
            with obj("Model", model, "DrapeGarment.Shape", "Mesh"):
                w.leaf("Version", 232)
                with w.node("Properties70"):
                    p("DefaultAttributeIndex", "int", 0, subtype="Integer")
                    p("Lcl Translation", "Lcl Translation", 0.0, 0.0, 0.0, flags="A")
                    p("Lcl Rotation", "Lcl Rotation", 0.0, 0.0, 0.0, flags="A")
                    p("Lcl Scaling", "Lcl Scaling", 1.0, 1.0, 1.0, flags="A")
                w.leaf("Shading", True)
                w.leaf("Culling", "CullingOff")
            with obj("Geometry", geom, "DrapeGarment", "Mesh"):
                w.leaf("Vertices", base.astype("<f8").ravel())
                indices, uv_indices, face_materials = [], [], []
                for vertices, uv, mat in faces:
                    indices.extend(vertices[:-1] + [-vertices[-1] - 1])
                    uv_indices.extend(uv)
                    face_materials.append(mat)
                w.leaf("PolygonVertexIndex", np.asarray(indices, dtype="<i4"))
                w.leaf("GeometryVersion", 124)
                normals = np.zeros_like(base)
                for vertices, _, _ in faces:
                    normal = np.cross(base[vertices[1]] - base[vertices[0]], base[vertices[2]] - base[vertices[0]])
                    normals[vertices] += normal
                lengths = np.linalg.norm(normals, axis=1)
                normals /= np.maximum(lengths[:, None], 1e-20)
                with w.node("LayerElementNormal", 0):
                    w.leaf("Version", 101)
                    w.leaf("Name", "")
                    w.leaf("MappingInformationType", "ByVertice")
                    w.leaf("ReferenceInformationType", "Direct")
                    w.leaf("Normals", normals.astype("<f8").ravel())
                has_uv = bool(uvs) and min(uv_indices) >= 0
                if has_uv:
                    with w.node("LayerElementUV", 0):
                        w.leaf("Version", 101)
                        w.leaf("Name", "DrapeUV")
                        w.leaf("MappingInformationType", "ByPolygonVertex")
                        w.leaf("ReferenceInformationType", "IndexToDirect")
                        w.leaf("UV", np.asarray(uvs, dtype="<f8").ravel())
                        w.leaf("UVIndex", np.asarray(uv_indices, dtype="<i4"))
                with w.node("LayerElementMaterial", 0):
                    w.leaf("Version", 101)
                    w.leaf("Name", "")
                    w.leaf("MappingInformationType", "ByPolygon")
                    w.leaf("ReferenceInformationType", "IndexToDirect")
                    w.leaf("Materials", np.asarray(face_materials, dtype="<i4"))
                with w.node("Layer", 0):
                    w.leaf("Version", 100)
                    for name in ["LayerElementNormal", "LayerElementMaterial"] + (["LayerElementUV"] if has_uv else []):
                        with w.node("LayerElement"):
                            w.leaf("Type", name)
                            w.leaf("TypedIndex", 0)
            for ident, name in zip(material_ids, materials):
                with obj("Material", ident, name, ""):
                    w.leaf("Version", 102)
                    w.leaf("ShadingModel", "phong")
                    w.leaf("MultiLayer", 0)
                connections.append(("OO", ident, model))
            with obj("Deformer", blend, "DrapeBlendShapes", "BlendShape"):
                w.leaf("Version", 100)
            with obj("AnimationStack", stack, "DrapeAnimation", ""):
                with w.node("Properties70"):
                    for key, value in (("LocalStart", first_tick), ("LocalStop", last_tick),
                                       ("ReferenceStart", first_tick), ("ReferenceStop", last_tick)):
                        p(key, "KTime", value, subtype="Time")
            with obj("AnimationLayer", layer, "BaseLayer", ""):
                pass
            for sample, (shape, channel, curve_node, curve) in enumerate(ids, 1):
                delta = frames.get(sample)[start:end].astype(np.float64) - base
                changed = np.flatnonzero(np.any(delta != 0, axis=1)).astype("<i4")
                # Empty shapes are accepted by FBX; retain the channel to keep sample timing explicit.
                name = "DrapeFrame_%d" % (frames.start_frame + sample)
                with obj("Geometry", shape, name, "Shape"):
                    w.leaf("Version", 100)
                    w.leaf("Indexes", changed)
                    w.leaf("Vertices", delta[changed].astype("<f8").ravel())
                with obj("Deformer", channel, name, "BlendShapeChannel"):
                    w.leaf("Version", 100)
                    w.leaf("DeformPercent", 0.0)
                    w.leaf("FullWeights", np.asarray([100.0], dtype="<f8"))
                with obj("AnimationCurveNode", curve_node, name, ""):
                    with w.node("Properties70"):
                        p("d|DeformPercent", "Number", 0.0, flags="A")
                key_samples = [sample - 1, sample] + ([sample + 1] if sample + 1 < frames.count else [])
                with obj("AnimationCurve", curve, name, ""):
                    w.leaf("Default", 0.0)
                    w.leaf("KeyVer", 4009)
                    w.leaf("KeyTime", np.asarray([ticks(frames.start_frame + k) for k in key_samples], dtype="<i8"))
                    w.leaf("KeyValueFloat", np.asarray([0, 100] + ([0] if len(key_samples) == 3 else []), dtype="<f4"))
                    w.leaf("KeyAttrFlags", np.full(len(key_samples), 4, dtype="<i4"))
                    w.leaf("KeyAttrDataFloat", np.zeros(len(key_samples) * 4, dtype="<f4"))
                    w.leaf("KeyAttrRefCount", np.ones(len(key_samples), dtype="<i4"))
                connections.extend([("OO", shape, channel), ("OO", channel, blend),
                                    ("OO", curve_node, layer), ("OP", curve_node, channel, "DeformPercent"),
                                    ("OP", curve, curve_node, "d|DeformPercent")])
        with w.node("Connections"):
            for connection in connections:
                w.leaf("C", *(I64(v) if isinstance(v, int) else v for v in connection))
        with w.node("Takes"):
            w.leaf("Current", "DrapeAnimation")
            with w.node("Take", "DrapeAnimation"):
                w.leaf("FileName", "DrapeAnimation.tak")
                w.leaf("LocalTime", first_tick, last_tick)
                w.leaf("ReferenceTime", first_tick, last_tick)
        w.finish()
