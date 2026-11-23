# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Isolated FBX-to-mesh adapter, kept apart from the DAZ bridge and the cloth solver."""
from pathlib import Path
import argparse
import json
import math
import re
import sys
import os
import subprocess
import numpy as np


def extract(scene):
    vertices, uvs, polygons = [], [], []
    holders = [scene]
    nodes = scene.nodes
    holders.append(nodes)
    for ni in range(len(nodes)):
        node = nodes[ni]
        holders.append(node)
        mesh = node.mesh
        if mesh is None or not node.visible:
            continue
        holders.append(mesh)
        attribute = mesh.skinned_position
        if not attribute.exists:
            attribute = mesh.vertex_position
        values = attribute.values
        holders.extend([attribute, values])
        points = np.asarray([tuple(value) for value in values], dtype=np.float64)
        if attribute.unique_per_vertex:
            points = points[:mesh.num_vertices]
        else:
            indices, first = attribute.indices, mesh.vertex_first_index
            holders.extend([indices, first])
            points = points[[indices[first[v]] for v in range(mesh.num_vertices)]]
        if mesh.skinned_is_local:
            matrix = np.asarray(node.geometry_to_world, dtype=float).T
            points = points @ matrix[:, :3].T + matrix[:, 3]
        offset, uv_offset = len(vertices), len(uvs)
        vertices.extend(points.tolist())
        uv = mesh.vertex_uv
        holders.append(uv)
        uv_indices = None
        if uv.exists:
            uv_values, uv_indices = uv.values, uv.indices
            holders.extend([uv_values, uv_indices])
            uvs.extend([tuple(value) for value in uv_values])
        faces, indices, materials, face_material = mesh.faces, mesh.vertex_indices, node.materials, mesh.face_material
        holders.extend([faces, indices, materials, face_material])
        material_names = []
        for mi in range(len(materials)):
            material_object = materials[mi]
            holders.append(material_object)
            material_names.append(material_object.name)
        for fi in range(len(faces)):
            face = faces[fi]
            if face.num_indices < 3:
                continue
            corners = range(face.index_begin, face.index_begin + face.num_indices)
            mi = int(face_material[fi]) if material_names else -1
            material = material_names[mi] if 0 <= mi < len(material_names) else 'surface'
            material = f'node{ni}_' + re.sub(r'\s+', '_', material)
            polygons.append(([int(indices[c]) + offset for c in corners],
                             [int(uv_indices[c]) + uv_offset if uv_indices is not None else -1 for c in corners], material))
    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(polygons) or not np.isfinite(points).all():
        raise ValueError('FBX contains no valid visible polygon meshes.')
    return points, uvs, polygons, holders


def write_obj(path, points, uvs, polygons):
    with Path(path).open('w', encoding='utf-8') as out:
        for p in points: out.write('v %.9g %.9g %.9g\n' % tuple(p))
        for uv in uvs: out.write('vt %.9g %.9g\n' % tuple(uv))
        material = None
        for vertices, uv_indices, name in polygons:
            if name != material:
                out.write('usemtl ' + name + '\n')
                material = name
            out.write('f ' + ' '.join(str(v+1) + ('/' + str(uv+1) if uv >= 0 else '')
                                     for v, uv in zip(vertices, uv_indices)) + '\n')


def _batch(source, target, fps=30., animate=False, take=None, first=0, count=24):
    import ufbx
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    scene = ufbx.load_file(str(source), target_axes=ufbx.axes_right_handed_y_up, target_unit_meters=.01)
    stacks = scene.anim_stacks
    stack = None
    stack_refs = []
    if len(stacks):
        for i in range(len(stacks)):
            candidate = stacks[i]
            stack_refs.append(candidate)
            if stack is None and (not take or candidate.name == take): stack = candidate
        if stack is None: raise ValueError('FBX animation take was not found: ' + str(take))
    elif take:
        raise ValueError('FBX contains no animation takes.')
    anim = stack.anim if stack is not None else scene.anim
    begin = stack.time_begin if stack is not None else 0.
    end = stack.time_end if animate and stack is not None else begin
    intervals = max(0, int(math.ceil((end - begin) * fps - 1e-7)))
    samples, topology = [], None
    keep_alive = [scene, stacks, stack_refs, stack, anim]
    for sample in range(first, min(intervals + 1, first + count)):
        time = min(end, begin + sample / fps)
        evaluated = scene.evaluate(anim, time, evaluate_skinning=True)
        points, uvs, polygons, holders = extract(evaluated)
        current = [p[0] for p in polygons]
        if topology is None:
            topology = current
            vertex_count = len(points)
            signature = dict(vertices=vertex_count, faces=current)
            if first == 0:
                write_obj(target / 'base.obj', points, uvs, polygons)
                (target/'topology.json').write_text(json.dumps(signature), encoding='utf-8')
            elif signature != json.loads((target/'topology.json').read_text(encoding='utf-8')):
                raise ValueError('Animated FBX changed topology between frame batches.')
        elif len(points) != vertex_count or current != topology:
            raise ValueError('Animated FBX changed mesh topology or visible mesh membership.')
        path = target / f'{sample:06d}.npy'
        np.save(path, points * .01)
        samples.append(str(path))
        keep_alive.extend(holders)
    result = dict(base=str(target/'base.obj'), samples=samples, fps=fps, source=str(source),
                  take=stack.name if stack is not None else None, start_time=begin, count=intervals+1)
    (target/f'batch-{first}.json').write_text(json.dumps(result), encoding='utf-8')
    # ufbx 0.0.5 has unsafe ownership on evaluated-scene destruction. This is a
    # dedicated, bounded native worker: retain its wrappers, flush files, then
    # let the OS release the process. Never use this path in the GUI process.
    if sys.stdout is not None: sys.stdout.flush()
    if sys.stderr is not None: sys.stderr.flush()
    os._exit(0)


def convert(source, target, fps=30., animate=False, take=None,
            progress=lambda text: None, cancelled=lambda: False):
    if not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError('Frame rate must be between 0 and 240.')
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    samples, first, total = [], 0, 1
    while first < total:
        if cancelled(): raise ValueError('Import canceled.')
        progress(f'Reading {Path(source).name}: frame {first+1}' + (f'/{total}' if first else ''))
        python = Path(sys.executable)
        if python.name.casefold() == 'pythonw.exe': python = python.with_name('python.exe')
        command = [str(python), str(Path(__file__).resolve()), str(Path(source).resolve()), str(target),
                   '--fps', str(fps), '--batch', str(first)]
        if animate: command.append('--animate')
        if take: command += ['--take', take]
        run = subprocess.run(command, capture_output=True, text=True,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if run.returncode:
            raise ValueError('FBX import failed: ' + (run.stderr.strip() or 'native reader stopped unexpectedly'))
        batch = target/f'batch-{first}.json'
        result = json.loads(batch.read_text(encoding='utf-8'))
        batch.unlink()
        if not result['samples']: raise ValueError('FBX reader produced no frames.')
        samples.extend(result['samples'])
        first += len(result['samples'])
        total = result['count']
    result['samples'] = samples
    (target/'asset.json').write_text(json.dumps(result), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--take')
    parser.add_argument('--batch', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.batch is not None:
            _batch(args.source, args.target, args.fps, args.animate, args.take, args.batch)
        else:
            convert(args.source, args.target, args.fps, args.animate, args.take)
    except Exception as error:
        if sys.stderr is not None: print(str(error), file=sys.stderr)
        sys.exit(1)
