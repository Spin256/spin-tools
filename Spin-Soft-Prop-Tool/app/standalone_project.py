# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Standalone input adapters. The DAZ capture/session contract is not used here."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import tempfile
import numpy as np

from mesh_io import read_mesh
from project_settings import session_directory

UNITS = {'cm': 1., 'm': 100., 'mm': .1}


def natural_key(path):
    return [int(s) if s.isdigit() else s.casefold() for s in re.split(r'(\d+)', str(path))]


def normalize_obj(source, target, scale):
    """World-space Y-up OBJ -> cm; keep UVs/groups and resolve relative indices."""
    source, target = Path(source), Path(target)
    counts = [0, 0, 0]
    with source.open(encoding='utf-8-sig') as inp, target.open('w', encoding='utf-8', newline='\n') as out:
        for line in inp:
            fields = line.partition('#')[0].split()
            if not fields:
                continue
            kind = fields[0]
            if kind == 'v':
                point = np.asarray(fields[1:4], dtype=float) * scale
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError('Invalid OBJ vertex: ' + str(source))
                out.write('v %.9g %.9g %.9g\n' % tuple(point))
                counts[0] += 1
            elif kind in ('vt', 'vn'):
                counts[1 if kind == 'vt' else 2] += 1
                if kind == 'vt': out.write(line.rstrip() + '\n')
            elif kind == 'f':
                corners = []
                for corner in fields[1:]:
                    values = corner.split('/')
                    resolved = []
                    for i, value in enumerate(values[:2]):
                        if value:
                            index = int(value)
                            index = index if index > 0 else counts[i] + index + 1
                            if index <= 0: raise ValueError('Invalid OBJ face index.')
                            resolved.append(str(index))
                        else: resolved.append('')
                    corners.append('/'.join(resolved).rstrip('/'))
                out.write('f ' + ' '.join(corners) + '\n')
            elif kind == 'mtllib':
                # Keep material/texture lookup rooted at the original asset, not our cache.
                names = line.strip()[len('mtllib'):].strip()
                exact = source.parent / names
                paths = [exact] if exact.exists() else [source.parent / name for name in fields[1:]]
                out.write('mtllib ' + ' '.join(str(p.resolve()) for p in paths) + '\n')
            else:
                out.write(line.rstrip() + '\n')


class AssetMotion:
    def __init__(self, colliders, samples, fps):
        self.colliders, self.samples, self.fps = colliders, samples, fps
        lengths = {len(paths) for paths in samples.values()}
        if len(lengths) != 1 or next(iter(lengths)) < 2:
            raise ValueError('Moving objects must contain the same number of frames (at least two).')
        self.start_frame = 0
        self.end_frame = self.intervals = next(iter(lengths)) - 1
        self.duration = self.intervals / fps
        self.reset()

    def reset(self):
        self.cache = {(i, 0): self.colliders[i][0] for i in self.samples}

    def positions(self, index, sample):
        key = index, sample
        if key not in self.cache:
            points = np.load(self.samples[index][sample], allow_pickle=False)
            if points.shape != self.colliders[index][0].shape or not np.isfinite(points).all():
                raise ValueError('Invalid moving-object frame.')
            self.cache = {k: v for k, v in self.cache.items() if k[0] != index or k[1] in (sample-1, sample)}
            self.cache[key] = points
        return self.cache[key]

    def pair(self, index, interval):
        return (self.positions(index, min(interval, self.intervals)),
                self.positions(index, min(interval+1, self.intervals)))

    def preview(self, frame):
        return {i: self.positions(i, min(frame, self.intervals))*100 for i in self.samples}


@dataclass
class Project:
    cache: object
    config: dict
    entries: list
    inputs: dict

    def close(self):
        self.cache.cleanup()


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or config.get('version', 1) != 1:
        raise ValueError('Unsupported standalone project version.')
    def resolve(value):
        p = Path(value)
        return str((path.parent/p).resolve()) if not p.is_absolute() else str(p)
    for key in ('props', 'avatars', 'garments', 'sequences'):
        config[key] = [resolve(p) for p in config.get(key, [])]
    for key in ('output', 'result', 'after_export'):
        if config.get(key): config[key] = resolve(config[key])
    return config


def prepare_project(config, progress=lambda text: None, cancelled=lambda: False):
    config = dict(config)
    fps = float(config.get('fps', 30))
    units = config.get('units', 'cm')
    if units not in UNITS or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError('Choose valid OBJ units and a frame rate between 0 and 240.')
    if config.get('output_mode', 'static') not in ('static', 'animation'):
        raise ValueError('Output mode must be static or animation.')
    if not config.get('avatars') and not config.get('sequences'):
        raise ValueError('Choose at least one collision object.')
    if not config.get('garments'):
        raise ValueError('Choose at least one soft prop.')
    cache = tempfile.TemporaryDirectory(prefix='standalone-', dir=session_directory())
    folder = Path(cache.name)
    garments, collisions, samples = [], [], {}
    sources = set()
    try:
        def asset(source, role):
            if cancelled(): raise ValueError('Import cancelled.')
            source = Path(source).resolve()
            if not source.is_file(): raise ValueError('File not found: ' + str(source))
            if source in sources: raise ValueError('Each input must have only one role: ' + str(source))
            sources.add(source)
            progress('Loading ' + source.name)
            target = folder / str(len(sources))
            target.mkdir()
            if source.suffix.casefold() == '.fbx':
                from fbx_input import convert
                data = convert(source, target, fps, role == 'moving' and config.get('output_mode','static') == 'animation', config.get('take') if role == 'moving' else None,
                               progress, cancelled)
                base, frames = data['base'], data['samples']
            elif source.suffix.casefold() == '.obj':
                base = str(target/'base.obj')
                normalize_obj(source, base, UNITS[units])
                frames = []
            else:
                raise ValueError('Supported inputs are OBJ and FBX: ' + str(source))
            entry = dict(path=base, source=str(source), label=source.stem, role=role)
            points, faces = read_mesh(base)
            if not np.isfinite(points).all(): raise ValueError('Invalid mesh positions.')
            entry['mesh'] = points, faces
            entry['samples'] = frames if len(frames) > 1 else []
            return entry

        for source in config.get('garments', []): garments.append(asset(source, 'garment'))
        for source in config.get('props', []): collisions.append(asset(source, 'fixed'))
        for source in config.get('avatars', []): collisions.append(asset(source, 'moving'))
        for source in config.get('sequences', []):
            files = sorted(Path(source).glob('*.obj'), key=natural_key)
            if len(files) < 2: raise ValueError('An OBJ sequence folder needs at least two .obj files.')
            entry = asset(files[0], 'moving')
            entry['source'], entry['label'] = str(Path(source).resolve()), Path(source).name
            entry['samples'] = []
            for n, path in enumerate(files):
                if cancelled(): raise ValueError('Import cancelled.')
                progress(f'Loading {entry["label"]}: frame {n+1}/{len(files)}')
                points, faces = read_mesh(path)
                points *= UNITS[units]
                if points.shape != entry['mesh'][0].shape or not np.array_equal(faces, entry['mesh'][1]):
                    raise ValueError('OBJ sequence changed mesh topology: ' + str(path))
                if not np.isfinite(points).all(): raise ValueError('Invalid OBJ sequence vertex.')
                target = Path(entry['path']).parent/f'{n:06d}.npy'
                np.save(target, points)
                entry['samples'].append(str(target))
            collisions.append(entry)

        water = None
        vertices, faces, ranges, buttons, offset = [], [], [], [], 0
        from drape_fbx import read_obj
        for entry in garments:
            points, triangles = entry['mesh']
            vertices.append(points)
            faces.append(triangles+offset)
            ranges.append((offset, offset+len(points)))
            _, _, polygons, names = read_obj(entry['path'])
            buttons.extend(offset+v for v in sorted({v for p, _, m in polygons
                           if re.search(r'button|btn', names[m], re.I) for v in p}))
            offset += len(points)
        collider_meshes = [e['mesh'] for e in collisions]
        for i, entry in enumerate(collisions):
            if entry['samples']:
                samples[i] = entry['samples']
        motion = AssetMotion(collider_meshes, samples, fps) if samples else None
        inputs = dict(garment=(np.concatenate(vertices), np.concatenate(faces)), colliders=collider_meshes,
                      garment_label=', '.join(e['label'] for e in garments),
                      collision_labels=[e['label'] for e in collisions],
                      collision_roles=[e['role'] for e in collisions], garment_ranges=ranges,
                      button_vertices=buttons, water_source=water, motion=motion,
                      output_mode=config.get('output_mode', 'static'))
        if inputs['output_mode'] == 'animation' and motion is None:
            raise ValueError('Animation output requires an animated FBX or an OBJ sequence.')
        return Project(cache, config, garments, inputs)
    except BaseException:
        cache.cleanup()
        raise
