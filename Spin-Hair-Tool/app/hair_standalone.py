# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Standalone Hair: OBJ/FBX files in, OBJ/FBX files out (Spin256 · spin256.com/tools).

The chosen files are converted to Y-up centimeters and written as a capture session in the same form the DAZ capture
writes, so the window, the simulation and the result writers are exactly those used with DAZ. Export runs the same
writers into a scratch folder, converts OBJ results back to the chosen units and commits the files, a result manifest
and an optional callback (no shell)."""
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np

from project_settings import session_directory

UNITS = {'cm': 1., 'm': 100., 'mm': .1}
CALLBACK_PROCESSES = []


def natural_key(path):
    return [int(s) if s.isdigit() else s.casefold() for s in re.split(r'(\d+)', str(path))]


def normalize_obj(source, target, scale):
    """OBJ (Y-up world coordinates) -> centimeters, resolving relative indices; UVs and surfaces kept."""
    counts = [0, 0]
    tokens = []
    with Path(source).open(encoding='utf-8-sig') as inp, Path(target).open('w', encoding='utf-8', newline='\n') as out:
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
            elif kind == 'vt':
                counts[1] += 1
                out.write('vt %s %s\n' % (fields[1], fields[2] if len(fields) > 2 else '0'))
            elif kind == 'f':
                corners = []
                for corner in fields[1:]:
                    parts = corner.split('/')
                    vertex = int(parts[0])
                    vertex = vertex if vertex > 0 else counts[0] + vertex + 1
                    text = str(vertex)
                    if len(parts) > 1 and parts[1]:
                        uv = int(parts[1])
                        text += '/' + str(uv if uv > 0 else counts[1] + uv + 1)
                    corners.append(text)
                out.write('f ' + ' '.join(corners) + '\n')
            elif kind == 'usemtl':
                token = re.sub(r'\s+', '_', ' '.join(fields[1:])) or 'surface'
                if token not in tokens:
                    tokens.append(token)
                out.write('usemtl ' + token + '\n')
            elif kind == 'mtllib':
                # Materials and textures stay with the original asset.
                names = line.strip()[len('mtllib'):].strip()
                exact = Path(source).parent / names
                paths = [exact] if exact.exists() else [Path(source).parent / name for name in fields[1:]]
                out.write('mtllib ' + ' '.join(str(p.resolve()) for p in paths) + '\n')
            elif kind != 'vn':
                out.write(line.rstrip() + '\n')
    if not counts[0]:
        raise ValueError('The OBJ has no vertices: ' + str(source))
    return tokens


def obj_tokens(path):
    tokens = []
    with Path(path).open(encoding='utf-8-sig') as source:
        for line in source:
            if line.startswith('usemtl'):
                token = line.split()[1] if len(line.split()) > 1 else ''
                if token not in tokens:
                    tokens.append(token)
    return tokens


def write_frame(path, points_cm):
    with Path(path).open('w', encoding='utf-8', newline='\n') as out:
        out.writelines('v %.9g %.9g %.9g\n' % tuple(p) for p in points_cm)


class Project:
    def __init__(self, folder, config, session, hair):
        self.folder, self.config, self.session, self.hair = folder, config, session, hair

    def close(self):
        self.folder.cleanup()


def prepare(config, progress=lambda text: None, cancelled=lambda: False):
    """Convert the chosen files and write a capture session. Returns a Project; ``project.session`` is its path."""
    config = dict(config)
    units = config.get('units', 'cm')
    fps = float(config.get('fps', 30))
    animated = config.get('output_mode', 'static') == 'animation'
    if units not in UNITS or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError('Choose valid OBJ units and a frame rate between 0 and 240.')
    avatar, sequence = config.get('avatar'), config.get('avatar_sequence')
    if bool(avatar) == bool(sequence):
        raise ValueError('Choose one avatar: an OBJ or FBX file, or an OBJ sequence folder.')
    if not config.get('hair'):
        raise ValueError('Choose at least one hair OBJ or FBX.')
    folder = tempfile.TemporaryDirectory(prefix='standalone-', dir=session_directory())
    root = Path(folder.name)
    stem = root / 'session'
    sources, meshes = set(), []
    try:
        def load(source, role, animate=False):
            if cancelled():
                raise ValueError('Import canceled.')
            source = Path(source).resolve()
            if not source.is_file():
                raise ValueError('File not found: ' + str(source))
            if source in sources:
                raise ValueError('Each file can have only one role: ' + str(source))
            sources.add(source)
            progress('Loading ' + source.name)
            index = len(meshes)
            path = Path(f'{stem}-{index}-0.obj')
            frames = []
            if source.suffix.casefold() == '.fbx':
                from fbx_input import convert
                data = convert(source, root / f'fbx-{index}', fps, animate, config.get('take') if animate else None,
                               progress, cancelled)
                shutil.copyfile(data['base'], path)
                tokens = obj_tokens(path)
                frames = data['samples'] if animate and len(data['samples']) > 1 else []
            elif source.suffix.casefold() == '.obj':
                tokens = normalize_obj(source, path, UNITS[units])
            else:
                raise ValueError('Use OBJ or FBX files: ' + str(source))
            entry = dict(id=index + 1, role=role, label=source.stem, path=str(path), source=str(source))
            if role == 'garment':
                from hair_session import read_obj
                entry.update(vertex_count=len(read_obj(path)[0]),
                             materials=[dict(token=t, name=t) for t in tokens],
                             return_path=f'{stem}-draped-{index}.obj', animation_path=f'{stem}-draped-{index}.fbx')
            meshes.append(entry)
            return entry, frames

        for source in config['hair']:
            load(source, 'garment')
        if sequence:
            files = sorted(Path(sequence).glob('*.obj'), key=natural_key)
            if len(files) < 2:
                raise ValueError('An OBJ sequence folder needs at least two .obj files.')
            entry, _ = load(files[0], 'avatar')
            entry['label'], entry['source'] = Path(sequence).name, str(Path(sequence).resolve())
            frames = [None] + [str(f) for f in files[1:]]
        else:
            entry, frames = load(avatar, 'avatar', animated)
        moving = {len(meshes) - 1: frames}
        for source in config.get('colliders', []):
            entry, frames = load(source, 'collision', animated)
            moving[len(meshes) - 1] = frames
        animated = animated and any(len(frames) > 1 for frames in moving.values())
        if config.get('output_mode') == 'animation' and not animated:
            raise ValueError('Animation needs an animated FBX avatar or an OBJ sequence folder.')
        count = max((len(f) for f in moving.values()), default=1)
        session = dict(version=2, frame=0, mode='timeline' if animated else 'current_frame',
                       output_mode='animation' if animated else 'static', live_return=True,
                       session_id=str(stem), result_path=f'{stem}-result.json',
                       water_return_path=f'{stem}-water.obj', meshes=meshes)
        if animated:
            from hair_session import read_obj
            for index, frames in moving.items():
                entry = meshes[index]
                if frames and len(frames) != count:
                    raise ValueError('Every moving object needs the same number of frames.')
                samples = [entry['path']]
                base_count = len(read_obj(entry['path'])[0])
                for n in range(1, count):
                    if cancelled():
                        raise ValueError('Import canceled.')
                    if not frames:
                        samples.append(entry['path'])          # a still object keeps its pose on every frame
                        continue
                    source = frames[n]
                    if source.endswith('.npy'):
                        points = np.load(source, allow_pickle=False) * 100.
                    else:
                        from hair_session import read_obj
                        points = read_obj(source)[0] * UNITS[units]
                    if len(points) != base_count:
                        raise ValueError(f'{entry["label"]} changed its vertex count at frame {n}.')
                    target = f'{stem}-{index}-{n}.obj'
                    write_frame(target, points)
                    samples.append(target)
                    if n % 10 == 0:
                        progress(f'Preparing {entry["label"]}: frame {n + 1}/{count}')
                entry['samples'] = samples
                if entry['role'] == 'collision' and not frames:
                    entry['role'] = 'fixed'
                    entry.pop('samples')
            session.update(start_frame=0, end_frame=count - 1, fps=fps)
        path = Path(f'{stem}.json')
        path.write_text(json.dumps(session), encoding='utf-8')
        return Project(folder, config, path, [m for m in meshes if m['role'] == 'garment'])
    except BaseException:
        folder.cleanup()
        raise


def scale_obj(path, factor):
    if factor == 1.:
        return
    temporary = Path(str(path) + '.scale')
    with Path(path).open(encoding='utf-8') as inp, temporary.open('w', encoding='utf-8', newline='\n') as out:
        for line in inp:
            if line.startswith('v '):
                x, y, z = (float(v) * factor for v in line.split()[1:4])
                out.write(f'v {x:.9g} {y:.9g} {z:.9g}\n')
            else:
                out.write(line)
    os.replace(temporary, path)


def callback_command(config, result):
    if config.get('after_export') and config.get('callback'):
        raise ValueError('Choose after_export or callback, not both.')
    if config.get('after_export'):
        script = Path(config['after_export']).resolve()
        if not script.is_file() or script.suffix.casefold() != '.py':
            raise ValueError('The after-export script must be an existing Python file.')
        python = Path(sys.executable)
        if python.name.casefold() == 'pythonw.exe':
            python = python.with_name('python.exe')
        return [str(python), str(script), str(result)]
    command = config.get('callback')
    if command is None:
        return None
    if not isinstance(command, list) or not command or not all(isinstance(a, str) and a for a in command):
        raise ValueError('Callback must be a non-empty command argument array.')
    if not any('{result}' in a for a in command):
        raise ValueError('Callback must include a {result} argument.')
    return [a.replace('{result}', str(result)) for a in command]


def export(project, inputs, output, vertices, frames=None, water=None, overwrite=False):
    """Write the hair (one file per hair object), grown hair and the optional water plane next to ``output``, then a
    result manifest. Animation exports FBX (centimeters); still results OBJ (the chosen units) or FBX."""
    from drape_return import write_animation_result, write_static_result
    output = Path(output).resolve()
    if output.suffix.casefold() not in ('.obj', '.fbx'):
        raise ValueError('Choose an .obj or .fbx output file.')
    if frames is not None and output.suffix.casefold() != '.fbx':
        raise ValueError('Animation exports FBX. Choose a .fbx file.')
    config, units = project.config, project.config.get('units', 'cm')
    hair = project.hair
    if not hair and not inputs.get('grown'):
        raise ValueError('There is no hair to export.')
    targets = []
    for i, entry in enumerate(hair):
        label = re.sub(r'[^\w.-]+', '-', entry['label']).strip('.-')[:60] or 'hair'
        targets.append(output if len(hair) == 1 else output.with_name(f'{output.stem}-{i + 1:02d}-{label}{output.suffix}'))
    grown = inputs.get('grown') or []
    grown_targets = [output.with_name(f'{output.stem}-grown-{i + 1}.obj') for i in range(len(grown))]
    water_target = output.with_name(output.stem + '-water.obj') if water else None
    result = Path(config.get('result') or output.with_suffix('.result.json')).resolve()
    every = targets + grown_targets + ([water_target] if water_target else []) + [result]
    if len(set(every)) != len(every):
        raise ValueError('Output paths must be distinct.')
    originals = {Path(p).resolve() for p in [*config.get('hair', []), *config.get('colliders', [])] +
                 ([config['avatar']] if config.get('avatar') else [])}
    if any(p in originals for p in every):
        raise ValueError('Export cannot replace an input file.')
    if any(p.parent != output.parent for p in every):
        raise ValueError('The result manifest and exported files must be in the same folder.')
    if not overwrite and any(p.exists() for p in every):
        raise ValueError('Output already exists. Choose another name or allow overwrite.')
    command = callback_command(config, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hair-export-', dir=output.parent) as temp:
        stage = Path(temp)
        stem = str(stage / 'export')
        garments = [dict(entry, return_path=f'{stem}-{i}.obj', animation_path=f'{stem}-{i}.fbx')
                    for i, entry in enumerate(hair)]
        spec = dict(session_id=stem, result_path=f'{stem}-writer.json', garments=garments, return_frame=0,
                    water_return_path=f'{stem}-water.obj', animation_format='fbx')
        if water:
            spec['water_plane'] = water
        ranges = inputs['ranges']
        staged = []
        if frames is not None:
            write_animation_result(spec, frames, ranges)
            for entry, target in zip(garments, targets):
                staged.append((Path(entry['animation_path']), target))
        else:
            end = grown[0]['start'] if grown else len(vertices)
            if hair:
                write_static_result(spec, vertices[:end], ranges, cap=inputs.get('cap'))
            if grown:
                from hair_grow_return import add_grown
                if not hair:
                    Path(spec['result_path']).write_text('{}', encoding='utf-8')
                add_grown(spec, vertices, inputs)
            for entry, target in zip(garments, targets):
                source = Path(entry['return_path'])
                if target.suffix.casefold() == '.fbx':
                    from drape_fbx import write_fbx
                    from hair_session import read_obj
                    points = read_obj(source)[0]
                    still = type('Still', (), dict(count=1, seen={0}, start_frame=0, end_frame=0, vertex_count=len(points),
                                                   fps=float(config.get('fps', 30)), get=lambda self, s: points))()
                    fbx = source.with_suffix('.fbx')
                    write_fbx(fbx, dict(entry, path=str(source), vertex_count=len(points)), still, (0, len(points)))
                    staged.append((fbx, target))
                else:
                    scale_obj(source, 1. / UNITS[units])
                    staged.append((source, target))
            for i, target in enumerate(grown_targets):
                source = stage / f'export-grown-{i + 1}.obj'
                scale_obj(source, 1. / UNITS[units])
                staged.append((source, target))
        if water_target:
            source = Path(spec['water_return_path'])
            scale_obj(source, 1. / UNITS[units])
            staged.append((source, water_target))
        animated = frames is not None
        payload = dict(version=1, status='complete', export_id=str(uuid.uuid4()),
                       output_mode='animation' if animated else 'static',
                       coordinates=dict(up_axis='Y', space='world',
                                        units='cm' if output.suffix.casefold() == '.fbx' else units),
                       fps=frames.fps if animated else float(config.get('fps', 30)),
                       start_frame=frames.start_frame if animated else 0,
                       end_frame=frames.end_frame if animated else 0,
                       outputs=[dict(path=str(t), source=e['source'], label=e['label'], vertex_count=e['vertex_count'])
                                for e, t in zip(hair, targets)] +
                               [dict(path=str(t), label=g['name'] + ' (grown hair)') for g, t in zip(grown, grown_targets)])
        if water_target:
            payload['water_plane'] = dict(path=str(water_target))
        manifest = stage / 'manifest.json'
        manifest.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        staged.append((manifest, result))
        result.unlink(missing_ok=True)          # no completion marker until every file is in place
        for source, target in staged:
            os.replace(source, target)
    if command:
        log = result.with_suffix('.callback.log')
        try:
            with log.open('ab') as handle:
                process = subprocess.Popen(command, cwd=result.parent, stdin=subprocess.DEVNULL, stdout=handle,
                                           stderr=handle, close_fds=True,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            CALLBACK_PROCESSES[:] = [p for p in CALLBACK_PROCESSES if p.poll() is None]
            CALLBACK_PROCESSES.append(process)
        except OSError as error:
            raise ValueError(f'Files exported to {result}, but the callback could not start: {error}') from error
    return payload, result
