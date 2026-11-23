# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""File exports and an opt-in, shell-free callback for non-DAZ bridges."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import numpy as np

from drape_fbx import write_fbx
from standalone_project import UNITS

# Retain handles for detached receivers without waiting for their application lifetime.
CALLBACK_PROCESSES = []


def output_paths(project, output, water=None):
    output = Path(output).resolve()
    if output.suffix.casefold() not in ('.obj', '.fbx'):
        raise ValueError('Choose an .obj or .fbx output file.')
    paths = []
    for i, entry in enumerate(project.entries):
        label = re.sub(r'[^\w.-]+', '-', entry['label']).strip('.-')[:60] or 'garment'
        paths.append(output if len(project.entries) == 1 else
                     output.with_name(f'{output.stem}-{i+1:02d}-{label}{output.suffix}'))
    result = Path(project.config.get('result') or output.with_suffix('.result.json')).resolve()
    water_path = output.with_name(output.stem+'-water.obj') if water else None
    all_paths = paths + ([water_path] if water_path else []) + [result]
    if len(set(all_paths)) != len(all_paths): raise ValueError('Output paths must be distinct.')
    originals = {Path(p).resolve() for k in ('props', 'avatars', 'garments') for p in project.config.get(k, [])}
    for folder in project.config.get('sequences', []): originals.update(p.resolve() for p in Path(folder).glob('*.obj'))
    if any(p in originals for p in all_paths): raise ValueError('Export cannot replace an input file.')
    if any(p.parent != output.parent for p in all_paths):
        raise ValueError('The result manifest and exported files must be in the same folder.')
    return paths, result, water_path


def write_obj(template, target, points_cm, units):
    points = iter(np.asarray(points_cm) / UNITS[units])
    count = 0
    with open(template, encoding='utf-8-sig') as inp, open(target, 'w', encoding='utf-8', newline='\n') as out:
        for line in inp:
            if line.split()[:1] == ['v']:
                try: point = next(points)
                except StopIteration: raise ValueError('Export vertex count does not match input.')
                out.write('v %.9g %.9g %.9g\n' % tuple(point))
                count += 1
            else: out.write(line)
    if count != len(points_cm): raise ValueError('Export vertex count does not match input.')


def callback_command(config, result):
    if config.get('after_export') and config.get('callback'):
        raise ValueError('Choose after_export or callback, not both.')
    if config.get('after_export'):
        script = Path(config['after_export']).resolve()
        if not script.is_file() or script.suffix.casefold() != '.py':
            raise ValueError('The after-export script must be an existing Python file.')
        python = Path(sys.executable)
        if python.name.casefold() == 'pythonw.exe': python = python.with_name('python.exe')
        return [str(python), str(script), str(result)]
    command = config.get('callback')
    if command is None: return None
    if not isinstance(command, list) or not command or not all(isinstance(a, str) and a for a in command):
        raise ValueError('Callback must be a non-empty command argument array.')
    if not any('{result}' in a for a in command):
        raise ValueError('Callback must include a {result} argument.')
    return [a.replace('{result}', str(result)) for a in command]


def export_result(project, output, vertices, animation=None, water=None, overwrite=False):
    paths, result_path, water_path = output_paths(project, output, water)
    if animation is not None and paths[0].suffix.casefold() != '.fbx':
        raise ValueError('Animation output uses FBX. Choose .fbx or export a static OBJ.')
    vertices = np.asarray(vertices)
    if vertices.shape != project.inputs['garment'][0].shape or not np.isfinite(vertices).all():
        raise ValueError('Invalid result geometry.')
    command = callback_command(project.config, result_path)
    targets = paths + ([water_path] if water_path else []) + [result_path]
    if not overwrite and any(p.exists() for p in targets):
        raise ValueError('Output already exists. Choose another name or allow overwrite.')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'animation' if animation is not None else 'static'
    frames = animation
    if frames is None:
        frames = SimpleNamespace(count=1, seen={0}, start_frame=0, end_frame=0,
                                 fps=float(project.config.get('fps', 30)), get=lambda sample: vertices)
    payload = dict(version=1, status='complete', export_id=str(uuid.uuid4()), output_mode=mode,
                   coordinates=dict(up_axis='Y', space='world', units='cm' if paths[0].suffix.lower()=='.fbx'
                                    else project.config.get('units', 'cm')), outputs=[],
                   fps=frames.fps, start_frame=frames.start_frame, end_frame=frames.end_frame)
    with tempfile.TemporaryDirectory(prefix='.drape-export-', dir=result_path.parent) as temp:
        stage = Path(temp)
        staged = []
        for i, (entry, target, bounds) in enumerate(zip(project.entries, paths, project.inputs['garment_ranges'])):
            dest = stage/target.name
            start, end = bounds
            if target.suffix.casefold() == '.fbx':
                write_fbx(dest, dict(path=entry['path'], vertex_count=end-start), frames, bounds)
            else:
                write_obj(entry['path'], dest, vertices[start:end], project.config.get('units', 'cm'))
            staged.append((dest, target))
            payload['outputs'].append(dict(path=str(target), source=entry['source'], label=entry['label'],
                                           vertex_count=end-start))
        if water_path:
            height, x, z, extent = (float(water[k]) for k in ('height', 'x', 'z', 'extent'))
            if not np.isfinite([height, x, z, extent]).all() or extent <= 0:
                raise ValueError('Invalid water reference plane.')
            scale = UNITS[project.config.get('units', 'cm')]
            points = np.asarray([(x-extent,height,z-extent), (x+extent,height,z-extent),
                                 (x+extent,height,z+extent), (x-extent,height,z+extent)]) / scale
            dest = stage/water_path.name
            dest.write_text('o Water\n'+''.join('v %.9g %.9g %.9g\n' % tuple(p) for p in points)+
                            'f 1 4 3 2\n', encoding='utf-8')
            staged.append((dest, water_path))
            payload['water_plane'] = dict(path=str(water_path), height=height/scale,
                                         units=project.config.get('units', 'cm'))
        manifest = stage/result_path.name
        manifest.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        staged.append((manifest, result_path))
        backups, committed = {}, []
        try:
            # Withdraw any earlier completion marker until the whole new export commits.
            for i, target in enumerate(targets):
                if target.exists():
                    backup = stage/f'backup-{i}'
                    shutil.copy2(target, backup)
                    backups[target] = backup
            if result_path.exists(): result_path.unlink()
            for source, target in staged:
                os.replace(source, target)
                committed.append(target)
        except BaseException:
            for target in committed:
                if target not in backups: target.unlink(missing_ok=True)
            for target, backup in backups.items(): os.replace(backup, target)
            raise
    if command:
        # The callback runs only after the manifest is committed. Do not wait for
        # the receiving application's lifetime, and never invoke a command shell.
        log_path = result_path.with_suffix('.callback.log')
        try:
            with log_path.open('ab') as log:
                process = subprocess.Popen(command, cwd=result_path.parent, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, close_fds=True,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                CALLBACK_PROCESSES[:] = [p for p in CALLBACK_PROCESSES if p.poll() is None]
                CALLBACK_PROCESSES.append(process)
        except OSError as error:
            raise ValueError(f'Files exported to {result_path}, but the callback could not start: {error}') from error
    return payload, result_path
