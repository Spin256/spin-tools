# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Standalone and custom-bridge export of split pieces, with an opt-in shell-free callback.

Writes ``<output stem>-NN-<name>.obj`` per piece plus a result manifest, all
committed together (an earlier export is restored if any write fails). The
callback starts only after the manifest exists, mirroring the Drape Tool's
contract: ``--after-export script.py`` runs ``python script.py <result.json>``.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

from multi_split import split_pieces

# Retain handles for detached receivers without waiting for their application lifetime.
CALLBACK_PROCESSES = []


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


def result_path(config, output):
    output = Path(output).resolve()
    return Path(config.get('result') or output.with_suffix('.result.json')).resolve()


def export_split(source, output, pieces, config=None, overwrite=False, allow_ngons=False):
    """Export ``pieces`` [(name, faces, pivot)] of ``source``; returns (payload, result path)."""
    config = config or {}
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.suffix.casefold() != '.obj':
        raise ValueError('Choose an .obj output file; each piece is written beside it.')
    result = result_path(config, output)
    if result.parent != output.parent:
        raise ValueError('The result manifest and exported files must be in the same folder.')
    command = callback_command(config, result)
    result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.split-export-', dir=result.parent) as temp:
        stage = Path(temp)
        records = split_pieces(source, stage / (output.stem + '.json'), pieces, allow_ngons=allow_ngons)
        targets = [output.with_name(r['file']) for r in records] + [result]
        if source in targets:
            raise ValueError('Export cannot replace the source mesh.')
        if len(set(targets)) != len(targets):
            raise ValueError('Output paths must be distinct.')
        if not overwrite and any(t.exists() for t in targets):
            raise ValueError('Output already exists. Choose another name or allow overwrite.')
        payload = dict(version=1, status='complete', export_id=str(uuid.uuid4()), source=str(source),
                       coordinates=dict(up_axis='Y', space='world', units='source'),
                       outputs=[dict(path=str(output.with_name(r['file'])), name=r['name'], faces=r['faces'],
                                     vertex_count=len(r['vertices']), pivot=r['pivot'],
                                     source_vertices=r['source_vertices']) for r in records])
        manifest = stage / result.name
        manifest.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        staged = [(stage / r['file'], output.with_name(r['file'])) for r in records] + [(manifest, result)]
        backups, committed = {}, []
        try:
            # Withdraw any earlier completion marker until the whole new export commits.
            for i, target in enumerate(targets):
                if target.exists():
                    backup = stage / f'backup-{i}'
                    shutil.copy2(target, backup)
                    backups[target] = backup
            if result.exists(): result.unlink()
            for staged_file, target in staged:
                os.replace(staged_file, target)
                committed.append(target)
        except BaseException:
            for target in committed:
                if target not in backups: target.unlink(missing_ok=True)
            for target, backup in backups.items(): os.replace(backup, target)
            raise
    if command:
        # The callback runs only after the manifest is committed. Do not wait for
        # the receiving application's lifetime, and never invoke a command shell.
        log_path = result.with_suffix('.callback.log')
        try:
            with log_path.open('ab') as log:
                process = subprocess.Popen(command, cwd=result.parent, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=log, close_fds=True,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                CALLBACK_PROCESSES[:] = [p for p in CALLBACK_PROCESSES if p.poll() is None]
                CALLBACK_PROCESSES.append(process)
        except OSError as error:
            raise ValueError(f'Files exported to {result}, but the callback could not start: {error}') from error
    return payload, result
