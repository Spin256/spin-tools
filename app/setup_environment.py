# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Install one shared Spin Tools environment; publish tool settings after validation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ('Spin-Drape-Tool','Spin-Soft-Prop-Tool','Spin-Prop-Split-Tool','Spin-Soft-Body-Tool','Spin-Hair-Tool')


def supported_python():
    if sys.platform != 'win32' or struct.calcsize('P') != 8 or not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError('Use 64-bit Windows CPython 3.10–3.13. Python 3.10 is the tested configuration.')


def publish_settings(root):
    previous={tool:(root/tool/'settings.json').read_bytes() for tool in TOOLS}
    try:
        for tool in TOOLS: publish_tool_settings(root/tool)
    except Exception:
        for tool,data in previous.items():
            target=root/tool/'settings.json'
            staged=target.with_suffix('.rollback.tmp');staged.write_bytes(data);os.replace(staged,target)
        raise


def publish_tool_settings(root):
    """Switch launchers only after install validation; preserve unrelated settings."""
    path = root/'settings.json'
    settings = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    if not isinstance(settings, dict): raise ValueError('settings.json must contain an object.')
    previous = path.read_bytes() if path.exists() else None
    settings.setdefault('session_folder', 'temp')
    settings['python_executable'] = '../.venv/Scripts/pythonw.exe'
    data = (json.dumps(settings, indent=2)+'\n').encode('utf-8')
    if data == previous: return
    if previous is not None:
        backup = root/'temp'/('settings-before-install-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(previous)
    handle, staged = tempfile.mkstemp(prefix='.settings-install-', suffix='.tmp', dir=root)
    try:
        with os.fdopen(handle, 'wb') as output: output.write(data)
        os.replace(staged, path)
    finally:
        Path(staged).unlink(missing_ok=True)


def install(root=ROOT, wheelhouse=None, skip_gpu_check=False):
    supported_python()
    root = Path(root).resolve()
    env = root/'.venv'
    python = env/'Scripts/python.exe'
    requirements = root/('requirements-lock.txt' if (root/'requirements-lock.txt').exists() else 'requirements.txt')
    from check_install import validate_requirements
    validate_requirements(root)
    # Refuse to reinterpret another folder as an environment; never auto-delete it.
    if env.exists() and any(env.iterdir()) and not (env/'pyvenv.cfg').is_file():
        raise RuntimeError('.venv is not a recognized virtual environment. Move it aside before running setup.')
    if (env/'pyvenv.cfg').exists() and not python.is_file():
        raise RuntimeError('The existing .venv is incomplete. Move it aside, then run setup again.')
    if not requirements.is_file(): raise RuntimeError('Missing requirements.txt.')
    lock = root/'.spin-install.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Another setup may be running. If it stopped, remove .spin-install.lock and retry.')
    os.close(fd)
    try:
        (root/'temp').mkdir(exist_ok=True)
        with (root/'temp/install.log').open('a', encoding='utf-8') as log:
            def run(command):
                print('Running: '+' '.join(str(a) for a in command), flush=True)
                log.write('\n'+' '.join(str(a) for a in command)+'\n')
                log.flush()
                process = subprocess.Popen([str(a) for a in command], cwd=root,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding='utf-8', errors='replace', env=dict(os.environ, PYTHONIOENCODING='utf-8',
                        PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CACHE_DIR=str(root/'temp/pip-cache')))
                with process.stdout:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                if process.wait(): raise RuntimeError('Setup step failed. See temp/install.log. Existing launcher settings were preserved.')
            if not python.exists():
                print('Creating local .venv...', flush=True)
                venv.EnvBuilder(with_pip=True, system_site_packages=False).create(env)
            run([python, '-c', 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,13), "Unsupported venv Python"'])
            command = [python, '-m', 'pip', 'install', '--only-binary=:all:', '--require-virtualenv', '-r', requirements]
            if wheelhouse:
                command += ['--no-index', '--find-links', str(Path(wheelhouse).resolve())]
            run(command)
            run([python, '-m', 'pip', 'check'])
            check = [python, root/'app/check_install.py', '--report', root/'temp/install-report.json']
            if skip_gpu_check: check.append('--skip-gpu-check')
            run(check)
            publish_settings(root)
            print('\nInstalled. All helper apps now use the .venv in this folder.\n'
                  'For DAZ: open Install Spin Tools Menu.dsa to register the suite under Scripts > Spin Tools.', flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Set up the Spin Tools shared Python environment.')
    parser.add_argument('--wheelhouse', help='Optional offline folder of downloaded dependency wheels')
    parser.add_argument('--skip-gpu-check', action='store_true', help='For development machines without an NVIDIA GPU')
    args = parser.parse_args()
    try:
        install(wheelhouse=args.wheelhouse, skip_gpu_check=args.skip_gpu_check)
    except Exception as error:
        print('\nSetup failed: '+str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__': sys.exit(main())
