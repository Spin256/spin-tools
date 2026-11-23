# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Verify the shared environment and import each tool in an isolated process."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent
TOOLS={'Spin-Drape-Tool':['drape_helper','warp.sim'],
       'Spin-Soft-Prop-Tool':['soft_app'],
       'Spin-Soft-Body-Tool':['body_app'],
       'Spin-Hair-Tool':['hair_app','hair_solver','warp'],
       'Spin-Prop-Split-Tool':['selector','multi_split']}


def pins(path):
    result={}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line=line.partition('#')[0].strip()
        if line:
            name,version=line.split('==')
            result[name.casefold().replace('_','-')]=version
    return result


def validate_requirements(root=ROOT):
    shared=pins(root/'requirements-lock.txt')
    for tool in TOOLS:
        for name,version in pins(root/tool/'requirements-lock.txt').items():
            if shared.get(name)!=version:
                raise RuntimeError(f'{tool} requires {name}=={version}; the shared environment is incompatible.')
    return shared


def check(skip_gpu_check=False):
    env=(ROOT/'.venv').resolve()
    if Path(sys.prefix).resolve()!=env:raise RuntimeError('Run with the .venv in the Spin Tools folder (.venv/Scripts/python.exe).')
    requirements=validate_requirements()
    for name,version in requirements.items():
        if metadata.version(name)!=version:raise RuntimeError(f'{name} must be {version}.')
    for name in ('numpy','PySide6','vtkmodules','warp','ufbx'):
        module=importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(env):raise RuntimeError(name+' loaded outside the suite environment.')
    import warp as wp
    wp.config.kernel_cache_dir=str(ROOT/'temp/.warp-check-cache');wp.init()
    cuda=wp.is_cuda_available()
    if not cuda and not skip_gpu_check:raise RuntimeError('Drape and Soft Prop require an NVIDIA CUDA GPU and driver.')
    if cuda:
        import numpy as np
        array=wp.array(np.asarray([1,2,3],dtype=np.float32),dtype=float,device='cuda:0')
        np.testing.assert_array_equal(array.numpy(),[1,2,3])
    for tool,names in TOOLS.items():
        code='import sys,importlib; sys.path.insert(0,sys.argv[1]); [importlib.import_module(n) for n in sys.argv[2:]]'
        process=subprocess.run([sys.executable,'-c',code,str(ROOT/tool/'app'),*names],
            cwd=ROOT/tool,capture_output=True,text=True,env=dict(os.environ,QT_QPA_PLATFORM='offscreen'),timeout=120)
        if process.returncode:raise RuntimeError(tool+' import check failed:\n'+process.stdout+process.stderr)
    return dict(python=sys.executable,environment=str(env),packages=requirements,cuda_available=cuda,tools=list(TOOLS))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report');parser.add_argument('--skip-gpu-check',action='store_true')
    args=parser.parse_args();report=check(args.skip_gpu_check)
    if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: shared dependencies, CUDA and all {len(TOOLS)} helper-app imports are available.')
