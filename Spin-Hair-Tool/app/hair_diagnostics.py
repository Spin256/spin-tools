# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Small per-session simulation trace; no meshes, never touches return files."""
import json
import os
from pathlib import Path
import numpy as np


class SimulationTrace:
    def __init__(self, path, settings):
        self.path = Path(path) if path else None
        self.settings = dict(settings)
        self.samples = []
        self.last_phase = None

    def record(self, sim, paused=False):
        if self.path is None:
            return
        phase = ('paused' if paused else 'settling' if sim.motion is None else
                 'initial settle' if not sim.playing else
                 'transition' if sim.time < sim.motion.duration-1e-6 else 'final pose')
        if phase == self.last_phase and sim.frame % 30:
            return
        self.last_phase = phase
        inv = sim.inv.numpy()
        free = inv > 0
        acceleration = sim.forces.accel.numpy()
        velocity = sim.vel.numpy()
        def stats(values):
            if not free.any():
                return [0.,0.,0.]
            return np.percentile(np.linalg.norm(values[free],axis=1),[0,50,95]).tolist()
        original = sim.wind_control
        if original is not None and hasattr(original,'control'):
            original = original.control
        sample = dict(phase=phase,simulation_frame=int(sim.frame),simulation_seconds=float(sim.frame/sim.fps),
            source_seconds=float(sim.time),free_particles=int(free.sum()),held_particles=int((~free).sum()),
            external_acceleration_cm_s2_min_median_p95=stats(acceleration),
            free_velocity_cm_s_min_median_p95=stats(velocity),
            gravity_cm_s2=sim.gravity_vector().tolist(),
            supported_gravity_cm_s2=sim.supported(sim.gravity_vector()).tolist(),
            wind=original.snapshot()[1] if original is not None else None,
            effective_wind=sim.wind_control.snapshot()[1] if sim.wind_control is not None else None,
            packed_wind=sim.forces.wind.numpy().tolist(),water=sim.forces.water.numpy().tolist())
        if sim.trapped_contact is not None:
            sample['trapped_particles']=int((sim.trapped_contact.owner.numpy()>=0).sum())
        sample['pull_holds']=len(sim.pull_holds)
        self.samples.append(sample)
        # Preserve the initial phases and the latest part of a long interactive run.
        if len(self.samples)>600:
            self.samples=self.samples[:300]+self.samples[-300:]
        temporary=self.path.with_suffix(self.path.suffix+'.tmp')
        try:
            temporary.write_text(json.dumps(dict(version=1,settings=self.settings,samples=self.samples),
                allow_nan=False,default=lambda value: value.tolist()),encoding='utf-8')
            os.replace(temporary,self.path)
        except (OSError,ValueError,TypeError,AttributeError):
            # Diagnostics must never stop the simulation or a result return.
            pass
