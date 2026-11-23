# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Wind and water on hair strands (GPU), in the solver's centimeter units.

Settings come from the same panels as the Drape Tool (wind_settings.pack_wind and
water_settings.pack_water, which use meters); kernels convert to centimeters.

Wind: hair behaves like thin strands, so air pushes on the part of the relative
wind that is across each strand segment (drag ~ |v_perp| v_perp), plus a little
along it. The beam/spread/falloff/gust rules and avatar shielding match the Drape
Tool's wind. Turbulence is hair flutter: small eddies carried downwind, out of
phase from strand to strand, and still present in the wake behind the avatar. Water: each point is buoyed and slowed by how far it
is below the waterline, with extra resistance moving down through the surface. Buoyancy
reduces the point's weight (and the style support follows that weight); drag is against
the moving water, so a wind setting doubles as a current.
"""
import numpy as np
import warp as wp

from water_settings import pack_water
from wind_settings import pack_wind

# Wind grip on hair (1/cm). A card stands for a clump of hairs that shelter each other, so
# this is well below the bare-strand value; 0.006 lets 6 m/s lift a ponytail in an arc while
# gravity still shapes it.
WIND_RESPONSE = 0.006
MAX_WIND_ACCEL = 8000.0       # cm/s^2, as the Drape Tool's 80 m/s^2 limit


@wp.func
def supported(g: wp.vec3, support: wp.vec4):
    """Gravity felt after the style support (hair_solver.HairSimulation.supported)."""
    size = wp.length(g)
    if size < 1.0e-6 or support[0] <= 0.0:
        return g
    facing = wp.max(wp.dot(g / size, wp.vec3(support[1], support[2], support[3])), 0.0)
    return g * (1.0 - support[0] * facing * facing * wp.min(size, 981.0) / size)


@wp.kernel
def external_accel(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                   neighbour: wp.array(dtype=int), owner: wp.array(dtype=int),
                   wind: wp.array(dtype=float), clock: wp.array(dtype=float),
                   colliders: wp.array(dtype=wp.uint64), water: wp.array(dtype=float),
                   gravity: wp.array(dtype=wp.vec3), dt: float, shelter: float, response: float,
                   support: wp.vec4, accel: wp.array(dtype=wp.vec3), wetness: wp.array(dtype=float)):
    i = wp.tid()
    a = wp.vec3(0.0)
    current = wp.vec3(0.0)
    wet = float(0.0)
    if water[0] > 0.5:
        level = water[1] * 100.0
        transition = water[5] * 100.0
        for k in range(4):
            if pos[i][1] < level - transition * (float(k) + 0.5) / 4.0:
                wet += 0.25
    wetness[i] = wet
    if inv_mass[i] > 0.0:
        p = pos[i]
        t = pos[neighbour[i]] - p
        tl = wp.length(t)
        if tl > 1.0e-9:
            t = t / tl
        # ---- wind (settings in metres; convert to cm) --------------------------------
        if wind[0] > 0.5 and wind[7] > 0.0:
            direction = wp.vec3(wind[4], wind[5], wind[6])
            source = wp.vec3(wind[1], wind[2], wind[3]) * 100.0
            delta = p - source
            axial = wp.dot(delta, direction)
            radial_offset = delta - direction * axial
            radial_distance = wp.length(radial_offset)
            axis = direction
            if wind[17] > 0.5:
                direction = axis * wind[18]
                if radial_distance > 1.0e-6:
                    direction += radial_offset * (wind[19] / radial_distance)
            factor = float(1.0)
            if (wind[8] > 0.5 or wind[10] > 0.5 or wind[19] > 1.0e-8) and axial < 0.0:
                factor = 0.0
            if wind[8] > 0.5:
                factor *= wp.clamp(1.0 - axial / (wind[9] * 100.0), 0.0, 1.0)
            if wind[10] > 0.5:
                outer = wind[11] * 100.0
                inner = outer * (1.0 - wind[12])
                edge = wp.clamp((outer - radial_distance) / wp.max(outer - inner, 1.0e-4), 0.0, 1.0)
                factor *= edge * edge * (3.0 - 2.0 * edge)
            if factor > 0.0:
                # Shelter calms the mean wind behind the avatar, but its wake stays turbulent.
                sheltered = float(1.0)
                if wind[13] > 0.5:
                    distance = float(1.0e6)
                    if wind[8] > 0.5 or wind[10] > 0.5:
                        distance = wp.max(axial - 0.1, 0.0)
                    ray = axis
                    if wp.length(direction) > 1.0e-8:
                        ray = wp.normalize(direction)
                    for m in range(colliders.shape[0]):
                        hit = wp.mesh_query_ray(colliders[m], p - ray * 0.1, -ray, distance)
                        if hit.result:
                            sheltered = shelter
                            break
                time = clock[0]
                speed = wind[7] * 100.0
                gust = wp.max(0.0, 1.0 + wind[14] * (0.65 * wp.sin(6.2831853 * wind[15] * time)
                                                     + 0.35 * wp.sin(3.883222 * wind[15] * time)))
                # Each strand catches a slightly different share of the wind, so hair never moves as one sheet.
                strand = float(owner[i])
                hash_a = wp.abs(wp.sin(strand * 12.9898) * 43758.5453)
                hash_a = hash_a - wp.floor(hash_a)
                hash_b = wp.abs(wp.sin(strand * 78.233) * 12345.6789)
                hash_b = hash_b - wp.floor(hash_b)
                air = direction * (speed * gust * (0.8 + 0.4 * hash_b) * sheltered)
                # Flutter: small air eddies carried downwind (about 30 cm apart, ~12 Hz at 6 m/s),
                # out of phase from strand to strand, across and along the flow.
                d = wp.normalize(axis)
                up = wp.vec3(0.0, 1.0, 0.0)
                if wp.abs(d[1]) > 0.9:
                    up = wp.vec3(1.0, 0.0, 0.0)
                e1 = wp.normalize(wp.cross(d, up))
                e2 = wp.cross(d, e1)
                phase = hash_a * 6.2831853
                wave = 6.2831853 / 30.0
                omega = wave * speed * 0.6
                x = wp.dot(p, d)
                flutter = e1 * wp.sin(wave * x - omega * time + phase)                     + e2 * wp.sin(1.3 * wave * x - 1.1 * omega * time + 2.1 * phase)                     + d * (0.5 * wp.sin(0.7 * wave * x - 0.8 * omega * time + 3.7 * phase))
                air += flutter * (speed * wind[16] * wp.max(sheltered, 0.8))
                air = air * factor
                current = air
                relative = air - vel[i]
                across = relative - t * wp.dot(relative, t)
                force = across * (wp.length(across) * response) + relative * (wp.length(relative) * response * 0.02)
                # Under water the wind stands for a current: water drag carries the hair instead.
                a = a + force * (1.0 - wet)
            magnitude = wp.length(a)
            if magnitude > MAX_WIND_ACCEL:
                a = a * (MAX_WIND_ACCEL / magnitude)
        # ---- water (settings in metres; convert to cm) -------------------------------
        if water[0] > 0.5:
            level = water[1] * 100.0
            transition = water[5] * 100.0
            if wet > 0.0:
                # Drag against the moving water (a wind setting doubles as the current).
                v = vel[i] - current
                c = water[3] * wet
                vertical = c
                if v[1] < 0.0:
                    depth = wp.max(level - p[1], 0.0)
                    surface = wp.clamp(1.0 - depth / (transition * 2.0), 0.0, 1.0)
                    vertical += water[4] * wet * surface
                drag = wp.vec3(-v[0] * c / (1.0 + c * dt), -v[1] * vertical / (1.0 + vertical * dt),
                               -v[2] * c / (1.0 + c * dt))
                # Buoyancy is gravity in reverse: the point's weight becomes g (1 - buoyancy x wet),
                # and the style support applies to that weight (the solver already adds the dry,
                # supported gravity, so only the difference goes here).
                g = gravity[0]
                a = a + drag + supported(g * (1.0 - water[2] * wet), support) - supported(g, support)
    accel[i] = a



class HairForces:
    """Live wind/water state on the GPU; ``update`` copies control snapshots between frames."""

    def __init__(self, particle_start, particle_count, colliders, shelter=0.1, response=WIND_RESPONSE):
        self.shelter = float(shelter)
        self.response = float(response)
        start, count = np.asarray(particle_start), np.asarray(particle_count)
        total = int(count.sum())
        index = np.arange(total)
        last = np.repeat(start + count - 1, count)
        neighbour = np.where(index < last, index + 1, index - 1)     # tip uses its previous point
        self.neighbour = wp.array(neighbour.astype(np.int32), dtype=int)
        self.owner = wp.array(np.repeat(np.arange(len(count)), count).astype(np.int32), dtype=int)
        self.wind = wp.zeros(20, dtype=float)
        self.water = wp.zeros(6, dtype=float)
        self.clock = wp.zeros(1, dtype=float)
        self.gravity = wp.zeros(1, dtype=wp.vec3)
        self.accel = wp.zeros(total, dtype=wp.vec3)
        self.wetness = wp.zeros(total, dtype=float)
        self.support = wp.vec4(0.0, 0.0, -1.0, 0.0)
        self.colliders = wp.array(np.asarray([m.id for m in colliders], np.uint64), dtype=wp.uint64)
        self.revisions = [None, None]
        self.active = False

    def update(self, wind_control, water_control, gravity_cm, time):
        active = False
        if wind_control is not None:
            revision, settings = wind_control.snapshot()
            if revision != self.revisions[0]:
                self.wind.assign(pack_wind(settings)); self.revisions[0] = revision
            active |= bool(settings.get("enabled"))
        if water_control is not None:
            revision, settings = water_control.snapshot()
            if revision != self.revisions[1]:
                self.water.assign(pack_water(settings)); self.revisions[1] = revision
            active |= bool(settings.get("enabled"))
        self.gravity.assign(np.asarray(gravity_cm, np.float32).reshape(1, 3))
        self.clock.assign(np.array([time], np.float32))
        self.active = active

    def compute(self, pos, vel, inv_mass, dt):
        wp.launch(external_accel, dim=len(self.accel), inputs=[pos, vel, inv_mass, self.neighbour, self.owner, self.wind,
                  self.clock, self.colliders, self.water, self.gravity, dt, self.shelter, self.response,
                  self.support, self.accel, self.wetness])
