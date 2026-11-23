# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Artistic starting points; paint different softness on different regions."""
PRESETS={
    'Soft tissue — belly / buttock':dict(softness=75.,depth=25.,spread=45.,recovery=5.,clearance=.2),
    'Very soft tissue — breast':dict(softness=95.,depth=35.,spread=50.,recovery=3.,clearance=.2),
    'Muscle — arm / thigh':dict(softness=45.,depth=15.,spread=40.,recovery=8.,clearance=.2),
    'Firm — head / shoulder / foot':dict(softness=15.,depth=5.,spread=35.,recovery=15.,clearance=.2),
}

JIGGLE_PRESETS = {
    'Silicone — quick motion': (2.4, 7., 80., False),
    'Soft — slow motion': (1., 7., 140., False),
    'Soft — lag through turnaround': (1., 7., 80., True),
}
