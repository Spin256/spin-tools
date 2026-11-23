# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Purpose-based collision rules, independent of topology and animation."""

ROLES = ('avatar','avatar_attachment','collision','fixed','moving')


def primary_index(colliders, head_collider=None):
    avatars = [i for i,c in enumerate(colliders) if c.get('role') == 'avatar']
    if len(avatars)>1:
        raise ValueError('Assign one avatar for the hair roots to follow.')
    if avatars:
        return avatars[0]
    if any(c.get('role') in ('avatar','avatar_attachment','collision') for c in colliders):
        raise ValueError('Assign one avatar for the hair roots to follow.')
    # Colliders without those roles: the given head collider, else the first moving one, else the first.
    if head_collider is not None:
        return head_collider
    return next((i for i,c in enumerate(colliders) if c.get('role') == 'moving'),
                0 if colliders else None)


def resolve_modes(colliders, head_collider=None):
    avatar = primary_index(colliders,head_collider)
    return ['solid' if i==avatar else 'surface' for i in range(len(colliders))]
