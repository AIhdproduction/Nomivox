"""Deterministic, georeferenced shape-rule benchmark fixtures.

The distributions mirror the HSLU post-processing profile: irregular indoor
rooms with noisy structural surfaces and mixed object sizes, plus clustered
outdoor objects, 28 large-crown trees and sloped terrain.  Coordinates are
kept in Swiss georeferenced space and rotated 23 degrees to prevent accidental
assumptions about an origin-centred, axis-aligned scene.
"""

from __future__ import annotations

import json

import numpy as np

from e57cls.vote.engine import VoteStore, voxel_keys


SHIFT = np.asarray((2665953.0, 1207269.0, 440.0))
ANGLE = np.deg2rad(23.0)
ROTATION = np.asarray(((np.cos(ANGLE), -np.sin(ANGLE), 0.),
                       (np.sin(ANGLE), np.cos(ANGLE), 0.), (0., 0., 1.)))


def _world(local):
    return np.asarray(local, np.float64) @ ROTATION.T+SHIFT


def _planes(path, by_id, count):
    cells = np.asarray([(x, z) for x in range(41) for z in range(61)], np.int32)
    normal = ROTATION @ np.asarray((0., 1., 0.))
    basis = np.asarray((ROTATION @ np.asarray((1., 0., 0.)),
                        ROTATION @ np.asarray((0., 0., 1.))))
    rows, data, columns = [], {}, int(np.ceil(np.sqrt(count)))
    for index in range(count):
        center = _world(((index % columns)*2.7, (index // columns)*3.1, (index % 3)*3.))
        span = np.abs(basis[0])*1.2+np.abs(basis[1])*1.7+np.abs(normal)*.12
        key = f'wall_{index}'
        rows.append({'class_id': 'wall', 'normal': normal.tolist(), 'center': center.tolist(),
                     'basis': basis.tolist(), 'area_m2': 6., 'cells': key,
                     'bbox_min': (center-span).tolist(), 'bbox_max': (center+span).tolist()})
        data[key] = cells
    np.savez(path, metadata=json.dumps({'cell_size': .05, 'planes': rows}), **data)
    return path


def _weights(winners, codes):
    values = np.zeros((len(winners), len(codes)), np.float32)
    slots = {code: index for index, code in enumerate(codes)}
    values[np.arange(len(winners)), [slots[int(code)] for code in winners]] = 1.
    return values


def _object_cloud(rng, centers, sizes, samples=6):
    """Return irregular surface samples, grouped enough for 4-cm components."""
    owner = np.repeat(np.arange(len(centers)), samples)
    offsets = rng.uniform(-.5, .5, (len(owner), 3))*sizes[owner]
    # Preserve a denser vertical envelope instead of the old single-point grid.
    offsets[:, 2] = np.where(np.arange(len(owner)) % 2, offsets[:, 2], np.sign(offsets[:, 2])*sizes[owner, 2]/2)
    return centers[owner]+offsets+rng.normal(0., .006, offsets.shape), owner


def _indoor(rng, by_id, scale):
    objects = max(120, int(round(45_368*scale)))
    rooms = rng.integers(0, 44, objects)
    room_x, room_y = rooms % 11, rooms // 11
    centers = np.column_stack((room_x*7.+rng.uniform(.5, 6.5, objects),
                               room_y*8.+rng.uniform(.5, 7.5, objects),
                               rng.integers(0, 4, objects)*3.5+rng.uniform(.1, 2.8, objects)))
    sizes = rng.uniform((.08, .08, .06), (1.2, .8, 1.8), (objects, 3))
    points, owner = _object_cloud(rng, centers, sizes)
    mix = np.asarray((by_id['furniture'], by_id['construction_material'], by_id['pipe'],
                      by_id['ceiling_equipment'], by_id['hvac'], by_id['wall']))
    codes = mix[owner % len(mix)]
    wall_count = max(8_000, int(round(520_000*scale)))
    x = rng.uniform(0., 77., wall_count)
    y = rng.integers(0, 45, wall_count)*.18+rng.normal(0., .015, wall_count)
    z = rng.uniform(0., 13.8, wall_count)
    keep = (np.rint(x/.02).astype(np.int64)+np.rint(z/.02).astype(np.int64)) % 11 != 0
    wall = np.column_stack((x[keep], y[keep], z[keep]))
    floor_count = max(5_000, int(round(260_000*scale)))
    floor = np.column_stack((rng.uniform(0., 77., floor_count), rng.uniform(0., 32., floor_count),
                             rng.choice((0., 3.5, 7., 10.5), floor_count)))
    return np.vstack((points, wall, floor, floor+np.asarray((0., 0., 3.))),), np.r_[codes,
        np.full(len(wall), by_id['wall']), np.full(len(floor), by_id['floor']), np.full(len(floor), by_id['ceiling'])]


def _outdoor(rng, by_id, scale):
    objects, stems = max(160, int(round(50_000*scale))), max(4, int(round(28*scale)))
    cluster = rng.integers(0, 3, objects)
    centers = np.column_stack((rng.uniform(-35., 125., objects)+cluster*18.,
                               rng.uniform(-25., 80., objects)+(cluster == 2)*24.,
                               rng.uniform(.05, 2.5, objects)))
    centers[:, 2] += .025*centers[:, 0]+.012*centers[:, 1]
    points, owner = _object_cloud(rng, centers, rng.uniform((.08, .05, .08), (3., .5, 2.5), (objects, 3)), 5)
    mix = np.asarray((by_id['vehicle'], by_id['fence'], by_id['traffic_sign'], by_id['outdoor_wall'], 1))
    sx, sy = rng.uniform(5., 110., stems), rng.uniform(5., 70., stems)
    levels = np.arange(0., 3.24, .04)
    trunk = np.column_stack((np.repeat(sx, len(levels)), np.repeat(sy, len(levels)),
                             np.repeat(.025*sx+.012*sy, len(levels))+np.tile(levels, stems)))
    angle = np.linspace(0., 2*np.pi, 32, endpoint=False)
    crown_radius = np.repeat(rng.uniform(3., 7., stems), len(angle))
    crown = np.column_stack((np.repeat(sx, len(angle))+np.tile(np.cos(angle), stems)*crown_radius,
                             np.repeat(sy, len(angle))+np.tile(np.sin(angle), stems)*crown_radius,
                             np.repeat(.025*sx+.012*sy+4.2, len(angle))))
    road_x = np.linspace(-45., 150., max(800, int(8_000*scale)))
    road = np.column_stack((road_x, np.full(len(road_x), -3.), .025*road_x-.036))
    return np.vstack((points, trunk, crown, road, road+np.asarray((0., .45, .12)))), np.r_[mix[owner % len(mix)],
        np.full(len(trunk), by_id['tree_trunk']), np.full(len(crown), by_id['tree']),
        np.full(len(road), by_id['road']), np.full(len(road), by_id['sidewalk'])]


def build_hslu_shape_fixture(tmp_path, part, scale, by_id, codes, name='hslu'):
    """Create a file-SQLite input for a deterministic slow shape-rule benchmark."""
    rng = np.random.default_rng(16016 if part == 'indoor' else 16017)
    local, winners = _indoor(rng, by_id, scale) if part == 'indoor' else _outdoor(rng, by_id, scale)
    world = _world(local)
    origin = np.floor(world.min(axis=0)/.02)*.02-.04
    keys = voxel_keys(world, origin, .02)
    order = np.argsort(keys, kind='stable')
    selected = order[np.r_[True, keys[order][1:] != keys[order][:-1]]]
    store = VoteStore(tmp_path/f'{name}_{part}.sqlite')
    store.add_scan(1, 'synthetic-1', [(keys[selected], _weights(winners[selected], codes))], codes)
    planes = _planes(tmp_path/f'{name}_{part}_planes.npz', by_id, max(3, int(round(1300*scale))))
    scene = 'indoor' if part == 'indoor' else 'outdoor'
    context = {1: {'scene': scene, 'outdoor': part == 'outdoor',
                   'position': _world((3., -3., 1.6)).tolist(), 'allowed_codes': set(codes)}}
    return store, origin, context, planes
