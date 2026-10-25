"""Last, vote-aware correction of classes that cannot occur in a scene."""

from __future__ import annotations

import time

import numpy as np

from e57cls.vote.scene_context import ensure_scene_context


DEFAULT_FALLBACKS = {
    'floor': ['ground', 'sidewalk', 'road'], 'ceiling': ['building'], 'wall': ['building'],
    'door': ['building'], 'railing': ['fence'], 'ground': ['floor'], 'road': ['floor'],
    'sidewalk': ['floor'], 'curb': ['floor'], 'manhole': ['floor'], 'building': ['wall'],
    'fence': ['railing'], 'outdoor_wall': ['wall'],
}


def _scene_codes(preset_classes, by_id, active_codes):
    """Build code-to-scene sets; aliases remain valid if multiple names share a code."""
    scenes = {}
    names = {}
    for item in preset_classes:
        name = item.get('id')
        if not name or name not in by_id:
            continue
        code = int(by_id[name])
        if code not in active_codes or not item.get('enabled', True):
            continue
        scene = item.get('scene', 'both')
        scenes.setdefault(code, set()).add(scene if scene in ('indoor', 'outdoor') else 'both')
        names[name] = code
    return scenes, names


def _fits(scenes, code, scene):
    values = scenes.get(int(code), set())
    return 'both' in values or ('outdoor' if scene == 1 else 'indoor') in values


def enforce_scene_consistency(votes, preset_classes, by_id, scan_context, origin, voxel_size, config,
                              active_codes, checkpoint=lambda: None, outline=None):
    """Replace only certain indoor/outdoor conflicts, retaining vote confidence where possible."""
    started = time.monotonic()
    result = {'enabled': bool(config.get('scene_consistency', True)),
              'context_voxels': {'outdoor': 0, 'indoor': 0, 'ambiguous': 0, 'unknown': 0},
              'changed': {}, 'unclassified': 0, 'duration_s': 0.}
    if not result['enabled']:
        return result
    context_report = ensure_scene_context(votes, by_id, scan_context, origin, voxel_size, config, checkpoint,
                                          outline=outline)
    result['context_voxels'] = context_report['context_voxels']
    result['scene_outline'] = context_report.get('scene_outline', {})
    active_codes = {int(code) for code in active_codes}
    scenes, names = _scene_codes(preset_classes, by_id, active_codes)
    indoor_only = sorted(code for code, values in scenes.items() if values == {'indoor'})
    outdoor_only = sorted(code for code, values in scenes.items() if values == {'outdoor'})
    db = votes.db
    db.executescript('DROP TABLE IF EXISTS temp.scene_fix;\n'
                     'CREATE TEMP TABLE scene_fix(key BLOB PRIMARY KEY,code INTEGER,confidence INTEGER,scene_context INTEGER) WITHOUT ROWID;')
    if indoor_only or outdoor_only:
        clauses, params = [], []
        if indoor_only:
            clauses.append('(c.scene_context=1 AND d.code IN ('+','.join('?' for _ in indoor_only)+'))')
            params.extend(indoor_only)
        if outdoor_only:
            clauses.append('(c.scene_context=0 AND d.code IN ('+','.join('?' for _ in outdoor_only)+'))')
            params.extend(outdoor_only)
        db.execute('''INSERT INTO scene_fix SELECT d.key,d.code,d.confidence,c.scene_context FROM decisions d
                      JOIN geometry_context c USING(key) WHERE '''+' OR '.join(clauses), params)

    configured = {**DEFAULT_FALLBACKS, **config.get('scene_consistency_fallbacks', {})}
    name_by_code = {}
    for name, code in names.items():
        name_by_code.setdefault(code, name)

    def resolve(groups):
        """Resolve complete SQL groups; all numerical vote work stays per chunk."""
        nonlocal changes
        if not groups:
            return
        keys = np.asarray([row[0] for row in groups], dtype='V24')
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
        for start, stop in zip(starts, np.r_[starts[1:], len(groups)]):
            group = groups[start:stop]
            key, current, confidence, scene = group[0][:4]
            candidates = [names[name] for name in configured.get(name_by_code.get(current, ''), ())
                          if name in names and _fits(scenes, names[name], scene)]
            vote_codes = np.asarray([row[4] for row in group if row[4] is not None], np.int64)
            vote_weights = np.asarray([row[5] for row in group if row[4] is not None], np.float64)
            weights = dict(zip(vote_codes.tolist(), vote_weights.tolist()))
            target = None
            new_confidence = int(confidence)
            if candidates:
                voted = [(weights[code], -index, code) for index, code in enumerate(candidates) if code in weights]
                target = max(voted)[2] if voted else candidates[0]
            if target is None:
                suitable = [(weight, -code, code) for code, weight in weights.items()
                            if code in active_codes and _fits(scenes, code, scene)]
                if suitable:
                    target = max(suitable)[2]
                    total = float(vote_weights.sum())
                    new_confidence = int(round(255.*weights[target]/total)) if total else 0
            if target is None:
                target, new_confidence = 1, 0
                result['unclassified'] += 1
            if target != current:
                changes[bytes(key)] = (int(target), int(new_confidence), int(current))

    # The target scene changes per row, so retain ordered names and filter per group.
    cursor = db.execute('''SELECT f.key,f.code,f.confidence,f.scene_context,v.class,v.weight
                         FROM scene_fix f LEFT JOIN votes v USING(key) ORDER BY f.key,v.class''')
    changes = {}
    carry = []
    while rows := cursor.fetchmany(100_000):
        checkpoint()
        rows = carry+rows
        split = len(rows)
        while split and rows[split-1][0] == rows[-1][0]:
            split -= 1
        carry = rows[split:]
        resolve(rows[:split])
    resolve(carry)
    updates = [(target, confidence, key) for key, (target, confidence, _) in changes.items()]
    for start in range(0, len(updates), 50_000):
        checkpoint()
        db.executemany('UPDATE decisions SET code=?,confidence=? WHERE key=?', updates[start:start+50_000])
    db.commit()
    if updates:
        votes._invalidate_lookup()
    for target, _, source in changes.values():
        label = f'{source}->{target}'
        result['changed'][label] = result['changed'].get(label, 0)+1
    result['duration_s'] = time.monotonic()-started
    return result
