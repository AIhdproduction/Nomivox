"""Pure scene-resolution helpers shared by static pipeline stages."""
import re

import numpy as np


def scan_display_name(metadata):
    value = metadata.get('rlms:scanposGuid') or metadata.get('name') or ''
    return str(value).rstrip('/').rsplit('/', 1)[-1]


def name_hint(metadata, config):
    if not config.get('scene_name_hints', True) or config.get('scene_name_hint_mode') == 'off':
        return None
    outdoor = [re.compile(pattern, re.IGNORECASE) for pattern in config['scene_name_outdoor_patterns']]
    indoor = [re.compile(pattern, re.IGNORECASE) for pattern in config['scene_name_indoor_patterns']]
    for field in config['scene_name_fields']:
        value = metadata.get(field)
        if not isinstance(value, str):
            continue
        for segment in reversed(value.split('/')):
            folded = segment.casefold()
            outside = next((pattern.pattern for pattern in outdoor if pattern.search(folded)), None)
            inside = next((pattern.pattern for pattern in indoor if pattern.search(folded)), None)
            if outside or inside:
                return {'scene': None if outside and inside else ('outdoor' if outside else 'indoor'),
                        'field': field, 'segment': segment, 'match': {'outdoor': outside, 'indoor': inside},
                        'ambiguous': bool(outside and inside)}
    return None


def _data_scene(detection):
    value = (detection or {}).get('scene')
    return 'outdoor' if value == 'covered_outdoor' else value if value in ('indoor', 'outdoor') else None


def resolve_scan_scene(detection, metadata, config, manual_mode=None):
    hint = name_hint(metadata or {}, config)
    data_scene = _data_scene(detection)
    detail = (detection or {}).get('scene', 'unsure')
    metrics = (detection or {}).get('metrics', {})
    thresholds = (detection or {}).get('thresholds', {})
    conflict = bool(manual_mode not in ('indoor', 'outdoor') and hint and hint['scene'] and data_scene
                    and hint['scene'] != data_scene)
    if manual_mode in ('indoor', 'outdoor'):
        scene, source, detail = manual_mode, 'manual', manual_mode
    elif hint and hint['scene'] and config.get('scene_name_hint_mode', 'prefer') == 'prefer':
        scene = hint['scene']
        source = 'name+data' if data_scene == scene else 'name'
    elif data_scene:
        scene = data_scene
        source = 'openness' if 'horizontal_openness' in metrics else 'data'
    elif hint and hint['scene'] and config.get('scene_name_hint_mode', 'prefer') == 'tiebreak':
        scene, source = hint['scene'], 'name'
    else:
        scene, source, detail = 'unsure', 'pending', detail if detail in ('mixed', 'unsure') else 'unsure'
    if detail == 'covered_outdoor':
        scene = 'outdoor'
    class_mode = scene if scene in ('indoor', 'outdoor') else ('indoor' if config.get('scene_unresolved_classes') == 'indoor' else 'both')
    return {'scene': scene, 'scene_detail': detail, 'data_scene': data_scene, 'name_hint': hint,
            'source': source, 'conflict': conflict, 'class_mode': class_mode,
            'metrics': metrics, 'thresholds': thresholds}


def resolve_neighbours(records, positions, config):
    records = [dict(record) for record in records]
    ordered = sorted(enumerate(records), key=lambda item: (item[1].get('scan', item[0]), item[0]))
    resolved = [(index, record) for index, record in ordered if record.get('scene') in ('indoor', 'outdoor')]
    warnings = []
    if not config.get('scene_neighbour_consensus', True):
        return records, warnings
    radius, height = config['scene_neighbour_radius'], config['scene_neighbour_height']
    for index, record in ordered:
        if record.get('scene') != 'unsure' and record.get('source') != 'pending':
            continue
        position = positions.get(record.get('scan')) if isinstance(positions, dict) else positions[index]
        if position is None:
            warnings.append({'code': 'scene_unresolved', 'scan': record.get('scan')})
            continue
        candidates = []
        for other_index, other in resolved:
            other_position = positions.get(other.get('scan')) if isinstance(positions, dict) else positions[other_index]
            if other_position is None:
                continue
            delta = np.asarray(position, float) - np.asarray(other_position, float)
            horizontal = float(np.hypot(delta[0], delta[1]))
            if horizontal <= radius and abs(delta[2]) <= height:
                candidates.append((horizontal, other.get('scan', other_index), other))
        if len(candidates) < config['scene_neighbour_min']:
            warnings.append({'code': 'scene_unresolved', 'scan': record.get('scan')})
            continue
        totals = {'indoor': 0., 'outdoor': 0.}
        for distance, _, other in candidates:
            totals[other['scene']] += 1/(distance+1)
        best_total = max(totals.values())
        tied = [scene for scene, total in totals.items() if np.isclose(total, best_total)]
        if len(tied) == 1:
            best = tied[0]
        else:
            best = min((distance, scan, other['scene']) for distance, scan, other in candidates
                       if other['scene'] in tied)[2]
        record.update(scene=best, scene_detail=best, source='neighbour', class_mode=best)
        warnings.append({'code': 'scene_neighbour_resolved', 'scan': record.get('scan')})
    for index, record in ordered:
        if record.get('scene') not in ('indoor', 'outdoor'):
            continue
        position = positions.get(record.get('scan')) if isinstance(positions, dict) else positions[index]
        if position is None:
            continue
        totals = {'indoor': 0., 'outdoor': 0.}
        for other_index, other in resolved:
            if other_index == index:
                continue
            other_position = positions.get(other.get('scan')) if isinstance(positions, dict) else positions[other_index]
            if other_position is None:
                continue
            delta = np.asarray(position, float) - np.asarray(other_position, float)
            distance = float(np.hypot(delta[0], delta[1]))
            if distance <= radius and abs(delta[2]) <= height:
                totals[other['scene']] += 1/(distance+1)
        total = sum(totals.values())
        if total and totals[record['scene']] / total <= 1/3:
            warnings.append({'code': 'scene_neighbour_disagreement', 'scan': record.get('scan')})
    return records, warnings
