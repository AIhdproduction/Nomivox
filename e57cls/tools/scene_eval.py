"""Evaluate saved scene decisions without rerunning segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e57cls.pipeline.config import DEFAULTS
from e57cls.scene.detect import decide
from e57cls.scene.resolve import name_hint


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hint(meta, scan, config):
    detected = name_hint(meta, config)
    if detected and detected.get('scene'):
        return detected['scene']
    resolution = scan.get('scene_resolution') or scan.get('scene_detection') or {}
    hint = resolution.get('name_hint')
    return hint.get('scene') if isinstance(hint, dict) else None


def _normalized_scene(value):
    return 'outdoor' if value == 'covered_outdoor' else value


def evaluate(directory, grid=False):
    directory = Path(directory)
    report, inspect = _load(directory/'report.json'), _load(directory/'inspect_report.json')
    config = {**DEFAULTS, **(_load(directory/'run_config.json') if (directory/'run_config.json').is_file() else {})}
    metadata = {row.get('index', index+1) if row.get('index') is not None else index+1: row.get('metadata', {})
                for index, row in enumerate(inspect.get('scans', []))}
    rows, correct = [], 0
    data_correct = data_total = neighbour_correct = neighbour_total = 0
    for scan in report.get('scans', []):
        truth = _hint(metadata.get(scan.get('scan'), {}), scan, config)
        if truth is None:
            continue
        resolution = scan.get('scene_resolution') or scan.get('scene_detection') or {}
        final = scan.get('scene')
        data = resolution.get('data_scene')
        if data is None:
            detection = scan.get('scene_detection') or resolution
            data = detection.get('scene')
        data_normalized = _normalized_scene(data)
        correct += final == truth
        if data_normalized in ('indoor', 'outdoor'):
            data_total += 1
            data_correct += data_normalized == truth
        if resolution.get('source') == 'neighbour':
            neighbour_total += 1
            neighbour_correct += final == truth
        metrics = resolution.get('metrics', {})
        rows.append({'scan': scan.get('scan'), 'truth': truth, 'final': final, 'data': data,
                     'data_normalized': data_normalized,
                     'source': resolution.get('source'),
                     'openness': metrics.get('horizontal_openness'), 'sky': metrics.get('sky_fraction'),
                     'ceiling': metrics.get('ceiling_fraction'), 'returns': metrics.get('upper_return_fraction')})
    result = {'final_accuracy': correct/max(1, len(rows)), 'final_count': len(rows),
              'data_accuracy': data_correct/max(1, data_total), 'data_count': data_total,
              'neighbour_accuracy': neighbour_correct/max(1, neighbour_total), 'neighbour_count': neighbour_total,
              'confusion': rows, 'scene_audit': report.get('scene_audit', {})}
    if grid:
        values = (.1, .2, .3, .4, .5, .6, .7, .8)
        result['grid'] = []
        for minimum in values:
            local = {**config, 'scene_openness_outdoor_min': minimum, 'scene_openness_indoor_max': max(0., minimum-.2)}
            evaluated = []
            for scan in report.get('scans', []):
                metrics = (scan.get('scene_resolution') or scan.get('scene_detection') or {}).get('metrics', {})
                if not {'sky_fraction', 'ceiling_fraction', 'upper_coverage', 'overhead_plane'}.issubset(metrics):
                    continue
                evaluated.append((decide(metrics, local), _hint(metadata.get(scan.get('scan'), {}), scan, config)))
            known = [(value, truth) for value, truth in evaluated if truth]
            result['grid'].append({'outdoor_min': minimum, 'indoor_max': local['scene_openness_indoor_max'],
                                   'accuracy': sum(('outdoor' if value == 'covered_outdoor' else value) == truth for value, truth in known)/max(1, len(known)),
                                   'count': len(known)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--grid', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.run_dir, args.grid), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
