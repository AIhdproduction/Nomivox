import json

from e57cls.pipeline.config import resolve_config
from e57cls.scene.presets import validate_preset


def test_reflection_only_mode_requires_mirror_analysis(tmp_path):
    source = tmp_path / 'scan.e57'
    source.touch()
    config = resolve_config({
        'source': str(source),
        'processing_mode': 'filter_only',
        'mirror_analysis': False,
    })
    assert config['processing_mode'] == 'filter_only'
    assert config['mirror_analysis'] is True


def test_reflection_only_preset_may_have_no_enabled_semantic_class():
    preset = json.loads(open('presets/default.json', encoding='utf-8').read())
    for entry in preset['classes']:
        entry['enabled'] = False
    errors, _ = validate_preset(preset, require_active_classes=False)
    assert 'no_active_classes' not in errors
