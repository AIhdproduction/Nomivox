import re
from pathlib import Path

from e57cls.pipeline.config import resolve_config


def test_auto_prompt_batch_is_the_default_and_fixed_batches_remain_valid(tmp_path):
    source = tmp_path/'scan.e57'
    source.touch()
    automatic = resolve_config({'source': str(source)})
    fixed = resolve_config({'source': str(source), 'prompt_batch': 16})
    assert automatic['prompt_batch'] == 'auto'
    assert fixed['prompt_batch'] == 16


def test_default_output_uses_sanitized_scan_name_and_timestamp(tmp_path):
    source = tmp_path/'HSLU_Trakt_4.e57'
    source.touch()

    output = Path(resolve_config({'source': str(source)})['output'])

    assert output.parent == tmp_path/'klassifiziert'
    assert re.fullmatch(r'HSLU_Trakt_4_\d{8}_\d{6}', output.name)


def test_default_output_sanitizes_spaces_and_special_characters(tmp_path):
    source = tmp_path/'Mein Scan (Aussen)#1.e57'
    source.touch()

    output = Path(resolve_config({'source': str(source)})['output'])

    assert re.fullmatch(r'Mein_Scan_Aussen_1_\d{8}_\d{6}', output.name)


def test_explicit_output_remains_unchanged(tmp_path):
    source = tmp_path/'scan.e57'
    source.touch()
    target = tmp_path/'custom-output'

    config = resolve_config({'source': str(source), 'output': str(target)})

    assert Path(config['output']) == target.resolve()
