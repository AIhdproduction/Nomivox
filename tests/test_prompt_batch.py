from pathlib import Path

from e57cls.pipeline.config import resolve_config


def test_auto_prompt_batch_is_the_default_and_fixed_batches_remain_valid(tmp_path):
    source = tmp_path/'scan.e57'
    source.touch()
    automatic = resolve_config({'source': str(source)})
    fixed = resolve_config({'source': str(source), 'prompt_batch': 16})
    assert automatic['prompt_batch'] == 'auto'
    assert fixed['prompt_batch'] == 16
