"""Validate user-editable semantic classes before allocating GPU work."""
import re
from collections import defaultdict

STANDARD = set(range(1, 23)) - {8, 12}


def validate_preset(preset):
    errors, warnings = [], []
    ids, codes, prompts = set(), set(), defaultdict(list)
    if preset.get('schema_version') != 1 or preset.get('unclassified_code') != 1:
        errors.append('schema_or_unclassified_code')
    entries = preset.get('classes', [])
    if not isinstance(entries, list) or not entries:
        return ['no_classes'], []
    mapping = preset.get('code_map', {})
    pairs = {frozenset(p) for p in preset.get('scene_resolved_pairs', [])}
    for item in [*preset.get('special', []), *entries]:
        ident, code = item.get('id', ''), item.get('code')
        if not re.fullmatch('[a-z0-9_]+', ident) or ident in ids:
            errors.append(f'invalid_or_duplicate_id:{ident}')
        ids.add(ident)
        if not isinstance(code, int) or isinstance(code, bool) or not 1 <= code <= 255:
            errors.append(f'invalid_code:{ident}')
            continue
        try:
            exported = int(mapping.get(str(code), code))
        except (ValueError, TypeError):
            errors.append(f'invalid_code_map:{ident}')
            continue
        if exported in codes or not 1 <= exported <= 255:
            errors.append(f'duplicate_or_invalid_export_code:{ident}')
        codes.add(exported)
        if code < 64 and code not in STANDARD:
            warnings.append(f'reserved_las_code:{ident}:{code}')
        if not re.fullmatch('#[0-9a-fA-F]{6}', item.get('color', '')):
            errors.append(f'invalid_color:{ident}')
        if item in entries:
            if code in (1, 68):
                errors.append(f'special_code_in_class:{ident}')
            if item.get('scene') not in ('indoor', 'outdoor', 'both'):
                errors.append(f'invalid_scene:{ident}')
            texts = item.get('prompts', [])
            if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
                errors.append(f'empty_prompts:{ident}')
            elif item.get('enabled', True):
                for text in texts:
                    prompts[text.strip().casefold()].append(ident)
            for field in ('min_mask_conf', 'min_instance_conf'):
                value = item.get(field)
                if value is not None and (not isinstance(value, (float, int)) or not 0 <= value <= 1):
                    errors.append(f'invalid_threshold:{ident}')
    for text, owners in prompts.items():
        if len(set(owners)) > 1 and frozenset(owners) not in pairs:
            errors.append(f'prompt_conflict:{text}')
    special = {c.get('id'): c.get('code') for c in preset.get('special', [])}
    if special.get('unclassified') != 1 or special.get('reflection') != 68:
        errors.append('missing_special_classes')
    if mapping.get('1', 1) != 1 or mapping.get('68', 68) != 68:
        errors.append('special_codes_cannot_be_remapped')
    if not any(c.get('enabled', True) for c in entries):
        errors.append('no_active_classes')
    return errors, warnings
