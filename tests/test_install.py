import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from e57cls import bootstrap


def test_install_only_checks_packages_without_requiring_gpu_or_model(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['bootstrap', '--install-only'])
    monkeypatch.setattr(bootstrap, 'matches', lambda: True)
    monkeypatch.setattr(bootstrap, 'smoke', lambda: True)
    # Accessing torch.cuda would fail; dependency installation must finish anyway.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    assert bootstrap.main() == 0


@pytest.mark.parametrize('repair_succeeds', [False, True])
def test_install_only_does_not_hide_dependency_failures(monkeypatch, repair_succeeds):
    monkeypatch.setattr(sys, 'argv', ['bootstrap', '--install-only'])
    checks = iter([False, repair_succeeds])
    monkeypatch.setattr(bootstrap, 'matches', lambda: next(checks))
    monkeypatch.setattr(bootstrap, 'smoke', lambda: repair_succeeds)
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    assert bootstrap.main() == (0 if repair_succeeds else 1)
    assert len(calls) == 2
    assert '-r' in calls[-1]


@pytest.mark.skipif(os.name != 'nt', reason='Windows BAT/PowerShell installer')
@pytest.mark.parametrize('empty_preferred_model', [False, True])
def test_installer_reports_model_even_without_local_environment(tmp_path, empty_preferred_model):
    project = tmp_path / 'Project with spaces'
    (project / 'scripts').mkdir(parents=True)
    shutil.copy2(bootstrap.PROJECT / 'install.bat', project)
    shutil.copy2(bootstrap.PROJECT / 'scripts/install.ps1', project / 'scripts')
    if empty_preferred_model:
        (project / 'models').mkdir()
        (project / 'models/sam.pt').touch()
        (project / 'models/sam3.pt').write_bytes(b'fallback must not hide empty preferred file')
    result = subprocess.run(
        ['cmd.exe', '/d', '/c', 'install.bat', '--check', '--no-pause'],
        cwd=project, capture_output=True, timeout=60,
    )
    output = result.stdout.decode(errors='replace')
    assert result.returncode == 1, output + result.stderr.decode(errors='replace')
    assert 'ABSCHLUSS' in output
    assert '[OFFEN] Lokale Python' in output
    assert '[FEHLT] SAM-3-Modell' in output
    assert 'https://huggingface.co/facebook/sam3' in output
    assert 'Files and versions' in output
    assert 'NOCH NICHT STARTBEREIT' in output
    assert not (project / '.cache/venv/Scripts/python.exe').exists()
    assert (project / '.cache/install.log').is_file()
