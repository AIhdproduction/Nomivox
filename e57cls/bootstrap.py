"""Standard-library lock verification and environment repair for the BAT launcher."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
LOCK = PROJECT/'requirements.lock'


def matches():
    try:
        for line in LOCK.read_text(encoding='utf-8').splitlines():
            if not line or line.startswith(('#','--')):
                continue
            if ' @ git+' in line:
                name, url = line.split(' @ ',1)
                info = json.loads(metadata.distribution(name).read_text('direct_url.json'))
                if info.get('vcs_info',{}).get('commit_id') != url.rsplit('@',1)[1]:
                    return False
            else:
                name,version = line.split('==')
                if metadata.version(name) != version:
                    return False
        return True
    except (metadata.PackageNotFoundError,ValueError,TypeError,OSError):
        return False


def smoke():
    return subprocess.run([sys.executable,'-c',
        'import numpy,scipy,pye57,PIL.Image,laspy,lazrs,torch,torchvision; from PySide6.QtWidgets import QApplication; from sam3.model_builder import build_sam3_image_model'],
        cwd=PROJECT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0


def main():
    strings = json.loads((PROJECT/'e57cls/gui/strings_de.json').read_text(encoding='utf-8'))
    matched = matches()
    if '--verify' in sys.argv:
        return 0 if matched and smoke() else 1
    if not matched or not smoke():
        build = [line for line in LOCK.read_text().splitlines() if line.lower().startswith(('setuptools==','wheel=='))]
        subprocess.run([sys.executable,'-m','pip','install','--no-deps',*build],check=True)
        command = [sys.executable,'-m','pip','install','--no-build-isolation','-r',str(LOCK)]
        if matched:
            command.append('--force-reinstall')
        subprocess.run(command,check=True)
        if not matches() or not smoke():
            print(strings['setup_failed'],flush=True)
            return 1
    if '--install-only' in sys.argv:
        return 0
    import torch
    if not torch.cuda.is_available():
        print(strings['gpu_missing'],flush=True)
        return 1
    if not any((PROJECT/'models'/name).is_file() for name in ('sam.pt','sam3.pt')):
        print(strings['model_missing'],flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
