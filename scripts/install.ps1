param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$env:TEMP = Join-Path $projectRoot '.cache\temp'
$env:TMP = $env:TEMP
$venvPython = Join-Path $projectRoot '.cache\venv\Scripts\python.exe'
$pythonReady = $false
$packagesReady = $false
$cudaReady = $false
$gitReady = $false
$transcriptStarted = $false
$logPath = Join-Path $projectRoot '.cache\install.log'

function Test-Python312([string]$Command, [string[]]$Prefix = @()) {
    try {
        & $Command @Prefix -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)' *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

Write-Host ''
Write-Host '=== PointSense - Installation ==='
Write-Host 'Richtet die Python-Umgebung und alle Projektpakete lokal ein.'
Write-Host 'Der erste Download kann mehrere GB umfassen und einige Zeit dauern.'
if ($CheckOnly) { Write-Host 'Pruefmodus: Es werden keine Pakete installiert.' }

try {
    New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
    try {
        Start-Transcript -LiteralPath $logPath -Force | Out-Null
        $transcriptStarted = $true
    } catch { Write-Host "Protokoll konnte nicht angelegt werden: $($_.Exception.Message)" }

    foreach ($gitDirectory in @("$env:ProgramFiles\Git\cmd", "$env:LOCALAPPDATA\Programs\Git\cmd")) {
        if (-not (Get-Command git.exe -ErrorAction SilentlyContinue) -and
            (Test-Path -LiteralPath (Join-Path $gitDirectory 'git.exe'))) {
            $env:PATH = "$gitDirectory;$env:PATH"
        }
    }
    if (Get-Command git.exe -ErrorAction SilentlyContinue) {
        & git.exe --version
        $gitReady = ($LASTEXITCODE -eq 0)
    }

    Write-Host ''
    Write-Host '[1/3] Python 3.12 (64 Bit) und lokale Umgebung'
    if (Test-Python312 $venvPython) {
        $pythonReady = $true
        Write-Host 'Vorhandene lokale Python-Umgebung ist passend.'
    } elseif (Test-Path -LiteralPath $venvPython) {
        Write-Host 'Die vorhandene .cache\venv ist defekt oder verwendet eine falsche Python-Version.'
        Write-Host 'Bitte .cache\venv umbenennen und install.bat erneut ausfuehren.'
    } else {
        $pythonCommand = $null
        $pythonPrefix = @()
        if (Test-Python312 'py.exe' @('-3.12')) {
            $pythonCommand = 'py.exe'
            $pythonPrefix = @('-3.12')
        } else {
            foreach ($candidate in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:ProgramFiles\Python312\python.exe", 'C:\Python312\python.exe', 'python.exe')) {
                if (Test-Python312 $candidate) {
                    $pythonCommand = $candidate
                    break
                }
            }
        }
        if ($pythonCommand) {
            if ($CheckOnly) {
                Write-Host 'Python ist vorhanden. install.bat ohne --check erstellt die lokale Umgebung.'
            } else {
                & $pythonCommand @pythonPrefix -m venv (Join-Path $projectRoot '.cache\venv')
                if ($LASTEXITCODE -ne 0) { throw 'Die lokale Python-Umgebung konnte nicht erstellt werden.' }
                $pythonReady = Test-Python312 $venvPython
            }
        } else { Write-Host 'Python 3.12 (64 Bit) fehlt.' }
    }

    Write-Host ''
    Write-Host '[2/3] Projektpakete, PyTorch mit CUDA und SAM 3'
    if ($pythonReady) {
        & $venvPython -m e57cls.bootstrap --verify
        $packagesReady = ($LASTEXITCODE -eq 0)
        if (-not $packagesReady -and -not $CheckOnly) {
            if (-not $gitReady) {
                Write-Host 'Git fehlt. Fuer die Installation des offiziellen SAM-3-Codes wird Git benoetigt.'
            } else {
                Write-Host 'Pakete werden installiert/repariert. Bitte dieses Fenster offen lassen ...'
                & $venvPython -m e57cls.bootstrap --install-only
                $packagesReady = ($LASTEXITCODE -eq 0)
            }
        }
    }

    Write-Host ''
    Write-Host '[3/3] NVIDIA-GPU / CUDA'
    if ($packagesReady) {
        & $venvPython -c "import torch, sys; ok = torch.cuda.is_available(); print(torch.cuda.get_device_name(0) if ok else 'Keine nutzbare NVIDIA-CUDA-GPU gefunden.'); x = torch.ones(1, device='cuda') if ok else None; torch.cuda.synchronize() if ok else None; sys.exit(0 if ok else 1)"
        $cudaReady = ($LASTEXITCODE -eq 0)
    } else { Write-Host 'CUDA kann erst nach erfolgreicher Paketinstallation geprueft werden.' }
} catch {
    Write-Host ''
    Write-Host "Installationsfehler: $($_.Exception.Message)"
}

# Always report the checkpoint, even after a prerequisite or installation failure.
$modelPath = $null
foreach ($name in @('sam.pt', 'sam3.pt')) {
    $candidate = Join-Path $projectRoot "models\$name"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        if ((Get-Item -LiteralPath $candidate).Length -gt 0) { $modelPath = $candidate }
        break
    }
}

Write-Host ''
Write-Host '============================================================'
Write-Host 'ABSCHLUSS - Was ist bereit, was fehlt noch?'
Write-Host '============================================================'
if ($pythonReady) {
    Write-Host '[OK] Python 3.12 (64 Bit), lokale Umgebung'
} else {
    Write-Host '[OFFEN] Lokale Python-3.12-Umgebung'
    Write-Host '  Falls Python fehlt: Python 3.12 (Windows installer, 64-bit) installieren.'
    Write-Host '  https://www.python.org/downloads/windows/'
    Write-Host '  Im Installer Python Launcher und Add python.exe to PATH aktivieren.'
    Write-Host '  Alternativ im Terminal: winget install -e --id Python.Python.3.12'
    Write-Host '  Danach install.bat erneut doppelklicken.'
}
if ($gitReady) {
    Write-Host '[OK] Git'
} elseif ($packagesReady) {
    Write-Host '[HINWEIS] Git fehlt; die vorhandenen Pakete sind bereits einsatzbereit.'
} else { Write-Host '[OFFEN] Git fuer die Paketinstallation' }
if (-not $gitReady) {
    Write-Host '  Git fuer Windows: https://git-scm.com/downloads/win'
    Write-Host '  Alternativ im Terminal: winget install -e --id Git.Git'
    Write-Host '  Danach install.bat erneut oeffnen.'
}
if ($packagesReady) {
    Write-Host '[OK] Alle festgelegten Projektpakete und Importpruefung'
} else {
    Write-Host '[OFFEN] Python-Pakete'
    Write-Host '  Zuerst offene Python-/Git-Schritte erledigen, dann install.bat erneut starten.'
    Write-Host '  Bei Downloadfehlern Internet/Proxy pruefen. Details stehen oben und im Protokoll.'
}
if ($cudaReady) {
    Write-Host '[OK] NVIDIA-GPU: CUDA-Berechnung erfolgreich'
} elseif ($packagesReady) {
    Write-Host '[OFFEN] NVIDIA-GPU / CUDA nicht nutzbar'
    Write-Host '  Eine kompatible NVIDIA-Grafikkarte ist erforderlich.'
    Write-Host '  Aktuellen Treiber installieren: https://www.nvidia.com/Download/index.aspx'
    Write-Host '  Danach Windows neu starten und install.bat erneut ausfuehren.'
    Write-Host '  Das CUDA Toolkit muss fuer diese PyTorch-Pakete nicht separat installiert werden.'
} else { Write-Host '[UNGEPRUEFT] NVIDIA-GPU / CUDA - zuerst Paketinstallation abschliessen' }
if ($modelPath) {
    Write-Host "[OK] SAM-3-Modelldatei vorhanden: $modelPath"
    Write-Host '  Die Modellinhalte werden beim ersten Klassifizierungslauf geladen.'
} else {
    Write-Host '[FEHLT] SAM-3-Modell: sam.pt / sam3.pt fehlt oder ist leer!'
    Write-Host '  1. https://huggingface.co/facebook/sam3 im Browser oeffnen.'
    Write-Host '  2. Bei Hugging Face anmelden oder ein kostenloses Konto erstellen.'
    Write-Host '  3. Modellbedingungen lesen, akzeptieren und Zugriff anfordern.'
    Write-Host '     Falls noetig auf die Freigabe warten.'
    Write-Host '  4. Unter Files and versions sam3.pt herunterladen (ca. 3.45 GB).'
    Write-Host '     https://huggingface.co/facebook/sam3/tree/main'
    Write-Host '  5. Die vollstaendig heruntergeladene Datei in diesen Ordner kopieren:'
    Write-Host "     $(Join-Path $projectRoot 'models')"
    Write-Host '     sam3.pt darf so heissen oder in sam.pt umbenannt werden.'
    Write-Host '     Eine leere sam.pt entfernen/ersetzen; sam.pt hat Vorrang vor sam3.pt.'
    Write-Host '     Keine SAM-2-Datei und keine model.safetensors verwenden.'
    Write-Host '  6. install.bat erneut starten, um die Installation zu pruefen.'
    Write-Host '  Manueller Download: Hugging Face verlangt eine persoenliche Zustimmung.'
}

$ready = $pythonReady -and $packagesReady -and $cudaReady -and $modelPath
Write-Host ''
if ($ready) {
    Write-Host 'INSTALLATION BEREIT. Zum Starten start_classifier.bat doppelklicken.'
} else {
    Write-Host 'NOCH NICHT STARTBEREIT. Bitte die oben genannten offenen Schritte erledigen.'
    Write-Host 'Anschliessend install.bat erneut starten. Vorhandene Pakete werden wiederverwendet.'
}
if ($transcriptStarted) {
    Write-Host "Protokoll: $logPath"
    Stop-Transcript | Out-Null
}
if ($ready) { exit 0 } else { exit 1 }
