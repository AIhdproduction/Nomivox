# Nomivox — E57 SAM 3 Point Cloud Classifier

Local classification of E57 point clouds using **SAM 3**, embedded scan images,
and an NVIDIA GPU. The application detects objects in the images, transfers their
classes to the corresponding 3D points, and exports a classified **LAS or LAZ file**.
The desktop interface is in German.

This is a standalone Windows application. Metashape and Cyclone are not required
to run the classifier.

> **License:** AIhdproduction Education and Research License 1.1;
> source-available, not open source. Free for noncommercial education, research
> and personal use; commercial and other uses require prior written permission.
> SAM 3 is licensed separately by Meta. See
> [License and provenance](#license-and-provenance).

## Features

- Process E57 files containing multiple scan positions and associated embedded images.
- Select indoor and outdoor classes automatically or use manual scene settings.
- Use 34 object classes with 246 English search prompts, and edit or add classes.
- Combine spatial evidence from multiple scan positions, prioritizing confidently detected stairs over floors and ceilings.
- Detect mirrors and window glass, assigning class 68 to points behind their estimated surfaces within the corresponding image masks.
- Export valid original points with coordinates, available RGB colors, and scan membership.
- Display progress, elapsed time, and estimated remaining time; pause, cancel, and resume runs.
- Save settings, class legends, checks, warnings, and diagnostic images alongside the output.

Image and point processing runs locally. Initial setup downloads software
packages from the internet. The model checkpoint is obtained separately.

## Requirements

| Requirement | Details |
|---|---|
| Operating system | 64-bit Windows |
| Python | Python 3.12, 64-bit, available through `py -3.12` or `python` |
| Git | Required to install the pinned SAM 3 source revision |
| Graphics | CUDA-capable NVIDIA GPU with a compatible driver; CPU-only operation is not supported |
| Model | Official SAM 3 image checkpoint at `models/sam.pt` or `models/sam3.pt` |
| Disk space | Space for the model, Python environment, point cache, masks, and outputs; allow substantially more than the input E57 file size |
| Input | E57 with usable scan/image poses, camera parameters, and unambiguous image associations |

The application has been tested with an **RTX 4090**, Python 3.12, and PyTorch
2.11.0 with CUDA 12.8. This is a tested configuration, not an established minimum
hardware requirement. Package versions are pinned in
[requirements.lock](requirements.lock).

Images must already be correctly aligned with the point cloud. The application
does not perform registration or automatically correct inaccurate camera poses.

## Installation and startup

1. Extract or clone [AIhdproduction/Nomivox](https://github.com/AIhdproduction/Nomivox) into a local folder.
2. Install Python 3.12 and Git if they are not already available.
3. Double-click **[`install.bat`](install.bat)**. It creates the local Python environment,
   installs or repairs the pinned dependencies, tests CUDA, and summarizes anything
   still missing. Leave the installer window open until it finishes.
4. Obtain the official checkpoint from [Meta / SAM 3](https://huggingface.co/facebook/sam3)
   and follow the applicable access and license terms.
5. Save the model as **`models/sam.pt`**. **`models/sam3.pt`** is also supported;
   if both exist, `sam.pt` takes precedence. Renaming another SAM version's weights
   does not make them compatible with SAM 3.
6. Run **`install.bat`** again to check that setup is ready, then double-click
   **[`start_classifier.bat`](start_classifier.bat)** to open the application.
7. Select an E57 file. **Images and camera metadata are checked first.** If no scan
   has usable associated images and camera data, a warning explains what is missing
   and the run stops before point-cache creation or SAM loading. Point RGB colors
   alone are insufficient. Otherwise, classification starts automatically.

The installer uses [scripts/install.ps1](scripts/install.ps1) and saves its log to
`.cache/install.log`. It can install packages before the model is available and
provides manual download instructions if the checkpoint is missing. Python, Git,
the NVIDIA driver, and model weights are not installed automatically by this script.

To check an existing setup without installing packages, run
`install.bat --check`. Add `--no-pause` when running it from an existing terminal
or a script. The initial package downloads can total several gigabytes.

The startup launcher also verifies the environment in `.cache/venv` and can repair
dependencies. See [models/README.md](models/README.md) for model folder instructions.

Input checks are logged. There is no separate image-alignment approval step,
but missing or invalid required data can still prevent processing.

## Using the interface

The following labels match the German desktop interface:

| Tab | Contents |
|---|---|
| **Eingabe** (Input) | E57 file, scan overview, image associations, and metadata |
| **Klassen** (Classes) | Scene selection, active classes, codes, colors, English prompts, presets, and mirror/glass options |
| **Einstellungen** (Settings) | Image profile, resolution, confidence, voxel size, visibility, GPU memory, and LAS/LAZ output |
| **Lauf** (Run) | Progress, log, elapsed and remaining time, pause, cancel, resume, and output links |

The initial automatic run uses the default settings. Once a run finishes or is
cancelled, change settings and select **Neuen Lauf starten** (Start new run).
**Fortsetzen** (Resume) uses the original settings. **Gespeicherten Lauf fortsetzen**
(Resume saved run) loads the `run.json` from a previous output folder. Reopen the
interface after updating the software or default preset.

The remaining-time estimate applies to the current phase and uses measured
progress across all its images or scans. Until enough information is available,
the interface displays **Restzeit wird berechnet …** (Calculating remaining time).
Cache hits, different image contents, and GPU load can affect the estimate.

The status line also shows **Gesamtrestzeit**: a rough estimate for the entire
remaining run, including the current and subsequent phases. It extrapolates
elapsed time from the weighted overall progress and updates continuously.
Because phase weights are fixed and actual phase durations vary, this estimate
can change substantially. Until enough progress is available, it displays
**Gesamtrestzeit wird berechnet …**. Hover over the status line for details.

## How classification works

1. **Inspect and read the E57.** Scan positions, images, poses, and attributes are
   checked. Point data is read into a local cache in bounded chunks.
2. **Prepare images.** Panoramas are rendered into perspective views for SAM.
   Scan and camera poses are handled separately. Spherical, cylindrical, and
   pinhole camera models are implemented; the real test dataset contains panoramas.
3. **Analyze mirrors and glass.** Text prompts produce masks. Planes are estimated
   from their outer edges and neighboring walls; the depth rule determines class 68.
4. **Determine the scene.** Sky, ceiling, and upper-hemisphere laser returns provide
   indoor/outdoor evidence. Outdoor classes require positive open-sky evidence
   without returns. Uncertain automatic decisions use indoor classes; supported
   mixed scenes can use both class sets. Manual settings take precedence.
5. **Segment objects.** SAM processes the English prompts for each class. Ordinary
   class segmentation combines prompt confidence maps by taking their maximum.
   Overlapping image tiles, confidence thresholds, and a mask-boundary margin
   control the transfer of labels.
6. **Project onto points and combine votes.** Projection and depth checks restrict
   label transfer to visible points. Spatial voxels collect class evidence;
   multiple views from the same scan do not count as independent scan observations.
7. **Export original points.** Classes are transferred to the valid original
   points. The export is read back to check point counts and class codes, plus
   sampled coordinates and scan membership.

The default voxel size is **2 cm**. Voxels combine classification evidence;
the export contains the valid original points, not just voxel representatives.
Stairs with at least **0.65 confidence** take priority over ground, floor, and
ceiling in conflicting observations, including observations from different scans.
Their vote share must also reach 25%. Post-processing preserves only geometrically
stepped stair components, retains sparse semantic surfaces when their connected
10 cm cell component is large enough, and expands direct class transfer from
3 mm per metre of scanner distance up to 5 cm. A zero transfer range factor
keeps the previous fixed-radius behaviour.

Raw masks are cached. Changing prompts recomputes the affected classes. Changes
to voting thresholds can reuse existing raw masks. If GPU memory runs out, the
application attempts smaller prompt batches or image tiles and logs the changes.

### Mirrors, glass, and class 68

Mirror and window-glass analysis are enabled by default. Mask detection requires
agreement from **two distinct prompts** by default.

Planes are estimated exclusively from measured points **outside the mask interior**:
first within a narrow, 10-pixel outer boundary band, then within a 30-pixel band
of the neighboring wall if the first attempt finds no sufficiently supported
plane. Enclosed mask holes also provide no plane-fitting samples. These adjustable
widths refer to the reduced mask resolution.

A point receives class **68** only if it belongs to the relevant scan, projects
inside that scan's associated image mask, and its scan ray intersects the bounded
plane before reaching the point. By default, the point must lie more than
**3 cm along the ray** behind the intersection. This classification is determined
separately for original points and is not propagated to other scans through
global voxel votes.

Without a sufficiently supported plane, the reflection status remains uncertain
and ordinary object classification remains available. A mask alone does not
assign class 68. The window-glass rule explicitly includes **real outdoor points
behind the pane** when they meet the mask and depth conditions. Class 68 therefore
represents this geometric rule, rather than a general physical proof of reflection.

The main output retains all valid points. An optional additional output excludes
class 68. Mask images and plane diagnostics support visual review.

## Editing classes and prompts

The central preset is **[presets/default.json](presets/default.json)**.

| Field | Meaning |
|---|---|
| `id` | Internal class identifier |
| `name_de` / `name_en` | Display name |
| `code` | Export class number unless an explicit code mapping is configured |
| `scene` | `indoor`, `outdoor`, or `both` |
| `prompts` | English terms used by SAM to find the class |
| `enabled` | Enable or disable the class |
| `min_instance_conf` / `min_mask_conf` | Optional per-class thresholds; `null` uses the global value |

For example, the outdoor ground class searches for `ground`, `soil`, `earth`,
`gravel`, and `pebbles`, among other terms. Indoor prompts such as `parquet floor`,
`tiled floor`, `sofa`, `wardrobe`, `toilet`, and `wash basin` belong to their
respective classes. Class 76 covers loose objects such as clothing, suitcases,
bags, toys, blankets, and tools.

Additional prompts broaden the search but do not guarantee higher accuracy and
can increase processing time. Prefer specific terms; duplicate prompts in
competing classes are flagged during preset validation. The `auxiliary` section
contains helper prompts for sky, mirrors, and window glass. **Unclassified (1)**
is the fallback class, while **Reflection (68)** follows the mask/geometry rule
above. Neither is an ordinary text-searched object class.

### Included object classes

| Code | Class | Scene |
|---|---|---|
| 2 | Ground | Outdoor |
| 3 | Low vegetation | Outdoor |
| 5 | Tall vegetation, tree canopies, and hedges | Outdoor |
| 6 | Buildings | Outdoor |
| 9 | Water | Outdoor |
| 11 | Roads | Outdoor |
| 14 | Overhead power lines | Outdoor |
| 63 | Sanitary fixtures | Indoor |
| 67 | Construction machinery | Outdoor |
| 73 | Sidewalks | Outdoor |
| 74 | Curbs | Outdoor |
| 75 | Manhole covers | Outdoor |
| 76 | Loose objects, household clutter, and materials | Indoor and outdoor |
| 77 | Fences and outdoor guardrails | Outdoor |
| 81 | Traffic signs | Outdoor |
| 84 | Utility poles and streetlights | Outdoor |
| 91 | Vehicles | Outdoor |
| 95 | Tree trunks | Outdoor |
| 100 | Walls | Indoor |
| 101 | Ceilings | Indoor |
| 102 | Floors | Indoor |
| 103 | Doors | Indoor |
| 104 | Windows | Indoor and outdoor |
| 105 | Stairs | Indoor |
| 107 | Railings | Indoor |
| 108 | Columns and supports | Indoor and outdoor |
| 109 | Beams and girders | Indoor and outdoor |
| 110 | Steel structures and scaffolding | Indoor and outdoor |
| 111 | Pipes | Indoor |
| 112 | Cables | Indoor |
| 113 | Electrical equipment | Indoor |
| 114 | Heating, ventilation, and air conditioning | Indoor |
| 116 | Ceiling equipment | Indoor |
| 117 | Furniture | Indoor |

The exported legend documents the project-specific meaning of each code.
**63** is explicitly used for sanitary fixtures here, although this code is
reserved in the LAS standard. The application reports a warning and exports
the requested code; check the mapping in your target software against the legend.

## Outputs and folder structure

The following directory is created automatically beside the selected E57 file:

```text
ProjectFolder/
├── scan.e57
└── klassifiziert/
    └── run_YYYYMMDD_HHMMSS/
        ├── scan_classified.las       # or .laz
        ├── scan_classified_no_reflection.las   # optional copy without class 68
        ├── report.html / report.json
        ├── legend.html / legend.csv / legend.json
        ├── preset.json              # when started through the interface
        ├── run.json / run_config.json
        ├── scan_index.json
        ├── inspect_report.json / inspect_report.md
        ├── store_report.json
        ├── reflection_report.json   # when mirror/glass analysis is enabled
        ├── mirror_registry.json
        ├── log.txt / events.jsonl
        └── diagnostics/
            └── reflections/         # masks, overlays, and plane details
```

The input E57 is not modified. LAS/LAZ 1.4 contains valid original points,
available RGB colors, classes, and the scan ID as `Point Source ID`. The extra
fields `class_conf` and `mirror_state` store the class voting share and reflection
status. Class colors are provided in the legend; they do not replace original
RGB colors. Points without an accepted class assignment remain in class 1.

Invalid points, failed scans, and attributes not transferred are logged. Not all
E57-specific attributes are carried into LAS. Coordinates use 1 mm LAS scaling
by default. GPS times and return information without a verified basis are not
invented. Export to a classified E57 is not currently implemented.

The application folder contains:

```text
install.bat             Dependency installation and readiness checks
scripts/install.ps1     Installer implementation and setup diagnostics
start_classifier.bat    Windows launcher
presets/               Default classes and prompts
e57cls/                Application source code
tests/                 Automated tests with generated test data
docs/                  Technical documentation and license notices
models/                Local checkpoint and setup instructions
.cache/                Python environment, model/point/mask caches, and run state
Test_pw/               Local test point clouds; excluded from the Git repository
```

## Command line and tests

Run these commands from the project folder:

```powershell
# Install or repair dependencies and report missing prerequisites
.\install.bat --no-pause

# Check setup without installing packages
.\install.bat --check --no-pause

# Check startup without classifying a point cloud
.\start_classifier.bat --check-startup

# Open the interface with an E57 file; classification starts automatically
.\start_classifier.bat "D:\Data\Project\scan.e57"

# Inspect input data only; use a new diagnostics folder
.cache\venv\Scripts\python.exe -m e57cls inspect "D:\Data\Project\scan.e57" --out "D:\Data\Project\klassifiziert\input_check"

# Run automated tests
$env:QT_QPA_PLATFORM = 'offscreen'
.cache\venv\Scripts\python.exe -m pytest --basetemp .cache\t -q
Remove-Item Env:QT_QPA_PLATFORM
```

For a new CLI run, create a configuration such as `.cache/run_example.json`:

```json
{
  "source": "D:/Data/Project/scan.e57",
  "scene_mode": "both",
  "format": "las"
}
```

```powershell
.cache\venv\Scripts\python.exe -m e57cls run --config .cache\run_example.json
```

Für einen abgeschlossenen Lauf erzeugt der folgende Befehl einen Qualitätsbericht
in `quality/` mit `quality.json`, deutschen Draufsichten und einer HTML-Übersicht:

```powershell
.cache\venv\Scripts\python.exe -m e57cls quality "D:\Data\Project\klassifiziert\run_..." --step 40
```

Mit `cache_cleanup: "keep_all"` kann die gespeicherte Abstimmung ohne erneute
Bildsegmentierung nachklassifiziert werden. Nur Nachbearbeitungs- und Exportwerte
in der Override-Datei sind zulässig; der ursprüngliche Vote-Cache bleibt erhalten.

```powershell
.cache\venv\Scripts\python.exe -m e57cls reclassify --from "D:\Data\Project\klassifiziert\run_..." --out "D:\Data\Project\klassifiziert\run_tuned" --config overrides.json
```

The retained post-processing controls include `geometry_rules`, `vote_families`,
and the `terrain_*`/`tree_*` thresholds. Geometry rules create a compact terrain
artifact for the run. Its optional output-only terrain fill assigns class 2 to an
otherwise unclassified point only when uncovered DTM evidence supports it.

Without an explicit `output` value, the CLI creates a new run folder beside the
E57. `scene_mode: "both"` enables automatic scene detection. Use `indoor` or
`outdoor` for a manual mode. To resume a run, use its saved configuration;
changed settings require a new output folder. Completed outputs are not overwritten.

## Mobile Mapping / SLAM

Neben statischen TLS-E57-Dateien erkennt Nomivox mobile SLAM-Exporte automatisch;
`input_mode` kann mit `auto`, `static` oder `slam` festgelegt werden. Im SLAM-Modus
werden nur z-buffer-sichtbare Punkte einer Kameraperspektive gelabelt und Kameras zu
virtuellen Stationen zusammengefasst. Klasse 68 ist dort deaktiviert. Die maximale
Reichweite beträgt standardmässig 20 m innen und 40 m aussen; `slam_max_range`
überschreibt beide Werte. SLAM ist bisher nur synthetisch verifiziert, nicht mit
realen NavVis- oder XGRIDS-Exporten.

Die Verarbeitung nutzt einen gemeinsamen SAM-Durchgang, Parallelisierung,
kachelweisen Transfer, kompakte Datenspeicher, einen Score-Floor und bereinigt
den Cache nach erfolgreichem Lauf gemäss Einstellung.

## Validation and limitations

Processing has been checked with a real file containing **10 scans and around
94 million points**. Tests cover projection, point-label transfer, export, scene
selection, stair priority, mirror masks, edge/wall plane fitting, and the interface.
All ten scan positions in the verified indoor dataset are recognized as indoor.

SAM can miss objects, produce overly broad masks, or confuse classes. The latest
prompt additions have been checked for consistency; improved recognition accuracy
has not yet been demonstrated. Real outdoor scenes and very large projects have
been tested less extensively than the available indoor dataset. Review results
in a point-cloud viewer and inspect the diagnostic images.

Technical details: [Status](docs/STATUS.md), [verification ledger](docs/verification.md),
[worker protocol](docs/protocol.md), [decisions](DECISIONS.md), and
[planned extensions](docs/future.md).

## Git and local files

[.gitignore](.gitignore) excludes local test data, point clouds, `klassifiziert/`,
model weights, `.cache/`, virtual environments, temporary files, local settings,
and secret files. Source code, presets, automated tests, the README, and license
notices remain eligible for version control. Adding ignore rules does not
automatically untrack files that Git already tracks.

## License and provenance

Original project material is covered by the
**AIhdproduction Education and Research License 1.1** in [LICENSE](LICENSE),
licensed by Domenic Hochstrasser, doing business as AIhdproduction. The
noncommercial education, noncommercial research and personal uses described
there are permitted without a fee. Commercial and other uses outside that
permission require prior written approval; request it by email at
hdproduction.ch@gmail.com. Contributions fall under the Contributions section
of [LICENSE](LICENSE); see [CONTRIBUTING.md](CONTRIBUTING.md). The English
license text is authoritative. This is a source-available license with purpose
restrictions, not an open-source license.

| Material | Terms |
|---|---|
| All Nomivox code, scripts, presets, and documentation in this repository | [LICENSE](LICENSE) |
| SAM 3 code and model weights (obtained separately, not in this repository) | Meta's SAM License, copy in [docs/LICENSE_SAM3.txt](docs/LICENSE_SAM3.txt) |
| PyTorch, PySide6/Qt, pye57, and other installed packages | Their own licenses, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

The two sets of terms apply side by side. Permission under the Nomivox license
does not extend Meta's terms, and Meta's SAM License does not lift the
noncommercial restriction on Nomivox itself. The SAM License has its own
conditions, including acknowledgment in research publications and trade-control
and prohibited-use restrictions; read it before using the model.

Dependency notices and the provenance of the SAM adapter are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights, the Python
environment, and local test data are not distributed with the repository.
