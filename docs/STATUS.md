# Status

This is a chronological development log. The first section describes the current
behaviour and supersedes the milestone sections below where they differ; for
example, mirror and glass analysis was off by default at M6 and is now on.
Paths under `Test_pw/` and `e57_findings.md` refer to local test data and notes
that are not part of the repository.

## Current owner corrections (2026-10-13)

- Mirror and window glass enabled by default. Two text prompts must agree. Code 68 uses only the source scan's exact mask and points behind a supported plane. The main cloud retains every point.
- Plane samples come only from the exterior mask edge / neighbouring wall, never mask interiors or enclosed holes. A failed 10-pixel edge expands to 30 pixels. Unknown masks no longer suppress semantic votes.
- Real validation of all 94,211,984 source points: 299,354 reflection flags, none outside the source mask. Scan 9 has 180,830 formerly uncertain points recovered. Report: `Test_pw/klassifiziert/validation_v3_20261013_195428/reflection_validation.json`.
- All ten scene decisions now indoor; uncertain automatic cases use indoor classes, outdoor requires positive open-sky evidence.
- Code 76 now includes loose household objects in both scenes. Strong stairs win floor/ceiling/ground conflicts across scans.
- Indoor sanitary fixtures 63, outdoor manhole covers 75 and tree trunks 95 added, with English prompts. Real SAM bathroom smoke identifies toilet, sink and bathtub. Outdoor object masks still need a real outdoor example.
- Reflection progress counts all images and views; parent-phase timing fixes the spurious zero-minute ETA. Initial / insufficient estimates show calculation pending. Segmentation duration excludes reflection analysis.
- Automated verification: 44 tests, including installer checks, exterior-wall fallback with deliberately corrupted near-edge points, strict own-mask/holes, uncertainty-to-semantic behavior, final class-68 export, scene gating, added classes, stair priority and GUI ETA.
- The completed user run `run_20261013_155456` precedes the latest wall-edge/ETA/new-class changes. No replacement full LAS has been launched; restart the BAT for the owner's next test.


## M0 — implemented and tested (2026-10-13)

`python -m e57cls inspect <file.e57> --out <empty-folder>` produces JSON and Markdown
reports using metadata, a bounded point sample per scan and every image blob
header. Real sample: 10 scans, 10 panoramas, 94211984 records, 100000 sampled per
scan, no invalid records in those samples. Input size/mtime unchanged.

Seven automated tests passed: bounded sample/input hash preservation, missing
images, output overwrite protection, chunk boundaries and buffer ownership,
invalid coordinates, JPEG/PNG bounded headers, poses and parameter bounds.

`start_classifier.bat` and `models/` were added at the owner's request. Model
path: `models/sam.pt`. At M0 the BAT launched the inspection chooser; this historical stage was replaced
by automatic classification and the M7 GUI described below.

Open VERIFY items: empirical image alignment and mask semantics, unbiased density,
mirror presence/evidence, full point counts, LAS specification/readback,
GPU/SAM smoke test, PySide6 and later dependency wheels. Details and evidence are
in verification.md. Geometry path A is selected in e57_findings.md.

## M1 — implemented and tested (2026-10-13)

The full sample was streamed into the scan store in 100000-record chunks.
94211984 valid points + 0 invalid records = 94211984 declared records.
Measured store construction: 68.86 seconds (excluding initial input hashing).
Coordinates remain float64 and all numeric attributes are preserved, including
RIEGL extensions. Original record ordering and validity masks are retained.
Mappings are closed after each bounded slice; completion manifests are atomic
and reference hash-verified arrays. Incomplete generations are never accepted.
The observed worker peak working set during processing was about 81 MiB; a full
165-scan resource acceptance test remains outstanding.

Nine automated tests passed, including complete record/attribute preservation,
reusing a completed scan with a changed chunk size, interrupted-write recovery,
and corruption detection. Report: Test_pw/klassifiziert/run_20261013_133730/store_report.json.

Open VERIFY items: empirical projection/image masks, surface density, mirrors,
LAS, SAM/CUDA, PySide6, later dependency wheels and 165-scan scaling.

## M2 — implemented and tested (2026-10-13)

All 10 images passed 200 independent projection probes each. Maximum discrepancy
was 0.00000132 pixels. Camera displacement, global float64 origins, panorama seam,
all three documented camera models, view rays and axis-depth visibility have
synthetic tests. Total suite: 17 tests passed. Full-file diagnostic projection
and three rendered views per panorama completed in 114.88 seconds. Observed peak
working set during rendering was about 1.34 GiB.

The owner confirmed image alignment in Cyclone and authorized classification.
Per the subsequent instruction, prechecks are logged and no review dialog blocks
classification. Low edge-alignment scores on scans 5–7 remain warnings.
The supplied sample only exercises spherical cameras; pinhole/cylindrical real
files remain UNCLEAR. Surface-density tuning and mirror presence remain open.

## M3 — implemented and smoke-tested (2026-10-13)

Official local SAM 3 checkpoint loaded on RTX 4090 with torch 2.11.0+cu128.
Real rendered view segmented using wall/floor/door prompts. Per-class float16
raw-confidence caches retain scores below voting thresholds; changing one prompt
only misses that class. Overlapping tiles merge by maximum. Prompt-batch and
tile OOM retries record effective reductions. 18 tests passed after switching to
SAM-compatible numpy 1.26.4. Forced real OOM acceptance remains open.

User-supplied models/sam3.pt is supported alongside preferred models/sam.pt.
SAM source is pinned to 0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc (official HEAD
verified with git ls-remote). Third-party SAM imports still emit a timm deprecation
warning; our adapter does not use that deprecated import.

## M4 — implemented and tested (2026-10-13)

Streaming LAS 1.4 output, global voxel votes, one observation per scan, raw RGB,
Point Source ID, extra bytes, count checks, coordinate readback, legends and reports.
Full synthetic E57 with embedded image -> LAS test passes; changing only minimum
observations reruns with zero SAM calls. 21 tests pass. Whole real-file classification
and 100000-point readback are next. Extended classes 68 and 152 and optional filtered
output are read back correctly. Unknown GPS/return attributes remain zero and logged.
Open VERIFY: real scene accuracy, mirror presence, GUI, forced OOM, 165-scan scaling.

## M5 — implemented and tested (2026-10-13)

Per-scan upper-hemisphere sky/ceiling evidence, upper returns and seeded overhead
plane fit. Mixed or insufficient coverage retains both class sets and logs a warning.
Manual indoor/outdoor selection and per-scan overrides supported. Preset validation
checks code ranges, duplicate IDs/codes/prompts, thresholds and fixed special codes.
24 tests pass. Human-labelled scene acceptance on all real scans remains UNCLEAR.

## M6 — implemented and synthetic evidence tests passed (2026-10-13)

Optional auxiliary mask pass precedes all semantic votes. Bounded frame-plane fits
build a global extent registry first. All scans, including scans without a mask,
are then tested against all registered planes. Direct measurements are disk-indexed;
confirmation excludes mirror rays and the candidate's own scan. States 0–4 and
class 68 are written without deleting primary-output points. Reflected pixels and
state-2 points cannot contribute semantic votes. Synthetic tests cover two mirror
views, direct third-scan confirmation, missing own mask, surface/front points,
unknown planes, E2 confirmation and registry merging. Optional feature defaults off.
Open VERIFY: actual mirrors in the supplied file and real mask/plane quality.

## M7 — implemented and tested (2026-10-13)

German four-tab PySide6 GUI, QProcess worker, automatic start after native E57
selection, class/preset validation, profiles/advanced parameters, progress/ETA,
pause/cancel/resume, saved-run loading, result/report/log buttons. Per latest owner
instruction the inspection gate and output-folder selector are absent. BAT repairs
the locked environment, verifies CUDA and local sam.pt/sam3.pt, then launches Qt.
Native Windows rendering was checked; a live watcher was opened for the first run.
28 automated tests pass, including real QProcess JSONL pause/resume/cancel and
classification cancellation/resume with identical synthetic class counts.

## Full sample acceptance — completed (2026-10-13)

Output: Test_pw/klassifiziert/run_20261013_133730/Test_PW_1_classified.las.
10/10 scans processed; 94,211,984 valid input and output points; zero invalid records.
LAS 1.4 format 7, original RGB, full-byte classes, scan IDs and extra bytes.
8.123601% unclassified. 6918 SAM grounding calls; total 2845.188 seconds.
Random 100000-point readback: maximum XYZ errors 0.0004999894, 0.0004999973,
0.0004999979 metres; all sampled scan IDs correct and all output classes in legend.
Input SHA256 remains ffee905b9a183ea48a2c3f4ebf895c72357c5b486e32aa7adc54a4785473076a.
Semantic overlays for all 10 first views exported with zero additional SAM calls;
overlays from scans 1, 5 and 10 visually inspected as development diagnostics.
Owner classification review in a point-cloud viewer remains an acceptance step.
Earlier 117 report/diagnostic files were SHA256-verified after copying beneath this
run, then the obsolete project-root output directory was removed.

Remaining empirical verification: real planar mirrors,
full pipeline resource scaling at 165 scans, hard process-kill recovery, new real
pinhole/cylindrical files, and owner-labelled scene/class accuracy. These are not
claimed as passed by the synthetic tests or coordinate-preservation checks.


## Additional GPU robustness verification

Real CUDA OOM was forced with 50% of RTX 4090 memory and an initial prompt batch
of 32. The backend reduced it to 16 and completed all 32 probability maps without
aborting. Evidence: diagnostics/oom_smoke.json in the sample run folder. No model
or input files were modified. Scene/class visual acceptance remains the owner's
review step; geometric preservation does not establish semantic accuracy.
