# Verification ledger — 2026-10-13

SLAM / Mobile Mapping ist nur synthetisch verifiziert, bis echte NavVis- oder
XGRIDS-Exporte getestet sind.

| Item | Status | Evidence |
|---|---|---|
| Separate scans and explicit image association | CONFIRMED | Real file has 10 Data3D sections, 10 spherical images and matching GUID associations. The proposed count of eight scans is REFUTED. |
| Chunked pye57 Foundation API reads | CONFIRMED | 94,211,984 records streamed in bounded 100000-point chunks, all valid; synthetic invalid-state and spherical-coordinate tests. |
| ScaledInteger values and extra numeric fields | CONFIRMED | Typed SourceDestBuffer with conversion/scaling; prototype scales and offsets retained in metadata; float64 XYZ and all numeric original arrays persisted. |
| World coordinates and image pose | CONFIRMED | Independent image and scan poses map local to file/world; image pose is not chained through the scan. Georeferenced origins retained in float64. |
| Camera axes and projections | CONFIRMED | E57 committee equations; all 10 actual images pass 200 independent probes, maximum discrepancy 0.00000132 px. Real pinhole/cylindrical samples remain UNCLEAR. |
| Camera/scanner displacement | CONFIRMED | Approximately 0.137652 m in every scan. Explicit camera depth splatting accounts for displacement. |
| Image masks | CONFIRMED | PNG dimensions agree with JPEG dimensions; nonzero is valid. Zero masks suppress rendered and segmented evidence. |
| Alignment review | CONFIRMED by owner | Owner reviewed the images in Cyclone and authorized classification. Low image-edge scores remain warnings, not a confirmation dialog. |
| Visibility | CONFIRMED in synthetic tests | Foreground/background rejection, radial/axis-depth distinction, tolerances use max(abs, rel*depth). Real occlusion spot-check by owner remains UNCLEAR. |
| Units and CRS | CONFIRMED | Metres; complete CH1903+ / LV95 compound WKT retained in LAS WKT VLR. |
| Row/column indices | CONFIRMED | Present in all prototypes; not assumed to define a dense rectangular image. |
| Original input preservation | CONFIRMED | Full SHA256 identical before and after classification: ffee905b9a183ea48a2c3f4ebf895c72357c5b486e32aa7adc54a4785473076a. |
| LAS class ranges | REFUTED original 19–63 assumption | ASPRS LAS 1.4 R16 defines 19–22; 8, 12 and 23–63 are reserved. 64–255 are user defined. Format 6/7 class field is a full byte. |
| LAS 1.4 formats 6/7 and extras | CONFIRMED | Synthetic formats 6/7 and class codes 68/152 read back. Actual output format 7: 94,211,984 points, all class codes present in legend, none zero. class_conf and mirror_state are uint8 extra dimensions. |
| Scan membership | CONFIRMED | Point Source ID is 1-based scan index; uint16 range checked; all 100000 randomly sampled IDs match source membership. |
| Coordinate preservation | CONFIRMED | LAS scale 0.001 m; actual 100000-point readback maximum error <0.0005 m per axis. |
| RGB/intensity | CONFIRMED | RGB preserved as E57-limit-normalized uint16 (8-bit channel *257 for sample); intensity normalized from source limits 0..30 to 0..65535. Original values are never replaced with class colors. |
| GPS/return structure | UNCLEAR / absent here | No timestamps or return fields in sample. LAS mandatory time/return fields remain zero, explicitly reported. No fabricated time base. |
| CUDA and SAM | CONFIRMED | NVIDIA RTX 4090, driver 610.88, torch 2.11.0+cu128, torchvision 0.26.0+cu128. Official SAM revision 0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc and local sam3.pt; 6918 grounding calls in successful full run. |
| CUDA OOM recovery | CONFIRMED | Real 50% memory limit and batch 32 triggered OOM; batch reduced to 16 and all 32 confidence maps completed. diagnostics/oom_smoke.json. |
| Cache reuse and vote-only changes | CONFIRMED | Per-class raw-score identity excludes vote thresholds. Synthetic full pipeline vote-only rerun has zero SAM calls. Ten real semantic overlays exported entirely from existing cache. |
| Pause/cancel/resume | CONFIRMED | Real QProcess command/event test; cooperative cancellation mid-synthetic classification resumes to identical counts. Hard process-kill acceptance remains UNCLEAR. |
| PySide6 Windows | CONFIRMED | 6.12.0 wheels installed; native window rendered and visually inspected. QProcess keeps GUI responsive; BAT startup smoke reports CUDA/model ready and automatic classification. |
| Scene detection accuracy | CONFIRMED on supplied indoor example | All ten scans now indoor, matching owner labels. Positive open-sky evidence is required for outdoor classes. Real outdoor accuracy remains unverified. |
| Density/voxel tuning | UNCLEAR | 0.02 m is a documented starting value; bounded contiguous sampling is not a valid surface-density estimate. |
| Mirror/glass mask-depth rule | CONFIRMED geometrically on supplied example | Supersedes E1/E2: original scan points inside own SAM mask and behind edge/wall plane become 68. All 94,211,984 points checked; zero reflection flags outside masks. Expanded exterior edge recovers scan 9 bathroom mirror, including 180,830 previously uncertain points. SAM mask accuracy is not guaranteed; rejected false masks retain semantic evidence. |
| Intensity/multi-echo for reflection | UNCLEAR | No echo fields; intensity only retained, never used as unverified reflection proof. |
| Large-project resources | UNCLEAR full acceptance | Bounded point chunks and disk grouping implemented. Full 165-scan pipeline stress acceptance outstanding. First full run observed working set about 7.3 GB at a sampled instant; not claimed as measured peak. New runs report sampled peaks. |
| E57 class output/import behavior | UNCLEAR / out of V1 | See future.md. LAS/LAZ is the delivery format. |

## Dependency verification

Every installed distribution was queried against its primary PyPI JSON endpoint
before final locking; evidence: `.cache/dependency_versions_full.json`. All pins
are in `requirements.lock`, including the official SAM Git commit and CUDA wheel
versions. Tested compatibility takes precedence over blindly upgrading to the
newest release. See DECISIONS.md for numpy/scipy, CUDA, Triton and transitive pins.
The lock verifier also checks SAM's installed Git revision, not only its version.
Fresh environment installation requires Git and initial package download access.

## Primary sources

- https://github.com/davidcaron/pye57/blob/master/src/pye57/libe57_wrapper.cpp
- https://github.com/davidcaron/pye57/blob/master/tests/test_main.py
- https://asmaloney.github.io/libE57Format-docs/d1/d28/structe57_1_1_image2_d.html
- https://e57-3d-imgfmt.sourceforge.net/bestCoordinates.html
- https://e57-3d-imgfmt.sourceforge.net/bestReader.html
- https://docs.rs/crate/e57/0.10.0/source/src/images.rs
- https://github.com/ASPRSorg/LAS/tree/1.4.R16
- https://laspy.readthedocs.io/en/latest/
- https://github.com/facebookresearch/sam3/tree/0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc
- https://download.pytorch.org/whl/cu128
- https://pypi.org/pypi/PySide6/json
