# Decisions

- 2026-10-19: The high image profile retains the established 17-view plan and
  adds eight 50° perspectives at −10° pitch for distant surfaces below the
  horizon. Existing fast and balanced view IDs and cache identities remain
  unchanged; ETA accounting uses the actual view count of each image.

- 2026-10-19: Stair labels require a stepped geometric profile: at least three
  horizontal levels with a configured rise and tread. Small stair components
  are restored rather than preserved, and levels matching a nearby floor are
  returned to that floor. Cleanup retains sparse components when their shared
  10 cm feature-cell component is sufficiently large; cable and electrical
  rules keep their existing voxel/span behavior. Class transfer may grow with
  distance to the source scanner (or observed SLAM station), while a zero
  range factor preserves the historic fixed-radius result.

- 2026-10-18: The previous no-extrapolation rule is lifted only for the
  class-2 terrain fill. An unclassified output point may become ground when
  the persisted DTM marks its cell as uncovered terrain and its height is
  within the configured terrain tolerance. The fill uses confidence 64 and
  remains independently switchable.

- 2026-10-18: Nachklassifizieren verwendet ausschliesslich eine Kopie der
  gespeicherten Vote-Datenbank aus Läufen mit `cache_cleanup = keep_all`.
  Segmentierungs-, Sichtbarkeits- und Reduktionsparameter bleiben gesperrt;
  dadurch sind mehrere sichere Nachbearbeitungsvarianten möglich.

- 2026-10-18: SLAM bewertet Sichtbarkeit pro Kameraperspektive mit einem
  z-gepufferten Tiefenbild. Dadurch erhalten verdeckte Punkte keine Bildlabels.
  Kameras werden entlang der Trajektorie zu virtuellen Stationen gruppiert; diese
  Stationen, nicht Datenblöcke, sind die unabhängigen Stimmen. Klasse 68 bleibt
  deaktiviert, weil ein SLAM-Datenblock keinen belastbaren Scanursprung besitzt.
  Spiegel-/Glasmasken dürfen vorhandene Bildpixel dennoch von normalen Labels
  ausschliessen. Die Gewichtung multipliziert SAM-Konfidenz mit Faktoren für
  Distanz, Einfallswinkel und Bildmitte.
- 2026-10-18: SLAM entscheidet die Szene einheitlich für die gesamte Trajektorie.
  Es verwendet Himmelsmasken und z-buffer-sichtbare Punkte über Kameras, niemals
  die TLS-Regel der oberen Laser-Halbkugel. Das stabilisiert die Reichweite über
  benachbarte Stationen (20 m innen, 40 m aussen, manuell überschreibbar).
- 2026-10-18: Gemeinsamer SAM-Durchgang, Parallelisierung, Kachel-Transfer,
  kompakte Stores, Score-Floor und gezieltes Cache-Cleanup begrenzen Laufzeit,
  GPU-Speicher und Wiederholungsaufwand.

- 2026-10-13, latest owner changes: replace the old mirror-only/E1/E2 policy with scan-local mask plus depth. Enable mirrors and window glass by default; label points behind the surface inside their own mask 68, including real exterior points behind glass. Never transfer class 68 through global voxel votes. Keep all points in the main LAS.
- Surface estimation uses only the outer mask edge and neighbouring wall. Try a 10-pixel outside band, then 30 pixels if no supported plane is found. Exclude mask interiors and enclosed holes. Unknown masks no longer erase ordinary semantic evidence. No plane guessed from virtual interior geometry.
- Require two distinct mirror/glass prompts for each accepted mask pixel. Remove the phrase mentioning a bathroom sink after it caused a real false positive. SAM can still make mistakes; diagnostics retain all masks and plane failures.
- Prefer indoor classes when automatic scene evidence is ambiguous. Require positive open-sky evidence to enable outdoor classes. Add household clutter to code 76 in both scenes and strong stair priority over floor/ceiling/ground across scan votes.
- Owner explicitly requested project code 63 for sanitary fixtures, 75 for outdoor manhole covers and 95 for outdoor tree trunks. Keep reserved-code warnings informational; do not remap code 63. All search prompts are English; tree-canopy prompts exclude the whole-tree prompt to reduce trunk competition.
- Estimate time using cumulative progress across the entire reflection phase and its actual start time. Show calculation pending until measurable progress exists; do not display zero minutes for an unfinished phase. Segmentation timing starts after reflection analysis.

- 2026-10-13: Owner removed the separate inspection/review stop: log validation and continue classification automatically after file selection. Still reject corrupt/missing mandatory data.
- 2026-10-13: Accept models/sam3.pt as a fallback to models/sam.pt because the owner supplied the official checkpoint under its original name.
- 2026-10-13: PyPI latest torch is 2.14.1, but the official CUDA 12.8 index provides at most 2.11.0; use torch 2.11.0+cu128 and torchvision 0.26.0+cu128, verified together.
- 2026-10-13: Current official SAM source still requires numpy<2; use numpy 1.26.4 and compatible scipy 1.17.1 instead of latest numpy 2.5.4/scipy 1.18.1. Re-run earlier tests after this dependency change.

- 2026-10-13: Owner confirmed image alignment after reviewing in Cyclone and explicitly authorized classification. Keep quantitative alignment warnings as diagnostics.
- 2026-10-13: Owner overrides output naming: create `klassifiziert/run_YYYYMMDD_HHMMSS` beside the input E57, containing checks, logs and the classified cloud together.

- 2026-10-13: Runtime writes go to the chosen output and cache directories; source code, tests and documentation are development artifacts in this repository. The input E57 is opened read-only.
- 2026-10-13: Use pye57 0.4.19 Foundation API, not whole-scan convenience reads: real-file bounded reads, scaled integers and image blobs work on Windows CPython 3.12.
- 2026-10-13: Use geometry path A for the supplied file: every image is spherical. Generate perspective views before SAM.
- 2026-10-13: Preserve coordinates in float64 in the scan store. This supersedes the proposed xyz_local.f32.npy because preservation is a fixed requirement.
- 2026-10-13: Use explicit GUID associations only at M0. Uncertain associations remain excluded; no guessed camera transformations.
- 2026-10-13: M0 samples the first 100000 records per scan. Report sampling bias and do not infer full invalid counts or a validated voxel size from this contiguous sample.
- 2026-10-13: Retain unflagged zero coordinates; zero range is not inherently invalid under the documented E57 coordinate restrictions.
- 2026-10-13: Quaternion signs are preserved; q and -q represent the same rotation. Several sample quaternions have negative w, although the library data-structure comment requests nonnegative w.
- 2026-10-13: User requested checkpoint filename models/sam.pt. Use this instead of the older project's models/sam3.pt.
- 2026-10-13: The BAT exposes the completed M0 inspector now. Its native file dialog uses standard-library Tk; the final four-tab GUI remains M7/PySide6. No model or CUDA check is needed for M0.
- 2026-10-13: Image headers are read with a 1 MiB bound. Full panorama decoding belongs to geometry and must enforce a configurable pixel budget.


- 2026-10-13: SQLite external grouping replaces proposed Parquet/pyarrow; bounded transactions and hash-verified NPY arrays avoid another heavy dependency while preserving scan-independent votes.
- 2026-10-13: Camera visibility is tested in the original camera projection. Spherical radial depth is only compared to spherical radial depth; perspective axis depth is used for view frustum projection, never mixed with the radial z-buffer.
- 2026-10-13: LAS 1.4 R16 standard codes include 19–22; 8, 12 and 23–63 are reserved. The proposed blanket 19–63 reserved range was refuted against ASPRS's tagged specification.
- 2026-10-13: Keep the tested CUDA dependency resolution in requirements.lock after querying current PyPI releases. Older transitive versions (filelock, fsspec, MarkupSafe, mpmath, networkx, setuptools, pip) are retained to reproduce the proven GPU run; latest version evidence is saved in .cache/dependency_versions_full.json. ftfy 6.1.1 is required exactly by upstream SAM; triton-windows 3.6 is paired with torch 2.11.
- 2026-10-13: Automatic launch supersedes the original configuration-before-start gate. Class/settings tabs remain available for a subsequent run; a running job's parameters are immutable. Scene overrides can be set by double-clicking a scan's scene cell after the job stops.
- 2026-10-13: Neighborhood cleanup remains disabled; no automatic noise filter or class-152 assignment is introduced. Voxel size remains a tuning value, not a proven surface-density recommendation.
- 2026-10-13: Default output has no independent GUI chooser, per owner request. Explicit CLI configurations can still select an output directory. A same-second GUI start receives a unique suffix.
- 2026-10-13: Reflection E3 is not used: the supplied data has no return structure and intensity evidence is not validated. Plane fitting is conservative; uncertain candidates retain their semantic class as mirror_state 4.
