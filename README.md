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
- Use 36 object classes with 268 English search prompts, and edit or add classes.
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
progress per view where a phase processes views. The saved profile
(`.cache/timing_profile.json`) records a `unit` for each stage; profiles with a
different unit, and implausible values above ten times the built-in default, are
ignored. The defaults are based on measured runs. Until enough information is
available, the interface displays **Restzeit wird berechnet …** (Calculating
remaining time). Cache hits, different image contents, and GPU load can affect
the estimate.

The status line also shows **Gesamtrestzeit**: a rough estimate for the entire
remaining run, including the current and subsequent phases. It combines the
per-view estimates with bounded fixed estimates for remaining phases and updates
continuously. Until enough progress is available, it displays
**Gesamtrestzeit wird berechnet …**. Hover over the status line for details.

## How classification works

1. **Inspect and read the E57.** Scan positions, images, poses, and attributes are
   checked. Point data is read into a local cache in bounded chunks.
2. **Prepare images.** Panoramas are rendered into perspective views for SAM.
   Scan and camera poses are handled separately. Spherical, cylindrical, and
   pinhole camera models are implemented; the real test dataset contains panoramas.
   The **Hohe Genauigkeit** profile additionally renders eight 50° views at −10°
   pitch for distant surfaces just below the horizon. This increases SAM work by
   approximately 45% and does not affect the fast or balanced profiles.
3. **Analyze mirrors and glass.** Text prompts produce masks. Planes are estimated
   from their outer edges and neighboring walls; the depth rule determines class 68.
4. **Determine the scene.** For each scan, SAM sky/ceiling evidence, laser
   returns, and horizontal openness from geometry alone are combined. Canopies
   and arcades resolve as **Aussen (überdacht)** (covered outdoor) and use
   outdoor classes. In scan scene resolution, a preferred scan-name hint
   (`scene_name_hint_mode: "prefer"`) ranks strictly above covered-outdoor data
   evidence; scan names can provide a hint, for example Riegl folders such as
   `01_Aussen` or floor names such as `UG1` and `1OG`, and conflicts with data
   are reported as warnings. Unresolved scans use nearby scans at a similar
   height when available; otherwise all scene classes are requested for them.
   Independently of the scan decision, each voxel takes the scene of its closest
   observing scanner when that scanner is clearly closer (`scene_context_margin`).
   A 2D building outline raster provides an additional scene signal: it applies
   only where a cell is enclosed AND covered (`scene_outline_min_cover: 0.5`),
   while voxels in the roof band (`scene_outline_roof_band: 0.4`) retain their
   scanner context. Unenclosed exterior cells are never forced to outdoor unless
   the project contains at least one outdoor scan. Facade surfaces are resolved
   per structural plane: the exterior skin becomes building, while the interior
   skin becomes wall. Door and window zones remain ambiguous and retain their
   vote result. A consistency pass, enabled by `scene_consistency`, remaps
   scene-specific classes to their counterpart (floor to ground/sidewalk/road;
   ceiling/wall/door to building; building to wall; ground/road/sidewalk to
   floor), then the best suitable alternative, or class 1. Classes marked `both`
   remain unchanged. Note that `reclassify` imports the resolved per-scan scene
   classification directly from the source run. Manual settings take precedence.
5. **Segment objects.** SAM processes the English prompts for each class. Ordinary
   class segmentation combines prompt confidence maps by taking their maximum.
   Overlapping image tiles, confidence thresholds, and a mask-boundary margin
   control the transfer of labels.
6. **Project onto points and combine votes.** Projection and depth checks restrict
   label transfer to visible points. Spatial voxels collect class evidence;
   multiple views from the same scan do not count as independent scan observations.
   Post-processing between voting and export is described in the section below.
7. **Export original points.** Final classes are transferred to the valid original points, not just voxel representatives. Direct class transfer expands from 3 mm per metre of scanner distance up to 5 cm (`class_transfer_radius_max`), and optional export fills populate unclassified points inside measured tree crowns and ground DTM bands. The exported LAS/LAZ file is read back to verify point counts, class codes, sampled coordinates, and scan membership.

### Post-processing after voting

The default voxel size is **2 cm** (`voxel_size: 0.02`). Between voxel voting and final point export, spatial voxel decisions undergo a structured, twelve-stage post-processing sequence (matching the order in `_postprocess`):

1. **Voxel decision** (`min_observations: 1`, `min_winning_share: 0.50`): Voxel-level votes are combined across scans. A class wins if its vote share reaches `min_winning_share` with at least `min_observations`. Stairs with at least `stairs_min_confidence` (0.65) and 25% vote share (`stairs_priority_min_share: 0.25`) take precedence over competing floor and ground observations (`stairs_priority_over: ['floor', 'ground']`). Vote families (`vote_families: [['ground', 'floor']]`) pool related classes during initial decision. Disable stair priority with `stairs_priority: false`.
2. **Surface ties** (`surface_tie_break: true`): Resolves close votes between wall, ceiling, and floor using local surface normals when the winning share is between 0.50 and `surface_tie_share` (0.55). Surfaces steeper than 60° resolve to wall, flatter than 30° to floor or ceiling depending on elevation relative to scanner height, and intermediate slopes to ceiling. Disable with `surface_tie_break: false`.
3. **Building enclosure (ground/floor)** (`building_enclosure: true`): Separates exterior ground from interior floors using a 2D building outline raster derived from building and wall evidence. A flood-fill from outside through openings isolates exterior terrain from enclosed rooms. Disable with `building_enclosure: false`.
4. **Segment voting** (`segment_vote: true`): Groups voxels into geometry-connected 10 cm feature-cell segments. Planar cells join when normals differ by at most 15° (`segment_normal_angle: 15.0`) and plane distance is at most 3 cm (`segment_plane_distance: 0.03`). If a segment has a winning class with at least 60% share (`segment_min_share: 0.6`), unclassified and weakly decided voxels adopt the segment class. Protected classes (stairs, doors, windows, electrical equipment, traffic signs, manhole covers, and class 68) are never overwritten. Disable with `segment_vote: false`.
5. **Geometry rules** (`geometry_rules: true`): Refines terrain and vegetation using 10 cm feature cells, evaluating local scatter, height distribution, and adjacency. Generates a compact terrain elevation grid (`terrain.npz`) for subsequent terrain fill. Disable with `geometry_rules: false` (or individually with `geometry_rules_outdoor_vegetation: false` and `geometry_rules_ground_floor: false`).
6. **Stairs check** (`stairs_geometry_check: true`): Verifies stair candidates against geometric step criteria (minimum total rise 25 cm, at least 3 distinct step levels with 10–23 cm rise and 20–45 cm tread, non-planar step structure). Voxel clusters that form flat planes or fail step geometry revert to competing floor or ground classes. Disable with `stairs_geometry_check: false`.
7. **Scene consistency** (`scene_consistency: true`): Enforces indoor/outdoor consistency per voxel based on scanner distance margins (`scene_context_margin: 0.8`) and building outline evidence (`scene_outline: true`). Remaps misplaced scene-specific classes to their opposite-scene counterparts or configured fallbacks, leaving classes valid in both scenes unchanged. Disable with `scene_consistency: false`.
8. **Plane extraction** (RANSAC, once): Detects major structural planes (walls, floors, ceilings) across the cloud and saves them to `planes.npz`. This RANSAC extraction runs once and is shared by geometry fill and shape rules without repeated plane fitting. Automatically executed when either `geometry_fill: true` or `shape_rules: true` is enabled.
9. **Shape rules** (`shape_rules: true`): Evaluates object candidates on a fine 4 cm grid against the measured RANSAC structure layer with its noise band envelope. Applies dedicated indoor, technical, and outdoor object rules based on dimensions, height above floor, wall distance, and mantle planarity. Disable with `shape_rules: false`.
10. **Ground DTM band** (`ground_dtm: true`): Constructs a Delaunay TIN digital terrain model (`dtm.npz`) from the lowest exterior ground points outside buildings. Ground points deviating more than 3 cm (`ground_dtm_band: 0.03`) from the DTM take their second-best SAM class (cached in `ground_alternatives.npz`), while unclassified points within 3 cm outside buildings become ground. Disable with `ground_dtm: false`.
11. **Small-component cleanup** (`component_cleanup: true`): Removes isolated voxel clusters below minimum size thresholds (12 voxels or 4 feature cells) to eliminate floating noise artifacts. Thin cables require a minimum span of 20 cm, and electrical equipment requires at least 6 cm. Disable with `component_cleanup: false`.
12. **Vegetation base** (`vegetation_base_to_ground: true`): Identifies low-vegetation voxels immediately adjacent to terrain within a 4 cm vertical band (`vegetation_ground_band: 0.04`). Converts these ground-contact vegetation bases into ground (class 2) to eliminate false shrub stems right at the surface. Disable with `vegetation_base_to_ground: false`.

### Shape rules

Shape rules operate after segment voting. They keep the measured RANSAC structure layer with a statistical noise band envelope (`shape_layer_sigma_factor: 2.5`, capped at `shape_layer_max_band: 0.08` to accommodate noisy SLAM scans) and apply rules on the original **4 cm** object grid (`shape_object_cell: 0.04`). Local floor and ceiling levels are determined dynamically per room zone, accommodating sloped roofs (slopes steeper than 60° count as walls, flatter slopes as ceilings). Large objects exceeding 6 m in extent (`shape_object_max_size: 6.0`) are skipped to protect whole facades or continuous surfaces. Rules only select classes for which SAM has voted in that segment or object (`shape_min_vote_share: 0.10`), with explicit exceptions confirmed by Domenic: furniture vs. household clutter, wall pilasters, flush wall items, and carpet.

- **Indoor objects:**
  - *Columns vs. pilasters:* Free-standing vertical structures with a footprint smaller than 1.5 × 1.5 m (`shape_column_max_size: 1.5`) spanning at least 70% of local room height (`shape_column_min_height_share: 0.7`) become columns (108). Vertical structures touching an adjacent wall become wall pilasters and are classified as walls (100).
  - *Built-in cabinets:* Objects standing on the floor with front surfaces 0.3–0.7 m in front of a wall (`shape_cabinet_offset: [0.3, 0.7]`) classify as furniture (117), even when built up to the ceiling.
  - *Beams:* Linear elements (length ≥ 1.5 m, width ≤ 0.6 m) within 0.5 m below ceilings or roof slopes classify as beams (109), explicitly including sloped rafters in attics.
  - *Carpet:* Thin floor coverings (≤ 3 cm above floor level) classify as floor (102).
  - *Wall items:* Elements mounted on walls that are flush within 3 cm (`shape_wall_flush_max: 0.03`) merge into wall (100); protruding items (> 3 cm) become unclassified (1), while deeper wall cabinets (depth ≥ 0.25 m, height ≥ 0.4 m) classify as furniture (117). Electrical wall fixtures (switches and outlets) are exempt from the 3 cm wall rule.
  - *Furniture vs. household clutter:* Portable items smaller than 0.5 m (`shape_clutter_max_size: 0.5`) resting on furniture or floors classify as loose objects and household clutter (76); larger items classify as furniture (117).
- **Technical installations:**
  - *Pipes:* Elongated cylinders (length ≥ 0.5 m, diameter 3–60 cm, non-planar mantle < 50%) classify as pipe (111). Round ventilation ducts and sprinkler heads/pipes also classify as pipe.
  - *HVAC:* Rectangular ventilation ducts (cross-section ≥ 15 cm, planar mantle ≥ 50%) and wall-mounted radiators (5–30 cm off wall, 5–25 cm above floor, height ≥ 30 cm, width ≥ 30 cm) classify as HVAC (114).
  - *Cables:* Thin wires (cross-section < 3 cm) and open cable trays (width 10–60 cm, height < 15 cm mounted along walls or ceilings) classify as cable (112).
  - *Ceiling equipment:* Fixtures suspended within 1.0 m (`shape_ceiling_equipment_max_drop: 1.0`) below ceilings, including ceiling lamps and smoke detectors, classify as ceiling equipment (116).
  - *Sanitary fixtures:* Sinks, toilets, bathtubs, bidets, urinals, glass shower partitions, and vanity under-cabinets classify as sanitary fixtures (63).
  - *Railings:* Barriers with handrails at 0.85–1.1 m height, explicitly including glass railings, classify as railing (107). Steel structures and scaffolding remain steel structures (110).
- **Outdoor objects:**
  - *Curbs:* Narrow steps (5–20 cm elevation step, width ≤ 0.35 m) between roads and sidewalks classify as curbs (74).
  - *Outdoor walls:* Free-standing garden, retaining, and boundary walls not part of a building classify as outdoor walls (78); building plinths stay with building (6).
  - *Utility poles:* Slender vertical structures (height ≥ 3 m, cross-section ≤ 0.5 m) classify as utility poles (84).
  - *Traffic signs:* Sign plates and their supporting posts together classify as traffic signs (81).
  - *Fences:* Linear outdoor barriers (height 0.8–2.0 m, length ≥ 2.0 m, thickness ≤ 0.3 m) classify as fences (77).
  - *Vehicles:* Cars, vans, trucks, as well as bicycles, motorcycles, scooters, and trailers classify as vehicles (91).
  - *Overhead lines:* Thin elevated spans (length ≥ 3 m, height ≥ 4 m above lowest ground) without immediate vertical supports classify as power lines (14).
  - *Canopies and carports:* Covered outdoor roof structures classify as building (6), unless identified as a visible steel structure (110).
  - *Road vs. sidewalk:* Decided strictly by SAM 3 votes and segment voting; shape rules perform no geometric reinterpretation between roads (11) and sidewalks (73).

### Trees

Tree crowns are reconstructed using a geometric lollipop model:
- **Stem detection:** Tree stems are identified from classification decisions (tree_trunk 95), vote evidence (trunk weight ≥ 30% of total votes), or geometric shape fitting (vertical cylinders with diameter 0.1–1.5 m and height ≥ 0.8 m below tall vegetation evidence).
- **Two-slice crown radius:** The crown radius is measured dynamically per tree in two horizontal slices: one directly above the top of the trunk and a second slice 1.0 m higher. The 90th percentile radial distance of points in each slice is measured, the larger radius wins, and the value is clamped between `tree_crown_radius_min` (default 1.0 m) and `tree_crown_radius_max` (default 8.0 m).
- **Crown voxel recoloring:** All voxels above the trunk top within the vertical crown cylinder up to `tree_crown_max_height` (default 25 m) are assigned to tall vegetation (class 5). Large RANSAC planes (building facades, roofs, ground), straight linear structures (utility poles, overhead lines), and protected classes (stairs 105, reflection 68) are never overwritten.
- **Tree trunks protected:** Tree trunks (class 95) are immutable and are never overwritten by crown expansion or fill rules.
- **Crown interior fill (`tree_crown_fill: true`):** Identified crown cylinders are saved to `tree_crowns.npz`. During point export, previously unclassified points located inside the measured crown volume adopt class 5 without overwriting any already classified point.

### Ground from a TIN DTM

Exterior ground points are validated and completed against a Triangulated Irregular Network Digital Terrain Model (TIN DTM):
- **DTM generation:** The lowest exterior ground-family points (ground 2, road 11, sidewalk 73, curb 74, manhole 75) outside building outlines and excluding reflection points (class 68) are aggregated into 2D cells (`dtm_cell_size: 0.5` m) using a robust 10th-percentile height, filtered by slope (`terrain_max_slope: 0.75`), and triangulated via 2D Delaunay triangulation into `dtm.npz`.
- **Height band check (`ground_dtm_band: 0.03`, 3 cm):**
  - Ground points (class 2) deviating more than 3 cm from the DTM surface are demoted to their second-best SAM vote class (e.g. low vegetation in bushes or indoor floor visible through windows); if no other vote exists, they become unclassified (class 1). Runner-up classes are cached in `ground_alternatives.npz`.
  - Unclassified points outside buildings located within 3 cm of the DTM surface are promoted to ground (class 2).
- **Exempt classes:** Low vegetation (lawn 3), paved surfaces (roads 11, sidewalks 73, curbs 74, manholes 75), stairs (105), tree trunks (95), and reflections (68) are protected and untouched by DTM adjustments.
- **Voxel and point application:** Applied to spatial voxels during post-processing and re-evaluated per point during LAS/LAZ export. When enabled (`ground_dtm: true`), it replaces the legacy 15 cm terrain fill.

### Post-processing settings

The following table summarizes the primary post-processing parameters and their default values (from `e57cls/pipeline/config.py`):

| Setting | Default | Description |
|---|---|---|
| `segment_vote` | `true` | Enable geometry-connected segment voting |
| `segment_min_share` | `0.6` | Minimum segment vote share to reclassify unclassified or weak voxels |
| `segment_protected_classes` | `["stairs", "door", "window", "electrical", "traffic_sign", "manhole"]` | Classes never overwritten by segment voting |
| `shape_rules` | `true` | Enable indoor, technical, and outdoor shape rules |
| `shape_min_vote_share` | `0.10` | Minimum SAM vote share required for an object to adopt a shape-rule class |
| `shape_object_cell` | `0.04` | Grid resolution (4 cm) for clustering object candidate components |
| `shape_column_max_size` | `1.5` | Maximum horizontal cross-section (1.5 m) for free-standing columns |
| `shape_wall_flush_max` | `0.03` | Maximum protrusion (3 cm) for flush wall-mounted items |
| `shape_layer_max_band` | `0.08` | Upper bound (8 cm) for the RANSAC structure layer noise envelope |
| `shape_object_max_size` | `6.0` | Maximum extent (6 m) above which objects are skipped by shape rules |
| `scene_outline` | `true` | Use 2D building outline as an additional per-voxel scene context signal |
| `scene_outline_min_cover` | `0.5` | Minimum overhead structural cover (0.5 m) required for indoor enclosure |
| `scene_outline_roof_band` | `0.4` | Height band (0.4 m) around roof structures keeping scanner-derived context |
| `ground_dtm` | `true` | Enable TIN DTM generation and elevation band validation for ground points |
| `ground_dtm_band` | `0.03` | Elevation tolerance band (3 cm) around the DTM surface |
| `dtm_cell_size` | `0.5` | 2D cell grid size (0.5 m) for aggregating DTM ground candidate heights |
| `tree_crown_radius_min` | `1.0` | Minimum allowed tree crown radius (1.0 m) |
| `tree_crown_radius_max` | `8.0` | Maximum allowed tree crown radius (8.0 m) |
| `tree_crown_fill` | `true` | Fill unclassified export points within detected tree crown volumes |

All of these parameters can be adjusted via reclassify overrides using their respective configuration prefixes (`segment_`, `shape_`, `scene_`, `ground_dtm`, `dtm_`, `tree_crown_`) without requiring a new SAM segmentation run.

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

Kitchen and household appliances (118) covers stoves, cooktops, ovens,
dishwashers, range hoods, washing machines, clothes dryers, and microwave ovens;
refrigerators stay with furniture (117); televisions are not part of class 118.

Outdoor walls (78) covers free-standing garden, retaining and boundary walls;
building plinths stay with buildings (6). Vehicles (91) include bicycles,
motorcycles, scooters and trailers.

Additional prompts broaden the search but do not guarantee higher accuracy and
can increase processing time. Prefer specific terms; duplicate prompts in
competing classes are flagged during preset validation. The `auxiliary` section
contains helper prompts for sky, mirrors, and window glass. **Unclassified (1)**
is the fallback class, while **Reflection (68)** follows the mask/geometry rule
above. Neither is an ordinary text-searched object class.

Built-in classes keep fixed codes - their code cell is locked in the class table.
Classes added in the interface receive a free code from 23 to 60; codes of
built-in classes (including 1 and 68) are reserved even when the built-in class
is disabled or removed, and preset validation rejects a custom class or code
mapping that uses one.

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
| 78 | Outdoor walls | Outdoor |
| 81 | Traffic signs | Outdoor |
| 84 | Utility poles and streetlights | Outdoor |
| 91 | Vehicles | Outdoor |
| 95 | Tree trunks | Outdoor |
| 100 | Walls | Indoor |
| 101 | Ceilings | Indoor |
| 102 | Floors | Indoor |
| 103 | Doors | Indoor |
| 104 | Windows | Indoor and outdoor |
| 105 | Stairs | Indoor and outdoor |
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
| 118 | Kitchen and household appliances | Indoor |

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
    └── <scan name>_<YYYYMMDD_HHMMSS>/
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

During execution and reclassification, durable intermediate post-processing artifacts
are created in the run or reclassify cache (`.cache/votes/<identity>/` or
`<run>/reclassify_cache/`):

- `planes.npz`: Extracted RANSAC structural planes (walls, floors, ceilings) reused across geometry fill and shape rules.
- `terrain.npz`: 2D elevation grid of terrain features created by geometry rules for raster-based terrain fill.
- `dtm.npz`: Triangulated Irregular Network Digital Terrain Model (TIN DTM) built from lowest exterior ground points.
- `ground_alternatives.npz`: Second-best SAM candidate classes and confidences for ground points deviating off the DTM surface, applied during point export.
- `tree_crowns.npz`: Geometric parameters (x, y, radius, z_min, z_max) of recognized tree crowns used to fill unclassified points inside crowns at export.

The input E57 is not modified. LAS/LAZ 1.4 contains valid original points,
available RGB colors, classes, and the scan ID as `Point Source ID`. The extra
fields `class_conf` and `mirror_state` store the class voting share and reflection
status. Class colors are provided in the legend; they do not replace original
RGB colors. Points without an accepted class assignment remain in class 1.

Invalid points, failed scans, and attributes not transferred are logged. Not all
E57-specific attributes are carried into LAS. Coordinates use 1 mm LAS scaling
by default. GPS times and return information without a verified basis are not
invented. Export to a classified E57 is not currently implemented.

`report.json` additionally contains `scene_resolution`, `scene_consistency`,
and `scene_audit`. `report.html` lists each scan's name, resolved scene
(including **Aussen (überdacht)**), source, and any hint or resolution notice.

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
.cache\venv\Scripts\python.exe -m e57cls quality "D:\Data\Project\klassifiziert\<Scanname>_..." --step 40
```

Mit `cache_cleanup: "keep_all"` kann die gespeicherte Abstimmung ohne erneute
Bildsegmentierung nachklassifiziert werden. Nur Nachbearbeitungs- und Exportwerte
in der Override-Datei sind zulässig; der ursprüngliche Vote-Cache bleibt erhalten.

```powershell
.cache\venv\Scripts\python.exe -m e57cls reclassify --from "D:\Data\Project\klassifiziert\<Scanname>_..." --out "D:\Data\Project\klassifiziert\run_tuned" --config overrides.json
```

Reclassify overrides allow adjusting post-processing parameters without repeating
the SAM segmentation run. Supported override parameters include the `segment_*`,
`shape_*`, `scene_*`, `ground_dtm`, `dtm_*`, `tree_crown_*` (including
`tree_crown_fill`), `geometry_rules`, `geometry_fill*`, `stairs_*`,
`surface_tie_*`, `component_*`, `cable_*`, `electrical_*`, and `vegetation_*`
controls.

Das folgende Prüfwerkzeug vergleicht die Szenen pro Scan mit den Scan-Namen,
getrennt für Endergebnis und reine Datenerkennung, und zeigt Kennzahlen pro
Scan. `scene_audit` zeigt den Anteil falscher Szenenklassen pro Scan. Mit
`--grid` werden Offenheits-Schwellen auf den gespeicherten Kennzahlen erprobt.
Für Nachjustieren ohne SAM muss der Ursprungslauf `cache_cleanup: "keep_all"`
verwenden.

```powershell
.cache\venv\Scripts\python.exe -m e57cls.tools.scene_eval "D:\Data\Project\klassifiziert\<Scanname>_..." [--grid]
```

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

Die automatische Erkennung entscheidet anhand des Anteils betroffener
Datenblöcke oder Bilder (mindestens 50 %). Einzelne Ausreisser, etwa ein
Teilscan, erzeugen nur eine Warnung. Der Tiefentest berücksichtigt das belegte
Winkelfenster eines Blocks. Bei statischen Läufen werden einzelne Blöcke ohne
Scannerposition unklassifiziert ausgegeben, statt den Lauf abzubrechen.

Die Verarbeitung nutzt einen gemeinsamen SAM-Durchgang, Parallelisierung,
kachelweisen Transfer, kompakte Datenspeicher, einen Score-Floor und bereinigt
den Cache nach erfolgreichem Lauf gemäss Einstellung.

## Validation and limitations

Processing has been checked with a real file containing **10 scans and around
94 million points**. Tests cover projection, point-label transfer, export, scene
selection, stair priority, mirror masks, edge/wall plane fitting, and the interface.
All ten scan positions in the verified indoor dataset are recognized as indoor.

An additional real static dataset with **157 scans** (indoor and outdoor,
around **773 million points**) has been processed. Before the indoor/outdoor
package, 140 of its 157 scans were assigned correctly. The improvement from the
indoor/outdoor package has so far been verified only with synthetic tests; an
evaluation run on this dataset is still pending.

Shape rules and the TIN DTM have been tuned on **HSLU_Trakt_4** (static TLS) and
**Loris_home** (mobile SLAM). Noisy SLAM wall surfaces dynamically widen the RANSAC
structure noise envelope up to 8 cm (`shape_layer_max_band`). Runtime of
post-processing on large datasets (~50 million voxels) currently takes tens of
minutes and is actively being optimized. Memory during LAS/LAZ export peaks around
45 GB on large projects such as HSLU (773 million points).

SAM can miss objects, produce overly broad masks, or confuse classes. The latest
prompt additions have been checked for consistency; improved recognition accuracy
has not yet been demonstrated. Real outdoor scenes are represented by the
157-scan dataset, but the current indoor/outdoor resolution has not yet been
evaluated on it. Review results in a point-cloud viewer and inspect the
diagnostic images.

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
