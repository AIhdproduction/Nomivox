# Existing msam3 reuse assessment

Inspected source in `../metashape-sam3-classifier/src/msam3` before implementing M0.
No import of msam3 or runtime dependency on Metashape has been introduced.

| Module | Assessment | Reason / required change |
|---|---|---|
| __init__, __main__ | not reusable | Package identity and CLI routing change. |
| cache | reusable with changes | Keep atomic-write/hash pattern; use content hashes and per-class raw confidence cache, not image mtime and uint8 confidence PNGs. |
| camera_model | not reusable | Metashape distortion, positive-Z camera and chunk transforms do not implement E57 panorama models. |
| cli | not reusable | New inspect/store/geometry/run commands and E57 arguments. |
| config | reusable with changes | Validation concepts useful; fixed JSON schemas, German names, scenes, reserved codes, unknown-key preservation differ. |
| driver | reusable with changes | Stage-marker/job-lock concepts useful; remove Metashape stages and make work units scan-aware. |
| gui_options | not reusable | Tk settings and Metashape session contracts cannot satisfy four-tab PySide6 GUI. |
| gui_progress | not reusable | Log polling and Metashape stages replaced by versioned JSONL worker events. |
| gui_worker | not reusable | Metashape session/export contracts throughout. |
| metashape_runner | not reusable | Explicitly invokes Metashape executable. |
| pipeline | reusable with changes | Tiling/voting/transfer sequence useful; replace data, camera, validation and observation aggregation. |
| report | reusable with changes | Counts/timing concepts retained; add per-scan invalids, scenes, mirror states, untranslated attributes and German HTML. |
| validation | not reusable | Requires Metashape tie points and export transforms. E57 needs independent projection probes and edge diagnostics. |
| segmentation/__init__ | not reusable | Package boilerplate only. |
| segmentation/sam3_backend | reusable with changes | Local checkpoint and raw batched grounding contract useful. Verify current upstream revision, per-class thresholds and OOM retries. |
| segmentation/segearth_fusion | reusable as is | Pure probability fusion with explicit presence/instance confidence. Revalidate chosen SAM output contract. |
| segmentation/tiling | reusable with changes | Reuse overlapping tile placement/max merging; preserve per-class raw confidences and add retry/cache units. |
| segmentation/mask_filter | reusable as is | Confidence and boundary erosion are independent of Metashape; apply at voting time. |
| pointcloud/__init__ | not reusable | Package boilerplate only. |
| pointcloud/las_io | reusable with changes | Streaming writer/legend patterns; E57 attributes, formats 6/7, extra bytes and scan IDs need new inputs. |
| pointcloud/spatial | reusable with changes | Disk partition concepts useful; current fixed buckets can grow with total count, float32 store and nearest-neighbor transfer need revision. |
| pointcloud/visibility | reusable with changes | Min-depth buffer concept works; camera center and max(abs, rel*depth) tolerance differ; no Metashape depth maps. |
| pointcloud/voting | reusable with changes | Weighted decision useful; current views counter counts repeated additions, so deduplicate scan observations first. Remove ground/vegetation assists. |
| pointcloud/smoothing | reusable with changes | Optional only, off by default, bounded spatial working set required. |
| pointcloud/postprocess | reusable with changes | Optional generic outlier routines only; reject automatic terrain/vegetation assumptions and disable beyond V1 scope. |

Metashape scripts and `start_in_metashape.py`: not reusable; their host application
is explicitly outside the new runtime. Tests may provide ideas, but E57 fixtures,
projection ground truth and scan provenance must be tested independently.
