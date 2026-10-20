"""Detection of static TLS and mobile/kinematic E57 input layouts.

The metadata rules are deliberately independent of point sampling.  The
pseudo-setup rule is added later, after the bounded on-disk scan store exists.
"""

import logging

import numpy as np


LOG = logging.getLogger(__name__)


def _translation(item):
    try:
        return np.array([item[axis] for axis in "xyz"], np.float64)
    except (KeyError, TypeError, ValueError):
        return None


def _image_positions(images):
    result = []
    for image in images:
        if image.get("pose_status") != "valid":
            continue
        position = _translation(image.get("metadata", {}).get("pose", {}).get("translation", {}))
        if position is not None and np.isfinite(position).all():
            result.append((image, position))
    return result


def _spatially_separated_count(positions, distance=1.5):
    """Return a deterministic greedy count of positions separated by distance."""
    selected = []
    for position in positions:
        if all(np.linalg.norm(position-other) > distance for other in selected):
            selected.append(position)
    return len(selected)


def _bounds(scan):
    metadata = scan.get("metadata", {})
    value = metadata.get("cartesianBounds")
    if not isinstance(value, dict):
        return None
    try:
        low = np.array([value[f"cartesian{axis.upper()}Minimum"] for axis in "xyz"], np.float64)
        high = np.array([value[f"cartesian{axis.upper()}Maximum"] for axis in "xyz"], np.float64)
    except (KeyError, TypeError, ValueError):
        try:
            low = np.asarray(value["min"], np.float64)
            high = np.asarray(value["max"], np.float64)
        except (KeyError, TypeError, ValueError):
            return None
    return (low, high) if low.shape == (3,) and high.shape == (3,) and np.isfinite(np.r_[low, high]).all() else None


def _identity_pose(pose):
    if pose is None:
        return True
    try:
        rotation = np.array([pose["rotation"][axis] for axis in ("w", "x", "y", "z")], np.float64)
        translation = _translation(pose["translation"])
    except (KeyError, TypeError, ValueError):
        return False
    return translation is not None and np.allclose(rotation, [1., 0., 0., 0.], atol=1e-6) and np.allclose(translation, 0., atol=1e-6)


def _rule(identifier, value, threshold, hit, detail):
    return {"id": identifier, "value": value, "threshold": threshold, "hit": bool(hit), "detail": detail}


def detect_input_mode(inspection, config, points_check=None):
    """Classify an inspection as ``static`` or ``slam`` and retain all evidence.

    ``points_check`` is optional and receives the inspection.  It returns a
    pseudo-setup rule value, normally from :func:`pseudo_setup_points_check`.
    It is deliberately invoked only after the scan store has been built.
    """
    requested = (config or {}).get("input_mode", "auto")
    if requested not in ("auto", "static", "slam"):
        raise ValueError("Invalid input mode")
    scans = inspection.get("scans", [])
    images = inspection.get("images", [])
    positioned = _image_positions(images)
    valid_images = len(positioned)
    scan_count = len(scans)
    ratio = valid_images/max(1, scan_count)
    rules = [_rule("a", {"valid_images": valid_images, "data3d_blocks": scan_count, "ratio": ratio}, 4,
                   ratio > 4, f"{valid_images} Bilder mit gültiger Pose für {scan_count} Datenblöcke (Verhältnis {ratio:.2f})")]

    grouped = {}
    missing_association = []
    empty_association = []
    scan_by_index = {item.get("index"): item for item in scans}
    for image, position in positioned:
        scan_index = image.get("scan_index")
        scan = scan_by_index.get(scan_index)
        if image.get("association_status") != "explicit_guid" or scan_index is None:
            missing_association.append(image.get("index"))
        elif not scan or scan.get("record_count", 0) <= 0:
            empty_association.append(image.get("index"))
        else:
            grouped.setdefault(scan_index, []).append(position)
    separated = {str(index): _spatially_separated_count(points) for index, points in grouped.items()}
    shared = {index: count for index, count in separated.items() if count > 4}
    affected_images = [*missing_association, *empty_association]
    affected_image_share = len(affected_images)/max(1, valid_images)
    blocks_with_images = sorted(grouped)
    separated_blocks = [int(index) for index in shared]
    separated_block_share = len(separated_blocks)/max(1, len(blocks_with_images))
    b_hit = affected_image_share >= .50 or separated_block_share >= .50
    rules.append(_rule("b", {"without_association": len(missing_association), "without_points": len(empty_association),
                              "affected_images": affected_images, "affected_image_share": affected_image_share,
                              "separated_images_by_block": separated, "separated_blocks": separated_blocks,
                              "blocks_with_images": blocks_with_images, "separated_block_share": separated_block_share},
                       {"affected_image_share": .50, "separated_block_share": .50}, b_hit,
                       f"{len(missing_association)} Bilder ohne Zuordnung, {len(empty_association)} ohne zugehörige Punkte "
                       f"({affected_image_share:.0%}); räumlich getrennte Bilder je Block: {separated or 'keine'} "
                       f"({separated_block_share:.0%} der Blöcke mit Bildern)"))

    c_matches = []
    for scan in scans:
        box = _bounds(scan)
        if not box or not _identity_pose(scan.get("metadata", {}).get("pose")):
            continue
        low, high = box
        span = float(np.max(high-low))
        inside = sum(bool(np.all(position >= low) and np.all(position <= high)) for _, position in positioned)
        if span > 30 and inside >= 2:
            c_matches.append({"scan": scan.get("index"), "span_m": span, "image_poses_inside": inside})
    rules.append(_rule("c", c_matches, {"span_m": 30, "image_poses_inside": 2}, bool(c_matches),
                       "Identity-Blockpose mit grosser Ausdehnung und Kameraposen innerhalb der Blockgrenzen"))

    without_origin = [scan.get("index") for scan in scans if scan.get("metadata", {}).get("pose") is None]
    without_origin_share = len(without_origin)/max(1, scan_count)
    rules.append(_rule("e", {"without_scanner_pose": without_origin, "without_scanner_pose_share": without_origin_share,
                              "scan_count": scan_count}, .50, without_origin_share >= .50,
                       f"{len(without_origin)} von {scan_count} Datenblöcken ohne Scannerposition ({without_origin_share:.0%})"))

    if points_check is not None:
        checked = points_check(inspection)
        if not isinstance(checked, dict):
            raise ValueError("Pseudo-setup point check must return a dictionary")
        value = checked.get("value", checked)
        hit = bool(checked.get("hit", False))
        detail = checked.get("detail", "Tiefentest vom Block-Ursprung")
        rule = _rule("d", value, checked.get("threshold", .60), hit, detail)
        rule.update({key: checked[key] for key in ("flagged", "flagged_share", "without_origin") if key in checked})
        rules.append(rule)

    detected_slam = any(rule["hit"] for rule in rules)
    associated = sum(1 for image, _ in positioned if image.get("association_status") == "explicit_guid")
    mixed = detected_slam and associated > 0 and associated < valid_images
    warnings = []
    rule_b = next(rule for rule in rules if rule["id"] == "b")
    if rule_b["value"]["affected_images"] and not rule_b["hit"]:
        warnings.append("Einzelne Bilder ohne sichere Zuordnung oder Punkte erkannt; statische Verarbeitung bleibt beibehalten.")
    if rule_b["value"]["separated_blocks"] and not rule_b["hit"]:
        warnings.append("Einzelne Datenblöcke enthalten mehr als vier räumlich getrennte Bilder; statische Verarbeitung bleibt beibehalten.")
    rule_e = next(rule for rule in rules if rule["id"] == "e")
    if rule_e["value"]["without_scanner_pose"] and not rule_e["hit"]:
        scans_text = ", ".join(map(str, rule_e["value"]["without_scanner_pose"]))
        warnings.append(f"Scan {scans_text}: Scannerposition fehlt; statische Verarbeitung bleibt beibehalten und der Block wird unklassifiziert ausgegeben.")
    if points_check is not None and isinstance(checked, dict) and checked.get("flagged") and not checked.get("hit"):
        for item in checked["flagged"]:
            warnings.append(f"Scan {item['scan']}: Tiefentest auffällig ({item['visible_share']:.0%} sichtbar), vermutlich Teilscan – statische Verarbeitung bleibt beibehalten.")
    if mixed:
        warnings.append("Gemischte Hinweise auf statische und kinematische Aufnahme erkannt; kinematische Verarbeitung gewählt.")
    mode = requested if requested != "auto" else ("slam" if detected_slam else "static")
    if requested != "auto":
        warnings.append(f"Aufnahmeart manuell auf {'Kinematisch (SLAM)' if requested == 'slam' else 'Statisch (TLS)'} gesetzt.")
    result = {"mode": mode, "requested": requested, "rules": rules, "mixed": mixed, "warnings": warnings}
    for rule in rules:
        LOG.info("Input mode rule %s: value=%s threshold=%s hit=%s; %s", rule["id"], rule["value"],
                 rule["threshold"], rule["hit"], rule["detail"])
    for warning in warnings:
        LOG.warning("Input mode: %s", warning)
    LOG.info("Input mode selected: %s (requested=%s, mixed=%s)", mode, requested, mixed)
    return result


def pseudo_setup_points_check(stores, inspection=None, sample_limit=200_000, width=1024, config=None, checkpoint=lambda: None):
    """Use a sample-density-matched origin depth test to find pseudo-setups.

    Rule d deliberately uses its own one-pixel footprint.  It must
    measure radial duplicate layers consistently as the sample limit changes;
    the SLAM view setting ``slam_splat_max_px`` is not part of this check.
    """
    # Imported here to keep inspector -> input_mode -> camera metadata imports acyclic.
    from e57cls.geometry.cameras import has_scan_origin, scan_pose
    from e57cls.slam.visibility import origin_depth_width, visibility_from_origin
    results = []
    without_origin = []
    for store in stores:
        metadata = store.meta.get("metadata", {})
        if not has_scan_origin(metadata):
            if metadata.get("pose") is not None:
                scan_pose(metadata)
            without_origin.append(store.meta.get("scan_index"))
            continue
        count = int(store.meta.get("valid_count", 0))
        if not count:
            continue
        stride = max(1, int(np.ceil(count/sample_limit)))
        samples = []
        for _, xyz, valid in store.full_chunks():
            checkpoint()
            ids = np.flatnonzero(valid)
            if len(ids):
                samples.append(xyz[ids[::stride]])
        if not samples:
            continue
        points = np.concatenate(samples)
        if len(points) > sample_limit:
            points = points[:sample_limit]
        pose = scan_pose(metadata)
        world = pose.to_world(points)
        valid = np.isfinite(world).all(axis=1)
        world = world[valid]
        if not len(world):
            continue
        depth_width = origin_depth_width(int(valid.sum()), width)
        visible = visibility_from_origin(world, origin=pose.translation, width=width,
                                         tolerance_abs=(config or {}).get("slam_depth_tol_abs", .02),
                                         tolerance_rel=(config or {}).get("slam_depth_tol_rel", .01),
                                         splat_max_px=1,
                                         device=(config or {}).get("gpu_device", "cuda:0"))
        share = float(visible.mean())
        results.append({"scan": store.meta.get("scan_index"), "sampled_points": len(world),
                        "depth_width": depth_width, "visible_share": share})
    if without_origin and not results:
        return {"value": "nicht anwendbar", "without_origin": without_origin, "hit": False,
                "detail": "Nicht anwendbar: Datenblock ohne Scannerposition."}
    flagged = [item for item in results if item["visible_share"] < .60]
    flagged_share = len(flagged)/max(1, len(results))
    threshold = {"visible_share": .60, "flagged_share": .50}
    hit = flagged_share >= threshold["flagged_share"]
    return {"value": results, "without_origin": without_origin, "flagged": flagged, "flagged_share": flagged_share,
            "threshold": threshold, "hit": hit,
            "detail": "Tiefentest vom Block-Ursprung: winkelfensterbewusste Tiefenkarte mit ungefähr 7 Pixeln je Punkt und 1-Pixel-Footprint; auffällig bei unter 60 % sichtbaren Punkten in mindestens 50 % der Blöcke"}
