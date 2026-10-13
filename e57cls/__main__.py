"""Command-line entry point."""

import argparse
import logging
from pathlib import Path

from e57cls.io.inspector import write_inspection


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local E57 inspection and classification")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect metadata and bounded point samples")
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--out", required=True, type=Path)
    inspect.add_argument("--sample-points", type=int, default=100_000)
    store = commands.add_parser("store", help="Stream original records into a verified scan store")
    store.add_argument("source", type=Path)
    store.add_argument("--cache", type=Path, required=True)
    store.add_argument("--out", type=Path, required=True)
    store.add_argument("--chunk-points", type=int, default=100_000)
    geometry = commands.add_parser("geometry", help="Run independent probes and render geometry diagnostics")
    geometry.add_argument("source", type=Path)
    geometry.add_argument("--inspection", type=Path, required=True)
    geometry.add_argument("--store-report", type=Path, required=True)
    geometry.add_argument("--out", type=Path, required=True)
    geometry.add_argument("--cache", type=Path, required=True)
    geometry.add_argument("--view-size", type=int, default=768)
    segment = commands.add_parser("segment", help="Segment one image with local SAM 3")
    segment.add_argument("image", type=Path)
    segment.add_argument("--model", type=Path, required=True)
    segment.add_argument("--out", type=Path, required=True)
    segment.add_argument("--cache", type=Path, required=True)
    segment.add_argument("--prompt", action="append", required=True)
    run = commands.add_parser("run", help="Classify an E57 from a resolved JSON configuration")
    run.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from e57cls.pipeline.config import load_config
            from e57cls.pipeline.run import execute
            result = execute(load_config(args.config))
            return 130 if result.get("cancelled") else 0
        if args.command == "segment":
            import numpy as np
            from PIL import Image
            from e57cls.files import atomic_json, file_hash, new_output
            from e57cls.seg.backend import PromptClass, SAM3Backend
            from e57cls.seg.cache import MaskCache, segment_view
            output = new_output(args.out, args.image)
            with Image.open(args.image) as im:
                pixels = np.array(im.convert("RGB"))
            backend = None
            def factory():
                nonlocal backend
                if backend is None:
                    backend = SAM3Backend(args.model)
                return backend
            classes = [PromptClass(f"prompt_{i}", (p,)) for i, p in enumerate(args.prompt)]
            scores, stats = segment_view(pixels, np.ones(pixels.shape[:2], bool), classes, factory,
                                        MaskCache(args.cache), file_hash(args.image), {}, file_hash(args.model))
            for i, score in enumerate(scores):
                Image.fromarray(np.rint(score*255).astype(np.uint8)).save(output/f"confidence_{i:02d}.png")
            atomic_json(output/"segmentation_report.json", {"prompts": args.prompt, "cache": stats,
                        "sam_calls": backend.calls if backend else 0, "reductions": backend.reductions if backend else []})
            return 0
        if args.command == "geometry":
            from e57cls.geometry.diagnostics import diagnose
            report = diagnose(args.source, args.inspection, args.store_report, args.out, args.cache, args.view_size)
            print(f"All probes passed: {report['all_probes_passed']}; owner review pending")
            return 0 if report["all_probes_passed"] else 2
        if args.command == "store":
            from e57cls.files import atomic_json, new_output
            from e57cls.io.store import build_store
            output = new_output(args.out, args.source)
            def progress(scan, read, total):
                if read == total or read % 1_000_000 == 0:
                    print(f"Scan {scan}: {read}/{total}", flush=True)
            report = build_store(args.source, args.cache, args.chunk_points, progress)
            atomic_json(output / "store_report.json", report)
            print(f"Valid: {report['valid_count']}; invalid: {report['invalid_count']}")
            return 0
        report = write_inspection(args.source, args.out, args.sample_points)
        print(f"Scans: {report['scan_count']}; images: {report['image_count']}; records: {report['record_count']}")
        print(f"Report: {args.out.resolve() / 'inspect_report.md'}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("Inspection failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
