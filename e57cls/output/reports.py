"""English machine reports and localized HTML from one string table."""

import csv
import html
import io
import json
from pathlib import Path

from e57cls.files import atomic_json, atomic_text


def legend_rows(preset):
    rows = [*preset.get("special", []), *[c for c in preset["classes"] if c.get("enabled", True)]]
    mapping = preset.get("code_map", {})
    return [{"code": int(mapping.get(str(c["code"]), c["code"])), "internal_name": c["id"],
             "name_de": c.get("name_de", c["id"]), "name_en": c.get("name_en", c["id"]),
             "color": c["color"], "source": "ASPRS standard" if c["code"] in (1,2,3,5,6,9,11,14) else "project-defined",
             "prompts": c.get("prompts", [])} for c in rows]


def write_legends(output: Path, rows):
    atomic_json(output/"legend.json", rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
    writer.writeheader()
    writer.writerows({**row, "prompts": " | ".join(row["prompts"])} for row in rows)
    atomic_text(output/"legend.csv", stream.getvalue())
    strings = json.loads((Path(__file__).resolve().parents[1]/"gui/strings_de.json").read_text(encoding="utf-8"))
    body = "".join(f'<tr><td>{r["code"]}</td><td>{html.escape(r["name_de"])}</td><td style="background:{r["color"]}">{r["color"]}</td></tr>' for r in rows)
    atomic_text(output/"legend.html", f'<!doctype html><meta charset="utf-8"><title>{strings["legend_title"]}</title><h1>{strings["legend_title"]}</h1><table>{body}</table>')


def write_report(output: Path, report, rows):
    atomic_json(output/"report.json", report)
    strings = json.loads((Path(__file__).resolve().parents[1]/"gui/strings_de.json").read_text(encoding="utf-8"))
    counts = report["output"]["class_counts"]
    total = report["output"]["record_count"]
    body = "".join(f'<tr><td>{r["code"]}</td><td>{html.escape(r["name_de"])}</td><td>{counts.get(str(r["code"]),0):,}</td><td>{counts.get(str(r["code"]),0)/max(1,total)*100:.2f}%</td></tr>' for r in rows)
    hints = "".join(f'<li>{html.escape(strings.get(w["code"], strings["technical_warning"]))}</li>' for w in report.get("warnings", []))
    summary = '<p>'+html.escape(strings['report_original'])+'</p>'
    summary += '<p>'+strings['report_runtime'].format(minutes=report['total_seconds']/60)+' · '+strings['report_unclassified'].format(share=report['output']['unclassified_share']*100)+'</p>'
    error = max(report['output']['readback']['max_coordinate_error_m'])*1000
    summary += '<p>'+strings['report_preservation'].format(error=error)+'</p>'
    scenes = ''.join(f'<tr><td>{s["scan"]}</td><td>{html.escape(strings.get("scene_"+s["scene"],s["scene"]))}</td></tr>' for s in report['scans'])
    links = ''.join(f'<li><a href="{html.escape(name,quote=True)}">{strings[key]}</a></li>' for name,key in [(Path(report['output']['path']).name,'report_cloud'),('legend.html','legend_title'),('report.json','report_json'),('log.txt','report_log'),('run_config.json','report_config')])
    if report.get('reflection_analysis'):
        links += f'<li><a href="reflection_report.json">{strings["report_reflections"]}</a></li>'
        overlays = sorted((output/'diagnostics'/'reflections').glob('*_overlay.jpg'))
        links += ''.join(f'<li><a href="{p.relative_to(output).as_posix()}">{html.escape(p.stem)}</a></li>' for p in overlays)
    atomic_text(output/"report.html", f'<!doctype html><html lang="de"><meta charset="utf-8"><title>{strings["report_title"]}</title><style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ccc}}a{{color:#176a9b}}</style><h1>{strings["report_title"]}</h1><p>{strings["report_total"].format(count=total)}</p>{summary}<h2>{strings["report_links"]}</h2><ul>{links}</ul><table><tr><th>{strings["code"]}</th><th>{strings["class"]}</th><th>{strings["point_count"]}</th><th>{strings["share"]}</th></tr>{body}</table><h2>{strings["report_scans"]}</h2><table>{scenes}</table><h2>{strings["hints"]}</h2><ul>{hints}</ul><p>{strings["report_visual"]}</p><p>{strings["report_details"]}</p></html>')
