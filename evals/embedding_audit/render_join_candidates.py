"""Render the text-adjacency Bavli join candidates as a local HTML review sheet.

Reads ``join_candidates_text_bavli.json`` (from ``join_candidates_text.py``) and the image
manifest, and writes ``results/join_candidates_bavli_review.html``: one row per candidate
pair with both first images (public GCS URLs), the parsed frames, and the physical metadata,
for a scholar to accept/reject by eye. Nothing is fetched at build time.
"""

import html
import json
from pathlib import Path

from embed_utils import AUDIT_ROOT

HERE = Path(__file__).parent
MERGED = Path.home() / "Documents/GitHub/historical-document-analysis/src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl"


def main() -> None:
    """Write the review sheet."""
    cands = json.loads((AUDIT_ROOT / "join_candidates_text_bavli.json").read_text())["strong"]
    manifest = json.loads((AUDIT_ROOT / "image_manifest_v1.json").read_text())
    wanted = {c["a"] for c in cands} | {c["b"] for c in cands}
    info = {}
    with open(MERGED, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["canonical_id"] in wanted:
                ktiv = (rec.get("sources") or {}).get("ktiv") or {}
                frames = []
                for e in ktiv.get("scholarly_entries") or []:
                    fr = ((e.get("subsections") or {}).get("writing_characteristics") or {}).get("frame")
                    frames += fr if isinstance(fr, list) else ([fr] if fr else [])
                info[rec["canonical_id"]] = {"shelfmark": rec.get("shelfmark_display"), "frames": sorted(set(frames))[:3]}
    rows = []
    for c in cands:
        cells = []
        for side in ("a", "b"):
            cid = c[side]
            img = manifest["images"].get(cid)
            meta = info.get(cid, {})
            pic = f'<img loading="lazy" src="{html.escape(img)}">' if img else "<em>no image</em>"
            cells.append(f"<td>{pic}<div><b>{html.escape(str(meta.get('shelfmark')))}</b><br>"
                         f"{'<br>'.join(html.escape(f) for f in meta.get('frames', []))}</div></td>")
        rows.append(f"<tr><td>{html.escape(c['tractate'])}<br>gap {c['gap']} amud</td>{''.join(cells)}</tr>")
    page = f"""<!doctype html><meta charset="utf-8"><title>Bavli join candidates</title>
<style>body{{font:14px system-ui;margin:16px}}td{{vertical-align:top;padding:6px;border-bottom:1px solid #ccc}}
img{{max-width:420px;max-height:520px;display:block}}</style>
<h1>Talmud Bavli join candidates from text adjacency ({len(cands)} pairs)</h1>
<p>Same tractate, abutting amud ranges, no material/script conflict, lines-per-page within ±2.
Unverified — for scholarly review. Source: evals/embedding_audit/join_candidates_text.py</p>
<table>{''.join(rows)}</table>"""
    out = HERE / "results" / "join_candidates_bavli_review.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    main()
