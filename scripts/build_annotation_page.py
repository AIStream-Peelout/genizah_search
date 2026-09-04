"""Build a local human-annotation page from evaluation result JSONL files.

The page groups answers by case, shows every model's answer side by side
(judge scores hidden by default so human grades stay independent), and
collects per-answer 0-10 grades on the judge dimensions, a pass verdict,
free-text notes, and a per-case preferred model. Annotations autosave to the
browser's localStorage and export as JSON — the export is both the judge
calibration set (human vs judge agreement) and preference data for later
fine-tuning.

The page is a self-contained local file: nothing is uploaded anywhere.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/build_annotation_page.py \\
        --results evals/results/model_comparison_X/*.jsonl \\
        --output evals/annotations/comparison_x.html
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DIMENSIONS = [
    "question_answered",
    "answer_flow",
    "primary_source_linking",
    "groundedness_and_accuracy",
    "citation_and_quote_quality",
    "retrieval_evidence_coverage",
    "restraint_and_limitations",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    :returns: Parsed namespace.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, help="Result JSONL files (one per model/dataset).")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument("--title", default="Genizah RAG annotation")
    parser.add_argument("--dimensions", nargs="*", default=DEFAULT_DIMENSIONS,
                        help="Dimensions to grade (default: the judge v2 dimensions).")
    return parser.parse_args()


def load_rows(paths: List[str]) -> List[Dict[str, Any]]:
    """Load result rows from JSONL files, skipping error rows.

    :param paths: JSONL file paths.
    :returns: Rows with a response.
    :rtype: List[Dict[str, Any]]
    """
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("response"):
                row["_source_file"] = Path(path).name
                rows.append(row)
    return rows


def case_payload(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group rows by (dataset, case) into the page's embedded payload.

    :param rows: Result rows.
    :returns: One entry per case with all model answers.
    :rtype: List[Dict[str, Any]]
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('dataset_id')}::{row.get('case_id')}"
        response = row.get("response") or {}
        entry = grouped.setdefault(key, {
            "key": key,
            "dataset_id": row.get("dataset_id"),
            "case_id": row.get("case_id"),
            "question": row.get("question"),
            "conversation_history": (response.get("metrics") or {}).get("conversation_history"),
            "answers": [],
        })
        model = str(row.get("synthesis_model") or (response.get("metrics") or {}).get("synthesis_model") or "default")
        entry["answers"].append({
            "model": model,
            "source_file": row.get("_source_file"),
            "answer": response.get("answer") or "",
            "resolved_query": response.get("resolved_query"),
            "judge": row.get("judge"),
            "deterministic_pass": (row.get("deterministic") or {}).get("overall_pass"),
            "metrics": {
                "elapsed_seconds": (row.get("metrics") or {}).get("elapsed_seconds"),
                "verification_cycles": (row.get("metrics") or {}).get("verification_cycles"),
                "repair_attempts": (row.get("metrics") or {}).get("repair_attempts"),
            },
        })
    for entry in grouped.values():
        entry["answers"].sort(key=lambda a: a["model"])
    return sorted(grouped.values(), key=lambda e: e["key"])


PAGE_TEMPLATE = """<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 0; background: #f5f3ee; color: #1e1c18; }
  header { position: sticky; top: 0; background: #2c2a25; color: #f5f3ee; padding: 10px 18px;
           display: flex; gap: 14px; align-items: center; flex-wrap: wrap; z-index: 5; }
  header input { padding: 5px 8px; border-radius: 5px; border: none; }
  header button { padding: 6px 12px; border-radius: 5px; border: none; background: #c9a86a; cursor: pointer; font-weight: 600; }
  .progress { opacity: 0.85; font-size: 0.9em; }
  main { max-width: 1200px; margin: 0 auto; padding: 18px; }
  .case { background: #fff; border: 1px solid #d8d2c4; border-radius: 8px; margin-bottom: 26px; padding: 16px 18px; }
  .case h2 { margin: 0 0 4px; font-size: 1.05em; }
  .question { font-size: 1.05em; margin: 8px 0 2px; }
  .history { background: #faf8f2; border-left: 3px solid #c9a86a; padding: 8px 10px; margin: 8px 0;
             font-size: 0.88em; white-space: pre-wrap; max-height: 160px; overflow-y: auto; }
  .answers { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; margin-top: 12px; }
  .answer { border: 1px solid #e2dccd; border-radius: 6px; padding: 10px 12px; display: flex; flex-direction: column; }
  .answer h3 { margin: 0 0 6px; font-size: 0.92em; color: #6a5b3a; }
  .answer pre { white-space: pre-wrap; overflow-wrap: anywhere; font-family: Georgia, "Times New Roman", serif;
                font-size: 0.9em; background: #fcfbf7; border: 1px solid #eee7d6; border-radius: 4px;
                padding: 8px; max-height: 380px; overflow-y: auto; direction: auto; unicode-bidi: plaintext; }
  .grades { display: grid; grid-template-columns: 1fr 64px; gap: 4px 8px; margin-top: 8px; font-size: 0.85em; align-items: center; }
  .grades input[type=number] { width: 56px; padding: 3px; }
  .verdict { margin-top: 8px; font-size: 0.9em; }
  textarea { width: 100%; box-sizing: border-box; min-height: 44px; margin-top: 6px; font-size: 0.88em; }
  .judge { font-size: 0.82em; background: #f2efe6; border-radius: 4px; padding: 6px 8px; margin-top: 8px; display: none; }
  body.show-judge .judge { display: block; }
  .prefer { margin-top: 12px; padding-top: 10px; border-top: 1px dashed #d8d2c4; font-size: 0.92em; }
  .meta { font-size: 0.78em; color: #7a7466; }
  .done { outline: 3px solid #7da87b; }
</style>
<header>
  <strong>__TITLE__</strong>
  <label>Annotator: <input id="annotator" placeholder="name" size="10"></label>
  <label><input type="checkbox" id="showJudge"> show judge scores (grade blind first)</label>
  <button onclick="exportAnnotations()">Export annotations JSON</button>
  <span class="progress" id="progress"></span>
</header>
<main id="main"></main>
<script>
const DATA = __DATA__;
const DIMENSIONS = __DIMENSIONS__;
const STORE_KEY = "genizah_annotations::__STORE_ID__";

function store() { try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch (e) { return {}; } }
function save(s) { localStorage.setItem(STORE_KEY, JSON.stringify(s)); updateProgress(); }
function annKey(caseKey, model) { return caseKey + "@@" + model; }

function updateProgress() {
  const s = store();
  let total = 0, done = 0;
  for (const c of DATA) for (const a of c.answers) {
    total += 1;
    const ann = s[annKey(c.key, a.model)] || {};
    if (ann.verdict) done += 1;
  }
  document.getElementById("progress").textContent = done + " / " + total + " answers annotated";
  document.querySelectorAll(".answer").forEach(el => {
    const ann = s[annKey(el.dataset.caseKey, el.dataset.model)] || {};
    el.classList.toggle("done", Boolean(ann.verdict));
  });
}

function setField(caseKey, model, field, value) {
  const s = store();
  const k = annKey(caseKey, model);
  s[k] = s[k] || { case_key: caseKey, model: model, scores: {} };
  if (field.startsWith("score:")) s[k].scores[field.slice(6)] = value === "" ? null : Number(value);
  else s[k][field] = value;
  s[k].updated_at = new Date().toISOString();
  save(s);
}
function setCaseField(caseKey, field, value) {
  const s = store();
  const k = caseKey + "@@__case__";
  s[k] = s[k] || { case_key: caseKey, model: null };
  s[k][field] = value;
  s[k].updated_at = new Date().toISOString();
  save(s);
}
function fieldValue(caseKey, model, field) {
  const ann = store()[annKey(caseKey, model)] || {};
  if (field.startsWith("score:")) return (ann.scores || {})[field.slice(6)] ?? "";
  return ann[field] ?? "";
}

function exportAnnotations() {
  const payload = {
    exported_at: new Date().toISOString(),
    annotator: document.getElementById("annotator").value || null,
    store_id: "__STORE_ID__",
    dimensions: DIMENSIONS,
    annotations: store(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotations___STORE_ID__.json";
  a.click();
}

function esc(text) { const d = document.createElement("div"); d.textContent = text || ""; return d.innerHTML; }

function render() {
  const main = document.getElementById("main");
  for (const c of DATA) {
    const caseDiv = document.createElement("div");
    caseDiv.className = "case";
    let inner = `<h2>${esc(c.case_id)} <span class="meta">(${esc(c.dataset_id)})</span></h2>`;
    if (c.conversation_history) inner += `<div class="history">${esc(JSON.stringify(c.conversation_history, null, 1))}</div>`;
    inner += `<div class="question"><strong>Q:</strong> ${esc(c.question)}</div><div class="answers">`;
    for (const a of c.answers) {
      const key = c.key, m = a.model;
      inner += `<div class="answer" data-case-key="${esc(key)}" data-model="${esc(m)}">`;
      inner += `<h3>${esc(m)} <span class="meta">${a.metrics.elapsed_seconds ?? "?"}s · verify ${a.metrics.verification_cycles ?? "?"} · repairs ${a.metrics.repair_attempts ?? "?"}</span></h3>`;
      if (a.resolved_query) inner += `<div class="meta">interpreted as: ${esc(a.resolved_query)}</div>`;
      inner += `<pre>${esc(a.answer)}</pre>`;
      inner += `<div class="grades">`;
      for (const dim of DIMENSIONS) {
        inner += `<label for="${esc(key + m + dim)}">${esc(dim)}</label>` +
          `<input type="number" min="0" max="10" step="1" id="${esc(key + m + dim)}"` +
          ` value="${esc(String(fieldValue(key, m, "score:" + dim)))}"` +
          ` onchange="setField('${esc(key)}','${esc(m)}','score:${esc(dim)}',this.value)">`;
      }
      inner += `</div><div class="verdict">Verdict: `;
      for (const v of ["pass", "fail"]) {
        const checked = fieldValue(key, m, "verdict") === v ? "checked" : "";
        inner += `<label><input type="radio" name="v${esc(key + m)}" ${checked}` +
          ` onchange="setField('${esc(key)}','${esc(m)}','verdict','${v}')"> ${v}</label> `;
      }
      inner += `</div><textarea placeholder="notes (what is wrong / what is good)"` +
        ` onchange="setField('${esc(key)}','${esc(m)}','notes',this.value)">${esc(fieldValue(key, m, "notes"))}</textarea>`;
      if (a.judge) inner += `<div class="judge"><strong>LLM judge</strong> mean ${a.judge.score_mean}` +
        ` · ${esc(JSON.stringify(a.judge.scores))} · critical ${esc(JSON.stringify(a.judge.critical_failures))}</div>`;
      inner += `</div>`;
    }
    inner += `</div><div class="prefer">Preferred answer: `;
    for (const a of c.answers) {
      const checked = (store()[c.key + "@@__case__"] || {}).preferred_model === a.model ? "checked" : "";
      inner += `<label><input type="radio" name="p${esc(c.key)}" ${checked}` +
        ` onchange="setCaseField('${esc(c.key)}','preferred_model','${esc(a.model)}')"> ${esc(a.model)}</label> `;
    }
    inner += `<label><input type="radio" name="p${esc(c.key)}"` +
      ` onchange="setCaseField('${esc(c.key)}','preferred_model','tie')"> tie</label></div>`;
    caseDiv.innerHTML = inner;
    main.appendChild(caseDiv);
  }
  document.getElementById("showJudge").onchange = (e) =>
    document.body.classList.toggle("show-judge", e.target.checked);
  updateProgress();
}
render();
</script>
"""


def build_page(rows: List[Dict[str, Any]], title: str, dimensions: List[str], store_id: str) -> str:
    """Render the annotation page HTML.

    :param rows: Result rows.
    :param title: Page title.
    :param dimensions: Dimensions to grade.
    :param store_id: Stable id namespacing this page's localStorage and export.
    :returns: HTML text.
    :rtype: str
    """
    cases = case_payload(rows)
    page = PAGE_TEMPLATE
    page = page.replace("__TITLE__", html.escape(title))
    page = page.replace("__DATA__", json.dumps(cases, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__DIMENSIONS__", json.dumps(dimensions))
    page = page.replace("__STORE_ID__", store_id)
    return page


def main() -> None:
    """Build the annotation page."""
    args = parse_args()
    rows = load_rows(args.results)
    if not rows:
        raise SystemExit("No annotatable rows found in the given results")
    store_id = Path(args.output).stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%d")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_page(rows, args.title, args.dimensions, store_id), encoding="utf-8")
    models = sorted({str(r.get("synthesis_model") or "default") for r in rows})
    print(f"Wrote {output} ({len(case_payload(rows))} cases × models {models}). "
          f"Open it locally, grade blind, then Export annotations JSON.")


if __name__ == "__main__":
    main()
