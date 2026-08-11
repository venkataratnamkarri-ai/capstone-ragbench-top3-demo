#!/usr/bin/env python3
"""Generate docs/rag-demo/ static site for GitHub Pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo_data import CONFIG_METADATA, format_category_label, load_catalog, scoreboard_for_config

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DEFAULT_OUTPUT = SCRIPT_DIR / "docs" / "rag-demo" / "index.html"
DATA_DIR = SCRIPT_DIR / "docs" / "rag-demo" / "data"


def build_category_payload(catalog: dict, category: str) -> dict:
    configs = catalog[category]
    datasets: dict[str, dict] = {}
    scoreboards: dict[str, list[dict]] = {}
    for config_name, config_datasets in configs.items():
        datasets[config_name] = {
            dataset_id: {"label": meta["label"], "rows": meta["rows"]}
            for dataset_id, meta in config_datasets.items()
        }
        scoreboards[config_name] = scoreboard_for_config(category, config_name, catalog)
    return {
        "category": category,
        "configs": sorted(configs.keys()),
        "datasets": datasets,
        "scoreboards": scoreboards,
        "metadata": {k: v for k, v in CONFIG_METADATA.items() if k in configs},
    }


def build_manifest(catalog: dict) -> dict:
    return {
        "categories": [
            {"id": cat, "label": format_category_label(cat), "configs": sorted(configs.keys())}
            for cat, configs in sorted(catalog.items())
        ],
    }


def build_index_html(*, repo_name: str | None = None) -> str:
    base_tag = f'  <base href="/{repo_name}/rag-demo/" />\n' if repo_name else ""
    manifest_json = json.dumps(build_manifest(load_catalog()), ensure_ascii=False)
    manifest_json = manifest_json.replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RAGBench Checkpoint Demo</title>
{base_tag}  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
      margin: 0;
      padding: 24px 16px 48px;
      background: #f3f4f6;
      color: #1f2937;
      line-height: 1.5;
    }}
    .wrap {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 8px; }}
    .lead {{ color: #4b5563; margin: 0 0 24px; font-size: 0.95rem; }}
    .card {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 1px 2px rgba(0,0,0,.04);
    }}
    .row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    label {{
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: #374151;
      margin-bottom: 4px;
    }}
    select, textarea, button {{
      width: 100%;
      font: inherit;
      border-radius: 8px;
      border: 1px solid #d1d5db;
    }}
    select {{
      padding: 10px 12px;
      background: #fff;
    }}
    textarea {{
      padding: 12px;
      min-height: 140px;
      resize: vertical;
      background: #f9fafb;
    }}
    button {{
      width: auto;
      padding: 10px 20px;
      background: #f97316;
      color: #fff;
      border: none;
      font-weight: 600;
      cursor: pointer;
    }}
    button:hover {{ background: #ea580c; }}
    button:disabled {{ opacity: 0.6; cursor: wait; }}
    .section-title {{
      font-size: 0.85rem;
      font-weight: 600;
      margin: 0 0 8px;
      color: #374151;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: left;
    }}
    th {{ background: #f9fafb; font-weight: 600; }}
    .table-wrap {{ overflow-x: auto; }}
    .status {{ font-size: 0.85rem; color: #6b7280; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>RAGBench Checkpoint Demo</h1>
    <p class="lead">
      Static lookup over precomputed checkpoint results across bio-medical,
      general-knowledge, financial, and legal domains. No live API calls —
      responses and TRACe scores were generated when these checkpoints were produced.
    </p>

    <div class="card">
      <div id="status" class="status"></div>
      <div class="row">
        <div>
          <label for="category">Category</label>
          <select id="category"></select>
        </div>
        <div>
          <label for="config">Configuration</label>
          <select id="config"></select>
        </div>
        <div>
          <label for="dataset">Dataset</label>
          <select id="dataset"></select>
        </div>
        <div>
          <label for="question">Question</label>
          <select id="question"></select>
        </div>
      </div>
      <button type="button" id="show-btn">Show result</button>
    </div>

    <div class="card">
      <div class="section-title">Response</div>
      <textarea id="response" readonly></textarea>
    </div>

    <div class="card">
      <div class="section-title">TRACe scores (predicted vs. ground truth)</div>
      <div class="table-wrap" id="scores-table"></div>
    </div>

    <div class="card">
      <div class="section-title">RMSE / AUCROC scoreboard for this configuration</div>
      <div class="table-wrap" id="scoreboard-table"></div>
    </div>

    <div class="card">
      <div class="section-title">Configuration metadata</div>
      <div class="table-wrap" id="config-table"></div>
    </div>
  </div>

  <script id="manifest-data" type="application/json">{manifest_json}</script>
  <script>
    const MANIFEST = JSON.parse(document.getElementById("manifest-data").textContent);
    const categoryEl = document.getElementById("category");
    const configEl = document.getElementById("config");
    const datasetEl = document.getElementById("dataset");
    const questionEl = document.getElementById("question");
    const responseEl = document.getElementById("response");
    const statusEl = document.getElementById("status");
    const showBtn = document.getElementById("show-btn");

    let categoryData = null;
    let datasetMap = {{}};

    function fillSelect(el, options, selected) {{
      el.innerHTML = "";
      for (const opt of options) {{
        const o = document.createElement("option");
        if (typeof opt === "object") {{
          o.value = opt.value;
          o.textContent = opt.label;
        }} else {{
          o.value = opt;
          o.textContent = opt;
        }}
        if ((typeof opt === "object" ? opt.value : opt) === selected) o.selected = true;
        el.appendChild(o);
      }}
    }}

    function renderTable(container, headers, rows) {{
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const hr = document.createElement("tr");
      for (const h of headers) {{
        const th = document.createElement("th");
        th.textContent = h;
        hr.appendChild(th);
      }}
      thead.appendChild(hr);
      table.appendChild(thead);
      const tbody = document.createElement("tbody");
      for (const row of rows) {{
        const tr = document.createElement("tr");
        for (const cell of row) {{
          const td = document.createElement("td");
          td.textContent = cell == null ? "" : String(cell);
          tr.appendChild(td);
        }}
        tbody.appendChild(tr);
      }}
      table.appendChild(tbody);
      container.innerHTML = "";
      container.appendChild(table);
    }}

    async function loadCategory(categoryId) {{
      statusEl.textContent = "Loading checkpoint data…";
      showBtn.disabled = true;
      try {{
        const res = await fetch(`data/${{categoryId}}.json`);
        if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
        categoryData = await res.json();
        statusEl.textContent = "";
      }} catch (err) {{
        statusEl.textContent = `Failed to load data for ${{categoryId}}: ${{err.message}}`;
        categoryData = null;
      }} finally {{
        showBtn.disabled = false;
      }}
    }}

    function datasetsFor(config) {{
      const raw = categoryData?.datasets?.[config] || {{}};
      return Object.entries(raw)
        .map(([id, meta]) => ({{ id, label: meta.label }}))
        .sort((a, b) => a.label.localeCompare(b.label));
    }}

    function updateConfigs() {{
      const configs = categoryData?.configs || [];
      fillSelect(configEl, configs, configs[0] || "");
      updateDatasets();
    }}

    function updateDatasets() {{
      const config = configEl.value;
      const datasets = datasetsFor(config);
      datasetMap = Object.fromEntries(datasets.map(d => [d.label, d.id]));
      fillSelect(datasetEl, datasets.map(d => ({{ value: d.label, label: d.label }})), datasets[0]?.label || "");
      updateQuestions();
      renderScoreboardAndConfig();
    }}

    function updateQuestions() {{
      const config = configEl.value;
      const datasetId = datasetMap[datasetEl.value];
      const rows = categoryData?.datasets?.[config]?.[datasetId]?.rows || [];
      const questions = rows.map(r => r.question);
      fillSelect(questionEl, questions, questions[0] || "");
    }}

    function configMetadataRows() {{
      const config = configEl.value;
      const meta = categoryData?.metadata?.[config];
      if (meta) {{
        return Object.entries(meta).map(([k, v]) => [k, Array.isArray(v) ? JSON.stringify(v) : String(v)]);
      }}
      return [
        ["category", categoryEl.value],
        ["configuration", config],
        ["notes", "No extended metadata — see checkpoint JSONL under checkpoints/"],
      ];
    }}

    function renderScoreboardAndConfig() {{
      const config = configEl.value;
      const sbHeaders = ["domain", "n", "relevance_rmse", "utilization_rmse", "completeness_rmse", "adherence_aucroc"];
      const sbRows = (categoryData?.scoreboards?.[config] || []).map(r => sbHeaders.map(h => r[h]));
      renderTable(document.getElementById("scoreboard-table"), sbHeaders, sbRows);
      renderTable(document.getElementById("config-table"), ["Field", "Value"], configMetadataRows());
    }}

    function showResult() {{
      const config = configEl.value;
      const datasetId = datasetMap[datasetEl.value];
      const question = questionEl.value;
      renderScoreboardAndConfig();

      if (!question || !categoryData) {{
        responseEl.value = "Select a question first.";
        renderTable(document.getElementById("scores-table"), ["Metric", "Predicted", "Ground truth"], []);
        return;
      }}

      const rows = categoryData.datasets?.[config]?.[datasetId]?.rows || [];
      const row = rows.find(r => r.question === question);
      if (!row) {{
        responseEl.value = "No checkpoint row found for this question.";
        renderTable(document.getElementById("scores-table"), ["Metric", "Predicted", "Ground truth"], []);
        return;
      }}

      responseEl.value = row.response || "";
      const fmt = v => (v == null ? "" : Number(v).toFixed(4));
      const scoreRows = [
        ["Context relevance", row.pred_relevance, row.gt_relevance],
        ["Context utilization", row.pred_utilization, row.gt_utilization],
        ["Completeness", row.pred_completeness, row.gt_completeness],
        ["Adherence", row.pred_adherence, row.gt_adherence],
      ].map(([m, p, g]) => [m, fmt(p), fmt(g)]);
      renderTable(document.getElementById("scores-table"), ["Metric", "Predicted", "Ground truth"], scoreRows);
    }}

    async function onCategoryChange() {{
      await loadCategory(categoryEl.value);
      updateConfigs();
      showResult();
    }}

    fillSelect(
      categoryEl,
      MANIFEST.categories.map(c => ({{ value: c.id, label: c.label }})),
      MANIFEST.categories[0]?.id || ""
    );
    categoryEl.addEventListener("change", onCategoryChange);
    configEl.addEventListener("change", () => {{ updateDatasets(); showResult(); }});
    datasetEl.addEventListener("change", () => {{ updateQuestions(); showResult(); }});
    showBtn.addEventListener("click", showResult);

    onCategoryChange();
  </script>
</body>
</html>
"""


def write_data_files(catalog: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for category in catalog:
        payload = build_category_payload(catalog, category)
        out = DATA_DIR / f"{category}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"  Wrote {out} ({out.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static site for GitHub Pages.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-name", default=None)
    args = parser.parse_args()

    if not CHECKPOINTS_DIR.is_dir():
        raise SystemExit(f"Missing checkpoints dir at {CHECKPOINTS_DIR}")

    catalog = load_catalog()
    if not catalog:
        raise SystemExit("No checkpoint JSONL files found under checkpoints/")

    write_data_files(catalog)
    manifest_path = DATA_DIR.parent / "manifest.json"
    manifest_path.write_text(json.dumps(build_manifest(catalog), indent=2), encoding="utf-8")

    html = build_index_html(repo_name=args.repo_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    print(f"Wrote {args.output}")
    if args.repo_name:
        print(f"Base href: /{args.repo_name}/rag-demo/")


if __name__ == "__main__":
    main()
