"""Discover checkpoint JSONL files and normalize rows for the demo UI."""

from __future__ import annotations

import json
import re
from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"

# Optional metadata for bio-medical configs (shown in the config table).
CONFIG_METADATA: dict[str, dict] = {

    "v19_hybrid_json_optimal": {
        "generation_prompt_style": "rich_dual_track",
        "max_samples_per_domain": 30,
        "use_mmr": False,
        "domains": ["cuad"],
        "notes": "Hybrid (RRF) + Chunk 800 (The Undisputed Champion)",
    },
    "v25_hybrid_long_reasoning": {
        "generation_prompt_style": "long_cot",
        "max_samples_per_domain": 30,
        "use_mmr": False,
        "domains": ["cuad"],
        "notes": "Hybrid (RRF) + Chunk 800 (Factual Adherence Master)",
    },
    "v31_parent_child_standard": {
        "generation_prompt_style": "paper_short",
        "max_samples_per_domain": 30,
        "use_mmr": False,
        "domains": ["cuad"],
        "notes": "Parent-Child + Chunk 256 (Academic Baseline)",
    },

    "v8c": {
        "generation_prompt_style": "long",
        "max_samples_per_domain": 15,
        "use_mmr": True,
        "domains": ["pubmedqa"],
        "notes": "Whole-document retrieval + MMR, pubmedqa-only",
    },
    "v8e": {
        "generation_prompt_style": "long",
        "max_samples_per_domain": 15,
        "use_mmr": True,
        "domains": ["covidqa", "pubmedqa"],
        "notes": "Whole-document retrieval + MMR, both domains",
    },
    "v8l": {
        "generation_prompt_style": "long",
        "max_samples_per_domain": 15,
        "use_mmr": True,
        "domains": ["covidqa"],
        "notes": "Whole-document retrieval + MMR, covidqa-only",
    },

    "v4": {
        "embed_model": "bge-large-en-v1.5",
        "judge_model": "independent (openai/gpt-oss-120b)",
        "max_samples_per_domain": 15,
        "domains": ["finqa"],
        "notes": "Adopted FinQA config (#1) — best completeness & adherence AUCROC (0.786)",
    },
    "v3": {
        "judge_model": "independent (openai/gpt-oss-120b)",
        "max_samples_per_domain": 15,
        "domains": ["finqa"],
        "notes": "First independent-judge run (#2) — best relevance & utilization RMSE",
    },
    "v5": {
        "generation_temperature": 0.2,
        "max_samples_per_domain": 33,
        "domains": ["finqa"],
        "notes": "Largest-sample FinQA run (#3, diagnostic) — temp=0.2 not pinned",
    },
    "tatv3": {
        "sample_strategy": "random",
        "sample_seed": 42,
        "max_samples_per_domain": 15,
        "domains": ["tatqa"],
        "notes": "Adopted TatQA config (#1) — best RMSE across the board",
    },
    "tatv4": {
        "sample_strategy": "random",
        "sample_seed": 42,
        "max_samples_per_domain": 40,
        "domains": ["tatqa"],
        "notes": "tatv3 config at larger n (#2, reproducibility check, diagnostic)",
    },
    "tatv10": {
        "sample_strategy": "random",
        "sample_seed": 42,
        "max_samples_per_domain": 80,
        "domains": ["tatqa"],
        "notes": "Reverted judge prompt (#3) — highest AUCROC at n=77, but worst mismatch rate",
    },
}


def _config_name_for_jsonl(category_dir: Path, jsonl_path: Path) -> str:
    rel = jsonl_path.relative_to(category_dir)
    parts = list(rel.parts)
    if len(parts) >= 3 and parts[-2] == "checkpoints":
        return parts[-3]
    if len(parts) >= 2:
        return parts[0]
    return jsonl_path.stem


def _dataset_label(jsonl_path: Path, rows: list[dict]) -> str:
    stem = jsonl_path.stem
    domain = rows[0].get("domain") if rows else None
    if domain and stem.startswith(str(domain)):
        return str(domain)
    if domain:
        return f"{domain} ({stem})"
    return stem


def normalize_row(row: dict) -> dict:
    """Map heterogeneous checkpoint schemas to a common TRACe display shape."""
    return {
        "question": row.get("question", ""),
        "response": row.get("response") or row.get("generated_response", ""),
        "pred_relevance": _first_float(
            row, "pred_relevance", "computed_context_relevance"
        ),
        "pred_utilization": _first_float(
            row, "pred_utilization", "computed_context_utilization"
        ),
        "pred_completeness": _first_float(
            row, "pred_completeness", "computed_completeness"
        ),
        "pred_adherence": _first_float(row, "pred_adherence", "computed_adherence"),
        "gt_relevance": _first_float(row, "gt_relevance", "gt_context_relevance"),
        "gt_utilization": _first_float(row, "gt_utilization", "gt_context_utilization"),
        "gt_completeness": _first_float(row, "gt_completeness", "gt_completeness"),
        "gt_adherence": _first_float(row, "gt_adherence", "gt_adherence"),
        "domain": row.get("domain", "n/a"),
    }


def _first_float(row: dict, *keys: str) -> float | None:
    for key in keys:
        if key in row and row[key] is not None:
            return float(row[key])
    return None


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [normalize_row(row) for row in rows]


def load_catalog() -> dict[str, dict[str, dict[str, dict]]]:
    """Returns {category: {config: {dataset_id: {label, rows}}}}."""
    catalog: dict[str, dict[str, dict[str, dict]]] = {}
    if not CHECKPOINTS_DIR.is_dir():
        return catalog

    for category_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        category = category_dir.name
        catalog[category] = {}

        for jsonl_path in sorted(category_dir.rglob("*.jsonl")):
            config_name = _config_name_for_jsonl(category_dir, jsonl_path)
            rows = _read_jsonl(jsonl_path)
            if not rows:
                continue
            dataset_id = jsonl_path.stem
            label = _dataset_label(jsonl_path, rows)
            catalog[category].setdefault(config_name, {})[dataset_id] = {
                "label": label,
                "rows": rows,
            }

    return catalog


def compute_scoreboard(rows: list[dict]) -> list[dict]:
    def rmse(gt: list[float], pred: list[float]) -> float:
        return round((sum((g - p) ** 2 for g, p in zip(gt, pred)) / len(gt)) ** 0.5, 4)

    def aucroc(y_true: list[float], y_score: list[float]) -> float | None:
        if len(set(y_true)) < 2:
            return None
        sorted_pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        rank_sum = sum(i for i, (_, label) in enumerate(sorted_pairs, 1) if label == 1)
        return round((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 4)

    domains = sorted({r["domain"] for r in rows if r.get("domain")})
    scoreboard: list[dict] = []
    for domain in [*domains, "ALL"]:
        sub = rows if domain == "ALL" else [r for r in rows if r.get("domain") == domain]
        if not sub:
            continue
        gt_rel, pred_rel = _paired(sub, "gt_relevance", "pred_relevance")
        gt_util, pred_util = _paired(sub, "gt_utilization", "pred_utilization")
        gt_comp, pred_comp = _paired(sub, "gt_completeness", "pred_completeness")
        gt_adh, pred_adh = _paired(sub, "gt_adherence", "pred_adherence")
        if not gt_rel:
            continue
        y_true = [1 - g for g in gt_adh]
        y_score = [1 - p for p in pred_adh]
        scoreboard.append(
            {
                "domain": domain,
                "n": len(sub),
                "relevance_rmse": rmse(gt_rel, pred_rel),
                "utilization_rmse": rmse(gt_util, pred_util),
                "completeness_rmse": rmse(gt_comp, pred_comp),
                "adherence_aucroc": aucroc(y_true, y_score),
            }
        )
    return scoreboard


def _paired(rows: list[dict], gt_key: str, pred_key: str) -> tuple[list[float], list[float]]:
    gt_vals: list[float] = []
    pred_vals: list[float] = []
    for row in rows:
        gt = row.get(gt_key)
        pred = row.get(pred_key)
        if gt is None or pred is None:
            continue
        gt_vals.append(float(gt))
        pred_vals.append(float(pred))
    return gt_vals, pred_vals


def scoreboard_for_config(category: str, config_name: str, catalog: dict) -> list[dict]:
    datasets = catalog.get(category, {}).get(config_name, {})
    all_rows = [row for ds in datasets.values() for row in ds["rows"]]
    return compute_scoreboard(all_rows)


def format_category_label(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").title()


def format_config_table(category: str, config_name: str) -> list[list[str]]:
    meta = CONFIG_METADATA.get(config_name)
    if meta:
        return [[k, str(v)] for k, v in meta.items()]
    rows = [
        ["category", category],
        ["configuration", config_name],
        ["notes", "No extended metadata — see checkpoint JSONL under checkpoints/"],
    ]
    return rows
