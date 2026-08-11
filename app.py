"""
Local Gradio demo — static lookup over precomputed checkpoint results.

Checkpoint JSONL files live under checkpoints/<category>/... and are discovered
at startup. Select Category → Configuration → Dataset → Question to view the
stored response and TRACe scores plus a per-configuration scoreboard.

Run locally:
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from demo_data import (
    format_category_label,
    format_config_table,
    load_catalog,
    scoreboard_for_config,
)

CATALOG = load_catalog()
CATEGORIES = sorted(CATALOG.keys())


def _configs(category: str) -> list[str]:
    return sorted(CATALOG.get(category, {}).keys())


def _datasets(category: str, config_name: str) -> list[tuple[str, str]]:
    """Return [(dataset_id, label), ...] sorted by label."""
    items = CATALOG.get(category, {}).get(config_name, {})
    return sorted(((k, v["label"]) for k, v in items.items()), key=lambda x: x[1].lower())


def update_configs(category: str):
    configs = _configs(category)
    return gr.update(choices=configs, value=configs[0] if configs else None)


def update_datasets(category: str, config_name: str):
    datasets = _datasets(category, config_name)
    labels = [d[1] for d in datasets]
    return gr.update(choices=labels, value=labels[0] if labels else None)


def update_questions(category: str, config_name: str, dataset_label: str):
    datasets = _datasets(category, config_name)
    label_to_id = {label: did for did, label in datasets}
    dataset_id = label_to_id.get(dataset_label)
    if not dataset_id:
        return gr.update(choices=[], value=None)
    rows = CATALOG[category][config_name][dataset_id]["rows"]
    questions = [r["question"] for r in rows]
    return gr.update(choices=questions, value=questions[0] if questions else None)


def show_result(category: str, config_name: str, dataset_label: str, question: str):
    empty_scores = pd.DataFrame(columns=["Metric", "Predicted", "Ground truth"])
    scoreboard_df = pd.DataFrame(scoreboard_for_config(category, config_name, CATALOG))
    config_df = pd.DataFrame(format_config_table(category, config_name), columns=["Field", "Value"])

    if not question:
        return "Select a question first.", empty_scores, scoreboard_df, config_df

    datasets = _datasets(category, config_name)
    label_to_id = {label: did for did, label in datasets}
    dataset_id = label_to_id.get(dataset_label)
    if not dataset_id:
        return "Select a dataset first.", empty_scores, scoreboard_df, config_df

    rows = CATALOG[category][config_name][dataset_id]["rows"]
    row = next((r for r in rows if r["question"] == question), None)
    if row is None:
        return "No checkpoint row found for this question.", empty_scores, scoreboard_df, config_df

    scores_df = pd.DataFrame(
        [
            ["Context relevance", row["pred_relevance"], row["gt_relevance"]],
            ["Context utilization", row["pred_utilization"], row["gt_utilization"]],
            ["Completeness", row["pred_completeness"], row["gt_completeness"]],
            ["Adherence", row["pred_adherence"], row["gt_adherence"]],
        ],
        columns=["Metric", "Predicted", "Ground truth"],
    ).round(4)
    return row["response"], scores_df, scoreboard_df, config_df


default_category = CATEGORIES[0]
default_configs = _configs(default_category)
default_config = default_configs[0]
default_datasets = _datasets(default_category, default_config)
default_dataset_label = default_datasets[0][1]
default_questions = [
    r["question"]
    for r in CATALOG[default_category][default_config][default_datasets[0][0]]["rows"]
]

with gr.Blocks(title="RAGBench Checkpoint Demo") as demo:
    gr.Markdown(
        "## RAGBench Checkpoint Demo\n"
        "Static lookup over precomputed checkpoint results across bio-medical, "
        "general-knowledge, financial, and legal domains. No live API calls — "
        "responses and TRACe scores were generated when these checkpoints were produced."
    )

    with gr.Row():
        category_dd = gr.Dropdown(
            choices=[(format_category_label(c), c) for c in CATEGORIES],
            value=default_category,
            label="Category",
        )
        config_dd = gr.Dropdown(choices=default_configs, value=default_config, label="Configuration")
        dataset_dd = gr.Dropdown(choices=[d[1] for d in default_datasets], value=default_dataset_label, label="Dataset")
        question_dd = gr.Dropdown(choices=default_questions, value=default_questions[0], label="Question")

    show_btn = gr.Button("Show result", variant="primary")

    response_out = gr.Textbox(label="Response", lines=6)
    scores_out = gr.Dataframe(label="TRACe scores (predicted vs. ground truth)", interactive=False)
    scoreboard_out = gr.Dataframe(
        label="RMSE / AUCROC scoreboard for this configuration (all datasets pooled)",
        interactive=False,
    )
    config_out = gr.Dataframe(label="Configuration metadata", interactive=False)

    category_dd.change(update_configs, inputs=category_dd, outputs=config_dd).then(
        update_datasets,
        inputs=[category_dd, config_dd],
        outputs=dataset_dd,
    ).then(
        update_questions,
        inputs=[category_dd, config_dd, dataset_dd],
        outputs=question_dd,
    )

    config_dd.change(
        update_datasets,
        inputs=[category_dd, config_dd],
        outputs=dataset_dd,
    ).then(
        update_questions,
        inputs=[category_dd, config_dd, dataset_dd],
        outputs=question_dd,
    )

    dataset_dd.change(
        update_questions,
        inputs=[category_dd, config_dd, dataset_dd],
        outputs=question_dd,
    )

    show_btn.click(
        show_result,
        inputs=[category_dd, config_dd, dataset_dd, question_dd],
        outputs=[response_out, scores_out, scoreboard_out, config_out],
    )

if __name__ == "__main__":
    demo.launch()
