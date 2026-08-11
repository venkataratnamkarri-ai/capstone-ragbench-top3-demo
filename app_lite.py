"""Gradio Lite entrypoint -- stdlib + gradio only (no pandas/sklearn/micropip deps)."""

import json
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
SCOREBOARDS_PATH = Path(__file__).parent / "scoreboards.json"

SCOREBOARD_HEADERS = [
    "domain",
    "n",
    "relevance_rmse",
    "utilization_rmse",
    "completeness_rmse",
    "adherence_aucroc",
]
TRACE_HEADERS = ["Metric", "Predicted", "Ground truth"]
CONFIG_HEADERS = ["Field", "Value"]


@dataclass(frozen=True)
class NotebookConfigDefaults:
    generator_model: str = "llama-3.3-70b-versatile"
    judge_model: str = "llama-3.3-70b-versatile"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generation_prompt_style: str = "long_cot"
    chunking_strategy: str = "none"
    chunk_size_chars: int = 500
    chunk_overlap_sentences: int = 1
    top_k_retrieve: int = 20
    top_k_final: int = 5
    use_hybrid_search: bool = True
    use_reranker: bool = True
    rrf_k: int = 60
    use_mmr: bool = True
    mmr_fetch_k: int = 40
    mmr_lambda: float = 0.5
    domains: tuple = ("covidqa", "pubmedqa")
    max_samples_per_domain: int = 5
    max_retries: int = 4
    request_min_interval_s: float = 10.0
    strip_cot_answer: bool = True


CONFIG_OVERRIDES = {
    "v1.1": {"generation_prompt_style": "long", "max_samples_per_domain": 15},
    "v7": {"generation_prompt_style": "long", "max_samples_per_domain": 15, "use_mmr": False},
    "v8": {
        "generation_prompt_style": "long",
        "max_samples_per_domain": 15,
        "use_mmr": False,
        "domains": ("pubmedqa",),
    },
}

FULL_CONFIGS = {
    name: {**vars(NotebookConfigDefaults()), **overrides} for name, overrides in CONFIG_OVERRIDES.items()
}
CONFIGS = CONFIG_OVERRIDES


def load_checkpoints() -> dict[str, dict[str, list[dict]]]:
    data: dict[str, dict[str, list[dict]]] = {}
    for config_name in CONFIGS:
        config_dir = CHECKPOINTS_DIR / config_name
        domains: dict[str, list[dict]] = {}
        for path in sorted(config_dir.glob("*.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not rows:
                continue
            domains[rows[0]["domain"]] = rows
        data[config_name] = domains
    return data


def load_scoreboards() -> dict[str, list[dict]]:
    return json.loads(SCOREBOARDS_PATH.read_text(encoding="utf-8"))


def scoreboard_table(config_name: str) -> list[list]:
    return [[row.get(col) for col in SCOREBOARD_HEADERS] for row in SCOREBOARDS[config_name]]


def format_config_table(config_name: str) -> list[list]:
    cfg = FULL_CONFIGS[config_name]
    return [[key, str(value)] for key, value in cfg.items()]


CHECKPOINT_DATA = load_checkpoints()
SCOREBOARDS = load_scoreboards()


def update_domains(config_name: str):
    domains = sorted(CHECKPOINT_DATA[config_name].keys())
    return gr.update(choices=domains, value=domains[0] if domains else None)


def update_questions(config_name: str, domain: str):
    rows = CHECKPOINT_DATA[config_name].get(domain, [])
    questions = [r["question"] for r in rows]
    return gr.update(choices=questions, value=questions[0] if questions else None)


def show_result(config_name: str, domain: str, question: str):
    empty_scores: list[list] = []
    scoreboard = scoreboard_table(config_name)
    config_table = format_config_table(config_name)

    if not question:
        return "Select a question first.", empty_scores, scoreboard, config_table

    rows = CHECKPOINT_DATA[config_name].get(domain, [])
    row = next((r for r in rows if r["question"] == question), None)
    if row is None:
        return "No checkpoint row found for this question.", empty_scores, scoreboard, config_table

    scores = [
        ["Context relevance", row["pred_relevance"], row["gt_relevance"]],
        ["Context utilization", row["pred_utilization"], row["gt_utilization"]],
        ["Completeness", row["pred_completeness"], row["gt_completeness"]],
        ["Adherence", row["pred_adherence"], row["gt_adherence"]],
    ]
    return row["response"], scores, scoreboard, config_table


with gr.Blocks(title="RAGBench Top-3 Config Demo") as demo:
    gr.Markdown(
        "## RAGBench Top-3 Configuration Demo\n"
        "Static lookup over precomputed checkpoint results for the 3 best-performing "
        "configurations found during iteration (v1.1, v7, v8). No live API calls -- "
        "responses and TRACe scores shown here were generated when these checkpoints "
        "were produced."
    )

    default_config = "v1.1"
    default_domains = sorted(CHECKPOINT_DATA[default_config].keys())
    default_domain = default_domains[0]
    default_questions = [r["question"] for r in CHECKPOINT_DATA[default_config][default_domain]]

    with gr.Row():
        config_dd = gr.Dropdown(choices=list(CONFIGS.keys()), value=default_config, label="Configuration")
        domain_dd = gr.Dropdown(choices=default_domains, value=default_domain, label="Subdomain")
        question_dd = gr.Dropdown(choices=default_questions, value=default_questions[0], label="Question")

    show_btn = gr.Button("Show result", variant="primary")

    response_out = gr.Textbox(label="Response", lines=6)
    scores_out = gr.Dataframe(
        label="TRACe scores (predicted vs. ground truth)",
        headers=TRACE_HEADERS,
        interactive=False,
    )
    scoreboard_out = gr.Dataframe(
        label="RMSE / AUCROC scoreboard for this configuration (across all its checkpointed rows)",
        headers=SCOREBOARD_HEADERS,
        interactive=False,
    )
    config_out = gr.Dataframe(label="Full configuration used", headers=CONFIG_HEADERS, interactive=False)

    config_dd.change(update_domains, inputs=config_dd, outputs=domain_dd).then(
        update_questions, inputs=[config_dd, domain_dd], outputs=question_dd
    )
    domain_dd.change(update_questions, inputs=[config_dd, domain_dd], outputs=question_dd)

    show_btn.click(
        show_result,
        inputs=[config_dd, domain_dd, question_dd],
        outputs=[response_out, scores_out, scoreboard_out, config_out],
    )

demo.launch()
