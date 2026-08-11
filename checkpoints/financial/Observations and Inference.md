# RAG Pipeline Observations and Inference

This file tracks `RAG_Pipeline_run1.ipynb`, now canonically located at
`G:\My Drive\Colab Notebooks\RAG_Pipeline_run1.ipynb` (synced via Google Drive Desktop — the same
file Colab opens and the same file edited locally; no more upload/download step). Checkpoints live
under `G:\My Drive\ragbench_v2_run\<config_name>\checkpoints\`.

**Methodology correction:** earlier entries in this log were built from screenshots pasted into
chat, and — before Drive Desktop sync was set up — from a notebook copy that was never actually
the one running in Colab. Reading the real checkpoint JSONL files directly revealed that `v2` was
run twice, at two different points, under two different underlying `Config` states (different
`_config_fingerprint` hashes: `ad7e3634` then `cffdc825`), and that a proposed `v3` (judge-model
swap) had *never* actually executed — no `v3/` checkpoint folder ever existed. This entry replaces
the earlier, partly-unreliable `v2` writeup with figures read directly from the most recent
checkpoint (`finqa_cffdc825.jsonl`, all 15 rows), not transcribed from a screenshot. Going forward,
results in this file should be verified against the checkpoint files directly before logging.

**Note on prior history:** the notebook's own config registry records `baseline` (n=5) →
`v1` (`generation_prompt_style: "long"`) → `baseline_n15`/`v1.1` (n=15 re-runs) → `v2`. Only `v2`'s
checkpoint data was recoverable with confidence, so this file starts substantively there.

---

## Executive Summary

**Project:** Build and iteratively evaluate a Retrieval-Augmented Generation pipeline against
[RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench), across two financial-domain
sources — `finqa` (financial-report QA over tables and text) and `tatqa` (tabular + textual QA) —
using RAGBench's own **TRACe** framework (Relevance, Utilization, Completeness, Adherence) to score
retrieval quality and hallucination detection. Predicted scores are regressed against RAGBench's own
ground-truth annotations via RMSE (continuous metrics) and AUCROC (adherence/hallucination
detection).

**Methodology:** one config change tested at a time, each justified by a specific hypothesis and
pre-registered confirm/falsify criteria, decided only after reading the previous run's actual
results — never planned as a batch sweep in advance. Every configuration is a named, append-only
entry in the notebook's `CONFIG_DEFINITIONS` registry, each with its own checkpoint directory keyed
by a content hash of every field that affects output, so an edited config can never silently resume
stale results under an old name.

### At a Glance

**Run count, both tracks (as of `tatv10`):**

- **FinQA: 6 iterations** — `v2`, `v3`, `v4` (adopted/final), `v5`, `v6`, `finqa_v10`
  (8 standard eval runs total, since `v2` and `v5` were each run twice; plus 12 RGB robustness sub-runs)
- **TatQA: 11 iterations** — `tatv1`, `tatv2`, `tatv3` (adopted/final), `tatv3_recheck`, `tatv4`,
  `tatv5_best`, `tatv6`, `tatv7`, `tatv8`, `tatv9`, `tatv10`
  (11 standard eval runs, one per version — `tatv6` incomplete at 56/80; plus 6 RGB robustness sub-runs)
- **Total: 17 named config iterations, 38 checkpoint run files** across both domains.

**finqa track:**

| Run | Key change from previous | Relevance RMSE | Utilization RMSE | Completeness RMSE | Adherence AUCROC |
|---|---|---|---|---|---|
| `v2` (baseline) | — | 0.296 | 0.347 | 0.444 | 0.429 |
| `v3` | Independent judge model (was same as generator) | 0.098 | 0.078 | 0.248 | 0.714 |
| **`v4` (final, adopted)** | Larger embedding model (`bge-large` vs `bge-small`) | **0.122** | **0.112** | **0.186** | **0.786** |
| **Overall change (v2 → v4)** | | **↓ 59%** | **↓ 68%** | **↓ 58%** | **↑ 83%** |
| `v5` (diagnostic, not adopted)* | Same config as `v4`, `n` raised 15→20 | 0.106 | 0.060 | 0.366 | 0.167 |
| `v6` (incomplete)* | Same as `v5` + `random` sampling | *stalled at 10/15 rows* | | | |

*`v5` was run to fragility-check `v4`'s strong adherence result — it instead revealed that
`temperature=0.2` generation makes single-run comparisons noisy enough, on their own, to explain most
of `v4`↔`v5`'s swing (see the `v5` entry's row-level diff against `v4`'s own 15 questions). `v4` remains
the adopted configuration; `v5`/`v6` are diagnostic, not a replacement candidate. See "Revisiting
finqa" below for the full writeup, including a section-12 RGB robustness sweep (noise-robustness
accuracy + negative-rejection rate) not covered by this table.

**tatqa track** (starts from finqa `v4`'s proven judge/embedding levers, tests domain-specific
retrieval scale and sampling):

| Run | Key change from previous | Relevance RMSE | Utilization RMSE | Completeness RMSE | Adherence AUCROC |
|---|---|---|---|---|---|
| `tatv1` (baseline) | finqa `v4` levers + top_k raised for shorter docs | 0.171 | 0.174 | 0.202 | NaN* |
| `tatv2` | top_k reverted to finqa's values | 0.135 | 0.163 | 0.280 | NaN* |
| **`tatv3` (final, adopted)** | top_k restored to `tatv1` + random sampling | 0.157 | 0.160 | 0.397 | **0.558** |
| `tatv4` (diagnostic, not adopted)** | Same config as `tatv3`, `n` raised 15→40 | 0.205 | 0.164 | 0.389 | 0.617 |

*`tatv1`/`tatv2` used deterministic `first_n` sampling, which happened to draw 15 tatqa rows that
were all `gt_adherence=1.0` — no hallucination case existed in either sample to measure detection
against, making AUCROC structurally undefined regardless of config. Fixed in `tatv3` by adding
seeded random sampling. Note `tatv3`'s RMSE figures are on a *different* 15-question sample than
`tatv1`/`tatv2` (same retrieval config, different questions drawn) — not a like-for-like comparison.

**`tatv4` fragility-checked `tatv3` the same way finqa's `v5` fragility-checked `v4` (n raised, same
otherwise) — and independently reproduced `v5`'s core finding: `temperature=0.2` generation rewrites
most responses on re-run even at fixed config, making single-run deltas noisy. `tatv3` remains the
adopted configuration. See "Revisiting tatqa" below, including its own section-12 RGB robustness
sweep.

*(For all four metrics: RMSE — lower is better. AUCROC — higher is better, 0.5 = random guessing,
RAGBench's own paper reports 0.51-0.80 for comparable judges — finqa `v4` lands at the top of that
range; tatqa `tatv3` at the low end.)*

### RGB robustness at a glance (a different benchmark, section 12 — see "Suggested future work")

Chen et al. 2023's RGB paper — distinct from RAGBench's TRACe framework above — tests **Noise
Robustness** (accuracy as irrelevant-but-real documents fill the context) and **Negative Rejection**
(does the model refuse when no retrieved document actually contains the answer). Run once per domain,
on `v5`'s finqa config and `tatv4`'s tatqa config respectively (both inherit `v4`/`tatv3`'s proven
judge+embedding levers unchanged):

| Domain | Noise accuracy range (ratio 0.0→0.8) | Negative rejection rate |
|---|---|---|
| finqa (`v5`) | 0.47 – 0.67 (not monotonic; see caveat below) | **0.933** |
| tatqa (`tatv4`) | 0.80 – 0.87 (flatter, less noise-sensitive) | **0.867** |

Both domains show strong negative rejection — the model reliably says it can't answer when the context
is entirely irrelevant filler, rather than hallucinating. Noise-robustness accuracy does not decline
monotonically with noise ratio in either domain the way RGB's own paper shows for its search-derived
corpora; RAGBench's much smaller per-question document pool (avg ~1.2–1.3 positive docs/row) means the
*achieved* noise ratio at low nominal targets is coarser than intended — see the full writeups
("Revisiting finqa" / "Revisiting tatqa" below) for the achieved-ratio detail and per-ratio n=15 tables.

### Scoreboard — every configuration, config parameters and results together

Only `v2`, `v3`, `v4`, and (as diagnostic, not-adopted entries) `v5`/`tatv4` have results verified
directly against real checkpoint files; earlier entries in the notebook's registry (`baseline`, `v1`,
`baseline_n15`, `v1.1`) were run but their checkpoint data was not recoverable with confidence, so
their score cells are marked unrecorded rather than guessed at. `v6` is a genuinely incomplete run
(stalled at 10/15 rows) and is omitted from this table rather than shown with a misleading partial
number — see its own entry below. All rows share `domain=finqa`, no chunking (whole-document retrieval
units), `top_k_retrieve=20`/`top_k_final=5`; `n=15` except where noted (`baseline`/`v1` used `n=5`,
`v5` used `n=20`).

| Config | Retrieval | Reranker | MMR | Embedding model | Generator model | Judge model | Prompt style | Judge sees | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | Hybrid+RRF | On | On | `bge-small-en-v1.5` | `llama-3.3-70b-versatile` | `llama-3.3-70b-versatile` (same as gen.) | `long_cot` | parsed final answer only | *unrecorded (n=5)* | *unrecorded* | *unrecorded* | *unrecorded* |
| `v1` | Hybrid+RRF | On | On | `bge-small-en-v1.5` | `llama-3.3-70b-versatile` | `llama-3.3-70b-versatile` (same as gen.) | `long` | parsed final answer only | *unrecorded (n=5)* | *unrecorded* | *unrecorded* | *unrecorded* |
| `baseline_n15` | Hybrid+RRF | On | On | `bge-small-en-v1.5` | `llama-3.3-70b-versatile` | `llama-3.3-70b-versatile` (same as gen.) | `long_cot` | parsed final answer only | *unrecorded* | *unrecorded* | *unrecorded* | *unrecorded* |
| `v1.1` | Hybrid+RRF | On | On | `bge-small-en-v1.5` | `llama-3.3-70b-versatile` | `llama-3.3-70b-versatile` (same as gen.) | `long` | parsed final answer only | *unrecorded* | *unrecorded* | *unrecorded* | *unrecorded* |
| `v2` | Hybrid+RRF | On | On | `bge-small-en-v1.5` | `llama-3.3-70b-versatile` | `llama-3.3-70b-versatile` (**same as gen.**) | `long_cot` | **full reasoning** | 0.296268 | 0.346895 | 0.444160 | 0.428571 |
| `v3` | Hybrid+RRF | On | On | `bge-small-en-v1.5` | `llama-3.3-70b-versatile` | **`openai/gpt-oss-120b`** (independent) | `long_cot` | full reasoning | 0.097670 | 0.077866 | 0.248141 | 0.714286 |
| **`v4` (final, adopted)** | Hybrid+RRF | On | On | **`bge-large-en-v1.5`** | `llama-3.3-70b-versatile` | `openai/gpt-oss-120b` (independent) | `long_cot` | full reasoning | **0.122028** | **0.111656** | **0.186023** | **0.785714** |
| `v5` (diagnostic, n=20) | Hybrid+RRF | On | On | `bge-large-en-v1.5` | `llama-3.3-70b-versatile` | `openai/gpt-oss-120b` (independent) | `long_cot` | full reasoning | 0.106497 | 0.060266 | 0.365624 | 0.166667 |

**Bold** marks the field changed relative to the previous verified row, and the metrics that moved
as a result — this is the single-variable-at-a-time discipline made visible: `v2`→`v3` changed only
`judge_model`; `v3`→`v4` changed only `embed_model`. Every other field held fixed across all three.

**Net improvement, `v2` → `v4`:** relevance_rmse -59%, utilization_rmse -68%, completeness_rmse
-58%, adherence_aucroc from below-random (0.429) to the top of RAGBench's own reported range for
comparable judges (0.51-0.80).

**tatqa track** (same column meanings; domain=`tatqa` throughout, `n=15`, judge/embedding inherited
unchanged from finqa `v4`, chunking `"none"` throughout — tatqa documents are already shorter than
the chunk-size threshold):

| Config | top_k retrieve/final | Sampling | Embedding model | Judge model | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|---|---|---|
| `tatv1` | 25 / 8 | `first_n` | `bge-large-en-v1.5` | `openai/gpt-oss-120b` | 0.170620 | 0.174072 | 0.201843 | NaN |
| `tatv2` | **20 / 5** | `first_n` | `bge-large-en-v1.5` | `openai/gpt-oss-120b` | 0.135228 | 0.163446 | 0.279715 (+38%) | NaN |
| **`tatv3` (final, adopted)** | 25 / 8 (restored) | **`random`, seed 42** | `bge-large-en-v1.5` | `openai/gpt-oss-120b` | 0.156802 | 0.160025 | 0.397329* | **0.557692** |
| `tatv4` (diagnostic, n=40) | 25 / 8 | `random`, seed 42 | `bge-large-en-v1.5` | `openai/gpt-oss-120b` | 0.204771 | 0.164190 | 0.388990 | 0.617117 |

*`tatv3` evaluates a different 15-question sample than `tatv1`/`tatv2` (random vs. `first_n`), so its
completeness_rmse isn't directly comparable — the increase traces largely to one large-magnitude
outlier row, not a broad regression (see the `tatv3` entry below for detail). `tatv1` vs `tatv2` (the
same sample both times) is the clean, directly comparable pair: it showed `tatv1`'s wider retrieval
winning on completeness and adherence accuracy despite losing narrowly on relevance/utilization.
`tatv4` reuses `tatv3`'s exact seed at a larger `n` as a fragility check (parallel to finqa's `v5`); see
"Revisiting tatqa" below — it is diagnostic evidence about measurement noise, not a stronger config.

### Key findings

1. **Measurement/judge quality mattered more than retrieval sophistication.** `v2` enabled hybrid
   search, RRF fusion, MMR diversity, and cross-encoder reranking *simultaneously* — and produced
   results statistically indistinguishable from a plain dense-only retrieval baseline tested
   earlier. All of the project's real gains came from changing *who judges* the output (`v3`) and
   *what embeds* the documents (`v4`), not from retrieval-ranking sophistication.
2. **Self-evaluation bias is real and large.** Using the same model as both generator and judge
   (`v2`) suppressed hallucination detection dramatically — `adherence_aucroc=0.429`, worse than
   random guessing. Swapping to an independent judge model alone (`v3`) nearly doubled it to 0.714,
   with no other change.
3. **Embedding quality trades off across metrics, not uniformly.** Upgrading `bge-small`→`bge-large`
   (`v4`) improved completeness and adherence further, but slightly *worsened* relevance/utilization
   RMSE versus `v3`. A genuine, non-obvious tradeoff — the two continuous-metric wins came at a real
   cost, not a free lunch.
4. **The remaining error is explainable, not mysterious.** By capturing the judge's full reasoning
   in the checkpoint (`judge_sentence_support_information`, zero extra API cost), the residual
   adherence gap in `v4` was traced to two specific, distinct causes: judge inconsistency on
   non-factual "scaffolding" sentences (e.g. "to find X, we need to look at Y"), and genuine
   arithmetic errors in the generator's own multi-step calculations — the latter is compared
   against RAGBench's *own different* reference response, a structural evaluation property, not a
   pipeline bug. A rule-based automated fix for the first cause was attempted and explicitly
   rejected after verification showed it degraded overall accuracy.
5. **Verification infrastructure caught real bugs before they corrupted conclusions.** A content-hashed,
   append-only config registry, guards against stale in-memory objects (retrievers, and later a
   broader domain-coverage check after switching between finqa and tatqa silently kept evaluating
   the wrong domain's cached data), and — most importantly — verifying every result directly against
   the real checkpoint JSONL files (rather than trusting pasted screenshots or partial notebook
   re-runs) caught multiple cases where an apparently-new result was actually stale data left over
   from a previous configuration or even a duplicate copy of the notebook file itself. Several early
   "iterations" in this project's history had to be discarded and re-verified for exactly this
   reason — without that discipline, this project's conclusions would have been built on the wrong
   data.
6. **The judge's scaffolding-sentence leniency inconsistency generalizes across domains.** The same
   `openai/gpt-oss-120b` behavior — inconsistently marking harmless non-factual transition sentences
   ("to find X, we need to look at Y") as unsupported — was independently confirmed in finqa (`v3`,
   `v4`) and tatqa (`tatv1`, `tatv3`, across two different question samples). This is strong evidence
   it's a genuine, domain-independent model limitation rather than a finqa-specific or sample-specific
   artifact — and, since it was already investigated and a fix attempt already rejected on finqa's
   data, that conclusion transfers rather than needing separate re-investigation per domain.
7. **Retrieval-scale tuning (top_k) is domain-specific and requires direct testing, not assumption.**
   tatqa's shorter average document length (388 chars vs. finqa's 1,342) motivated raising
   `top_k_final` from finqa's proven 5 to 8 — and direct A/B testing (`tatv1` vs `tatv2`) confirmed
   this was the right call: reverting to finqa's smaller top_k improved relevance/utilization but
   regressed completeness and adherence accuracy by larger margins. A plausible-sounding adaptation
   was validated, not just assumed.
8. **Generation-temperature sampling noise is a first-order, previously-undisclosed source of
   run-to-run variance.** `generate_response` uses `temperature=0.2`, never held at `0.0` for
   reproducibility. Re-running `v4`'s exact config at a larger `n` (`v5`) showed only 1 of the same 15
   questions' generated responses were byte-identical to `v4`'s — the rest differed, and
   `pred_adherence` flipped on 8 of 15 as a direct result (completeness_rmse +97%, adherence_aucroc
   0.786→0.167). `tatv4` independently reproduced the same pattern in the tatqa track (also 1/15
   identical responses). One byte-identical response (finqa row 10) even changed `pred_adherence`
   between runs purely from the judge's own `temperature=0.0` call landing on a different verdict for
   the same scaffolding sentence — the judge is not perfectly deterministic either. This means every
   single-run metric delta reported in this file (including the confirmed `v2`→`v3`→`v4` gains) carries
   a real noise floor; large, directionally-consistent deltas (like `v2`→`v3`'s) are very unlikely to
   be pure noise, but smaller or mixed-direction deltas (like `v3`→`v4`'s relevance/utilization
   regression) should be read as noisy point estimates, not precise measurements. See the `v5` and
   `tatv4` entries for the full row-level evidence.
9. **Both domains reject unanswerable questions reliably, a materially different result from the
   TRACe adherence numbers.** Section 12's RGB-style Negative Rejection testbed (Chen et al. 2023) —
   a context built entirely from irrelevant-but-real documents, with zero information needed to answer
   — got the model to correctly say it couldn't answer 93.3% of the time on finqa (`v5`) and 86.7% on
   tatqa (`tatv4`). This is a cleaner, more reassuring signal than TRACe's adherence metric, which
   scores a different and harder failure mode (partial support *within* an otherwise-relevant context).
   Noise-robustness accuracy (correctness as noise fills the context) was noisier and non-monotonic in
   both domains at n=15 per ratio, and partly confounded by RAGBench's small per-question document pool
   making some nominal noise ratios unachievable in practice — see the RGB subsections under
   "Revisiting finqa" / "Revisiting tatqa" for the achieved-ratio detail.

### Final configurations

**finqa (`v4`):** Hybrid dense+BM25 retrieval (RRF fusion) + MMR diversity + cross-encoder rerank,
whole-document retrieval units (no chunking), `BAAI/bge-large-en-v1.5` embeddings,
`llama-3.3-70b-versatile` generator, `openai/gpt-oss-120b` judge (independent from the generator),
`long_cot` generation prompt with the judge seeing the full reasoning trace, n=15,
`top_k_retrieve=20`/`top_k_final=5`.

**tatqa (`tatv3`):** Same retrieval/judge/embedding levers as finqa `v4` (inherited, not
independently re-verified for this domain — see limitations), with `top_k_retrieve=25`/
`top_k_final=8` (directly tested and proven for tatqa's shorter documents) and `random` sampling
(seed 42, n=15) in place of `first_n`.

Both remain the adopted configurations after their respective large-`n` fragility checks (`v5` for
finqa, `tatv4` for tatqa) — neither check produced a config change, only the discovery of the
generation-noise floor documented in Key Finding 8.

### Known limitations (state explicitly, don't hide)

- The judge's sentence-keying uses a naive regex splitter, not ground-truth-aligned sentence
  boundaries — a fix that was built and validated (offline, against RAGBench's own data) in an
  earlier, since-retired version of this pipeline, but never ported into the final notebook. Applies
  to both domains.
- `adherence_aucroc` is statistically fragile at n=15 with only a handful of true-hallucination
  examples (1 for finqa `v4`, 2 for tatqa `tatv3`) — a small number of row-level judge disagreements
  can swing it substantially. tatqa's `first_n` sample additionally had *zero* hallucination
  examples in two consecutive attempts (`tatv1`, `tatv2`), making AUCROC briefly undefined until
  random sampling was added.
- Ground-truth scores are computed against RAGBench's own reference response for each question, not
  this pipeline's generated response — some residual RMSE reflects two different responses being
  compared, not pipeline error. Applies to both domains.
- The judge's scaffolding-sentence leniency inconsistency (documented across finqa `v3`/`v4` and
  tatqa `tatv1`/`tatv3`) is a real, unresolved, cross-domain limitation. A rule-based automated fix
  was attempted and explicitly rejected after verification showed it degraded overall accuracy.
- For tatqa specifically: hybrid/rerank/MMR retrieval and the embedding model were inherited from
  finqa's proven config, not independently re-verified on tatqa — only `top_k` and sampling strategy
  were directly tested on this domain. Occasional generator arithmetic mistakes (documented for
  finqa) are a distinct, inherently config-unfixable failure mode.
- **`generate_response` runs at `temperature=0.2`, never pinned to `0.0`.** `v5` and `tatv4` (both
  below) show this alone drives most of the response text to differ on re-runs of an identical config,
  which propagates into utilization/completeness/adherence swings as large as some of this project's
  confirmed config effects. Every single-run RMSE/AUCROC figure in this file should be read as a noisy
  point estimate, not a precise measurement — see Key Finding 8.
- The judge (`temperature=0.0`) is not perfectly deterministic either — `v5`'s row 10 shows the exact
  same judge call (identical question, documents, and response) reaching a different `fully_supported`
  verdict on a re-run. This is evidence for, not against, the Groq-API-hosted `openai/gpt-oss-120b`
  judge having some irreducible sampling noise even at `temperature=0.0`.
- `v6` (finqa, `random` sampling follow-up to `v5`) is an incomplete, stalled run (10/15 rows) and
  contributes no conclusion — flagged here so it isn't mistaken for a completed, simply-unfavorable
  result.
- Section 12's RGB noise-robustness sweep is confounded by RAGBench's small per-question document pool:
  most rows have only ~1 positive document, so nominal noise ratios below ~0.4 are often not achievable
  at `rgb_context_size=5` and collapse to the same near-zero-noise condition (see the RGB subsections
  under "Revisiting finqa" / "Revisiting tatqa"). Negative rejection is not affected by this, since it
  uses `noise_ratio=1.0` (zero positive docs) unconditionally.

### Suggested future work

- Port the ground-truth-aligned sentence-keying fix into the final notebook (both domains).
- Judge-prompt engineering targeted specifically at the scaffolding-sentence leniency inconsistency,
  validated with a fresh run — the highest-value remaining lever, since it's now confirmed to affect
  both domains rather than being domain-specific, and to occur even on byte-identical judge input.
- For tatqa: verify whether hybrid+rerank+MMR retrieval actually helps this domain (finqa found no
  effect; untested whether that holds for tatqa) — the one config axis never independently tested here.
- Chunking strategy (`sentence_window`) as an untested retrieval-side lever for finqa, with `top_k`
  raised in tandem to compensate for smaller retrieval units.
- Scale sample size beyond n=15 once a final config is locked, to get statistically stabler RMSE/AUCROC
  estimates for a production-quality readout — particularly valuable for tatqa given how few
  true-hallucination examples n=15 currently surfaces.
- **Pin `generation_prompt_style`'s generator call to `temperature=0.0`** (or, if some sampling
  temperature is kept deliberately, run each config multiple times and report a spread, not a single
  point estimate) — the single highest-value methodological fix identified by `v5`/`tatv4`, since it
  would let future config deltas be trusted at face value again instead of needing a noise-floor caveat.
- Resume `v6` (10/15 rows on disk, same fingerprint, checkpoint/resume logic will pick up where it
  left off) to get a `random`-sampling finqa read at n=15/20, now that the noise floor is understood
  well enough to interpret it correctly once it lands.
- Widen `rgb_context_size` or otherwise adapt `build_noise_context` so lower nominal noise ratios
  (0.2, and arguably 0.0/0.4) are actually achievable against RAGBench's small per-row document pool,
  rather than collapsing to the same near-zero-noise condition as seen in both domains' section-12
  sweeps.

---

## Iteration: v2 (Full CoT reasoning shown to Judge; hybrid+rerank+MMR baseline)

### Status

Completed. Verified directly against `finqa_cffdc825.jsonl` (15/15 rows, read from the checkpoint
file, not a screenshot).

### Exact Config Used

Everything not listed is the `Config` dataclass default (`generator_model`/`judge_model` =
`llama-3.3-70b-versatile`, `embed_model` = `BAAI/bge-small-en-v1.5`, `reranker_model` =
`cross-encoder/ms-marco-MiniLM-L-6-v2`, `chunking_strategy` = `none`, `top_k_retrieve` = `20`,
`top_k_final` = `5`, `use_hybrid_search` = `True`, `use_reranker` = `True`, `rrf_k` = `60`,
`use_mmr` = `True`, `mmr_fetch_k` = `40`, `mmr_lambda` = `0.5`, `generation_prompt_style` =
`long_cot`).

| Parameter | Value |
|---|---|
| Config name | `v2` |
| Domain | `finqa` |
| Max samples | `15` |
| Retrieval | hybrid (dense + BM25, RRF fusion) + MMR diversity + cross-encoder rerank — all three enabled simultaneously |
| Judge sees | full CoT reasoning + Final Answer (`strip_cot_answer=False`) |
| Generator model | `llama-3.3-70b-versatile` |
| Judge model | `llama-3.3-70b-versatile` (same as generator) |

Note: an earlier run also named `v2` (hash `ad7e3634`) produced different numbers
(0.305/0.287/0.438/0.393) under a slightly different `Config` state whose exact field diff from
this one is not recoverable. This entry treats the later, currently-active `v2` (`cffdc825`) as
authoritative; the earlier one is superseded, not a separate confirmed data point.

### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| finqa | 15 | 0.296268 | 0.346895 | 0.444160 | 0.428571 |

### Row-Level Detail

All 15 rows (read directly from `finqa_cffdc825.jsonl`):

| # | question (truncated) | behavior | pred rel/util/comp/adh | gt rel/util/comp/adh |
|---|---|---|---|---|
| 0 | rate of return, Cadence 2010→2011 | computed | 0.088/0.035/0.40/1.0 | 0.111/0.111/1.00/1.0 |
| 1 | ratio American:US Airways personnel | computed | 0.030/0.030/1.00/1.0 | 0.040/0.040/1.00/1.0 |
| 2 | portion of Anios assets that are intangible | refused | 0.191/0.191/1.00/1.0 | 0.100/0.050/0.50/1.0 |
| 3 | five-year total return, Goldman Sachs | refused | 0.085/0.085/1.00/1.0 | 0.111/0.111/1.00/**0.0** |
| 4 | % change goodwill 2016→2017 | refused | **0.287**/0.011/0.037/1.0 | 0.050/0.050/1.00/1.0 |
| 5 | net change in net revenue, Entergy, 2015 | refused | **0.800**/0.800/1.00/1.0 | 0.071/0.071/1.00/1.0 |
| 6 | 2015 total return for the peer group | refused | **1.000**/1.000/1.00/1.0 | 0.500/0.167/0.33/1.0 |
| 7 | profit margin, printing papers, 2005 | refused | **0.697**/0.030/0.04/1.0 | 0.024/0.024/1.00/1.0 |
| 8 | % change in aggregate net asset values | computed | 0.073/0.018/0.25/**0.0** | 0.154/0.077/0.50/1.0 |
| 9 | % cumulative total shareholder return | refused | 0.000/0.000/1.00/1.0 | 0.077/0.077/1.00/1.0 |
| 10 | % change cash dividends paid per share | computed | 0.020/0.020/1.00/1.0 | 0.034/0.034/1.00/1.0 |
| 11 | after-tax share-based comp cost, 2010 | computed | 0.019/0.019/1.00/**0.0** | 0.040/0.040/1.00/1.0 |
| 12 | average "other" loans held, 2012/2011 | refused | 0.077/0.769/1.00/1.0 | 0.034/0.034/1.00/1.0 |
| 13 | % change total financial liabilities | computed | 0.026/0.026/1.00/1.0 | 0.067/0.067/1.00/1.0 |
| 14 | net revenues, investment management, 2016 | computed | 0.023/0.023/1.00/1.0 | 0.029/0.029/1.00/1.0 |

**8 of 15 responses (53%) are refusals.** Adherence mismatches: row 3 (pred=1.0, gt=0.0 — judge
too lenient on a truly hallucinated refusal), rows 8 and 11 (pred=0.0, gt=1.0 — judge too harsh on
two genuinely correct computed answers). 3 mismatches / 15 = 80% raw exact-match rate.

### Observations

1. Relevance is dramatically over-marked on 4 of the 8 refusals — rows 4, 5, 6, 7 — by 2x to nearly
   30x the ground-truth value (row 5: 0.800 vs 0.071; row 7: 0.697 vs 0.024). The other 4 refusals
   (rows 2, 3, 9, 12) don't show this pattern as strongly. This looks like a real sub-pattern, not
   universal to all refusals: when the model refuses, the judge sometimes marks broad swaths of
   retrieved-but-unused content as "relevant to the topic," rather than the tighter set an actual
   computed answer's reasoning chain would anchor to.
2. Adherence AUCROC (0.429, below random) is driven by only 3 mismatched rows out of 15, on a
   severely imbalanced label (14 gt-adherent, 1 gt-hallucinated). The one true positive (row 3) got
   the *lowest possible* hallucination score from the judge, while two negatives (rows 8, 11) got
   the *highest* — about as anti-correlated as 3 mismatches out of 15 can produce. At this class
   balance, `adherence_aucroc` is not a statistically reliable signal; a single flipped judgment on
   the one true-hallucination row would swing it enormously. This should be flagged explicitly in
   the presentation rather than treated as a stable, comparable-across-configs metric at n=15.
3. Rows 8 and 11 are a *different* failure mode from the others: the judge marked genuinely correct,
   computed answers as unsupported (pred_adherence=0.0) even though gt says they're fully adherent
   — the opposite direction from the "judge too lenient on refusals" story in row 3.
4. `completeness_rmse` (0.444) remains the worst continuous metric.

### Inference

The refusal-specific relevance over-marking (Observation 1) suggests the judge's `relevance`
assessment is less disciplined when there's no concrete computed answer to anchor against — it may
be scanning for "does this document relate to the general topic" rather than "was this document
actually needed." This is a distinct, more specific hypothesis than the earlier "retrieval imprecision"
framing, and it's judge-side, not retrieval-side: rows 5, 6, and 7's *retrieval* may well be fine;
it's the judge's relevance labeling on refusal responses that inflates the denominator mismatch.

Given the adherence AUCROC's extreme sensitivity to single-row judgments at this sample size
(Observation 2), and that its two failure directions (row 3: too lenient; rows 8/11: too harsh) look
like judge inconsistency rather than a one-directional bias, **judge-model swap remains the best
next test** — but the n=15 caveat means even a real improvement could look noisy, and any AUCROC
delta from a 3-mismatch baseline should be read cautiously.

**Confounded / not yet isolated:** chunking strategy (`none`), embedding model (`BAAI/bge-small-en-v1.5`),
judge model (shares generator's model), and the structural point that ground-truth scores were
computed against RAGBench's own reference response, not this pipeline's — some residual gap may be
two different responses being judged, not a pipeline defect.

### Next-Config Decision

**Hypothesis:** Generator and judge sharing a model, combined with a possibly under-disciplined
relevance assessment on refusal responses, are both plausible contributors to the adherence and
relevance-marking issues. An independent judge model is the more isolable, more directly testable
of the two.

**Proposed isolated test (`v3`):** Swap `judge_model` to `openai/gpt-oss-120b`. Every other field
identical to `v2`. This has been registered in the notebook but **never actually executed** —
confirmed by the absence of any `v3/` folder under `ragbench_v2_run` in Drive.

**Confirm criteria:** `adherence_aucroc` improves and the refusal-specific relevance over-marking
(rows like 5, 6, 7 in this data) shrinks. Given the n=15/1-positive caveat above, also check whether
the *specific* rows that disagreed (3, 8, 11) resolve, not just the aggregate AUCROC number.

**Falsify criteria:** the same rows mismatch in the same directions with a different judge model —
meaning the issue is more structural (e.g. inherent to the judge prompt/task, or the gt-response
mismatch) than model-specific, and the next candidate becomes chunking or embedding.

**Status:** Registered in the notebook, notebook now correctly placed at the synced Drive path.
Not yet run — this will be the first time `v3` actually executes.

## Iteration: v3 (Independent Judge Model)

### Status

Completed. Verified directly against `G:\My Drive\ragbench_v2_run\v3\checkpoints\finqa_180e261e.jsonl`
(15/15 rows, real run — 9m19s wall-clock).

### Exact Config Used

Identical to `v2` except:

| Parameter | Value |
|---|---|
| Config name | `v3` |
| Judge model | `openai/gpt-oss-120b` (was `llama-3.3-70b-versatile`) |

### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| finqa | 15 | 0.097670 | 0.077866 | 0.248141 | 0.714286 |

| metric | v2 | v3 | change |
|---|---|---|---|
| relevance_rmse | 0.296268 | 0.097670 | -67% |
| utilization_rmse | 0.346895 | 0.077866 | -78% |
| completeness_rmse | 0.444160 | 0.248141 | -44% |
| adherence_aucroc | 0.428571 | 0.714286 | into RAGBench's reported good-judge range |

### Row-Level Detail

All 15 rows, adherence match/mismatch against gt (read directly from the checkpoint):

| # | question (truncated) | behavior | match? | pred_adh | gt_adh |
|---|---|---|---|---|---|
| 0 | rate of return, Cadence | computed | **MISMATCH** | 0.0 | 1.0 |
| 1 | ratio American:US Airways | computed | **MISMATCH** | 0.0 | 1.0 |
| 2 | portion of Anios assets | refused | **MISMATCH** | 0.0 | 1.0 |
| 3 | five-year return, Goldman Sachs | refused | match | 0.0 | 0.0 |
| 4 | % change goodwill | refused | match | 1.0 | 1.0 |
| 5 | net change net revenue, Entergy | refused | match | 1.0 | 1.0 |
| 6 | 2015 total return, peer group | refused | match | 1.0 | 1.0 |
| 7 | profit margin, printing papers | refused | **MISMATCH** | 0.0 | 1.0 |
| 8 | % change aggregate net asset values | computed | match | 1.0 | 1.0 |
| 9 | % cumulative shareholder return | refused | **MISMATCH** | 0.0 | 1.0 |
| 10 | % change cash dividends/share | computed | match | 1.0 | 1.0 |
| 11 | after-tax share-based comp cost | computed | **MISMATCH** | 0.0 | 1.0 |
| 12 | average "other" loans held | refused | **MISMATCH** | 0.0 | 1.0 |
| 13 | % change total financial liabilities | computed | **MISMATCH** | 0.0 | 1.0 |
| 14 | net revenues, investment mgmt | computed | match | 1.0 | 1.0 |

**8 of 15 rows (53%) mismatch on adherence — up from 3/15 (20%) under `v2`.** Notably, several
correct, straightforward computed answers (rows 0, 1, 11, 13) now get flagged `pred_adherence=0.0`
despite being genuinely well-supported.

### Observations

1. relevance_rmse, utilization_rmse, and completeness_rmse all improved substantially and cleanly
   (-67%, -78%, -44%) — this looks like a real, unambiguous win on these three metrics.
2. adherence_aucroc improved from 0.429 to 0.714 (crossing into RAGBench's own reported 0.51-0.80
   range for comparable judges) — but raw exact-match accuracy on adherence *dropped*, from 80%
   (12/15) to 47% (7/15). The new judge is now wrong on adherence more often than not; it's simply
   wrong in a different, AUCROC-favorable direction (flagging more things as unsupported, which
   happens to rank the one true hallucination — row 3 — more correctly, at the cost of flagging
   several genuinely correct computed answers as unsupported too).
3. The new mismatches concentrate on short, computed numeric answers (rows 0, 1, 11, 13) — cases
   where the answer is derived via arithmetic (subtraction, division) rather than quoted verbatim
   from the source text. The judge prompt has an explicit `numerical_reasoning` tag meant to handle
   exactly this case; `gpt-oss-120b` appears less reliable at applying it than
   `llama-3.3-70b-versatile` was.

### Inference

This is a genuine, well-motivated improvement on three of four metrics, but adherence's AUCROC gain
should be reported with the accuracy caveat attached, not as a clean win — the metric's extreme
sensitivity to a 14:1 class imbalance (flagged in the `v2` entry) means a harsher-but-differently-wrong
judge can look better on AUCROC while being *less* trustworthy example-by-example. For the
presentation, the defensible claim is: "switching judge models improved retrieval/utilization/
completeness scoring accuracy substantially, and improved adherence ranking, but revealed the new
judge is less reliable at recognizing arithmetic-derived (`numerical_reasoning`) answers as
supported" — not simply "adherence improved."

**Confounded / not yet isolated:** chunking strategy (`none`), embedding model
(`BAAI/bge-small-en-v1.5`). Both remain untested; `v3`'s improvement on relevance/utilization/
completeness came entirely from the judge swap, with retrieval unchanged from `v2`.

### Next-Config Decision

**Hypothesis:** `completeness_rmse` (0.248), while much improved, remains the weakest continuous
metric. Whole-document retrieval (`chunking_strategy="none"`) means each of the 5 retrieved units
can bundle substantial irrelevant content alongside the relevant part, inflating the sentence-count
denominator TRACe's ratios divide by — a stronger embedding model could change which documents enter
the candidate pool in the first place (a recall/precision question), independent of the judge-model
change just tested.

**Chunking was considered and rejected for this next test**, in favor of an embedding-model swap:
chunking requires also raising `top_k_retrieve`/`top_k_final` to compensate for chunks carrying less
content per unit than whole documents (documented elsewhere in this project's history — chunking
without raising top_k in tandem regressed every metric previously). That's two fields changing
together on a top_k choice that isn't empirically tuned for this notebook's granularity — a real risk
of an ambiguous or regressed result that would waste a full ~15-row run of Groq budget. Embedding
model is a single, cleanly attributable field with no compensating parameters needed.

**Proposed isolated test (`v4`):** `embed_model: "BAAI/bge-large-en-v1.5"` (same family as the
`bge-small-en-v1.5` baseline — same training methodology, larger/more accurate — isolating embedding
*quality* without an architecture or domain-adaptation confound). Everything else identical to `v3`
(`judge_model="openai/gpt-oss-120b"`, hybrid+rerank+MMR retrieval, `strip_cot_answer=False`, `n=15`).

**Also implemented alongside `v4` (applies going forward, not itself a tested variable):** the
checkpoint record now captures the judge's raw annotation (`relevance_explanation`,
`overall_supported_explanation`, `all_relevant_sentence_keys`, `all_utilized_sentence_keys`,
`sentence_support_information`) and the retrieved documents/rerank scores — previously computed on
every judge call and discarded after scoring. This costs no extra API tokens and means future
disagreements (e.g. `v3`'s `numerical_reasoning` mismatches) can be diagnosed directly from the
checkpoint instead of requiring a dedicated re-run.

**Confirm criteria:** completeness_rmse and/or relevance/utilization_rmse decrease further from
`v3`'s 0.248/0.098/0.078.

**Falsify criteria:** all three stay flat or regress — meaning candidate-pool composition isn't the
remaining bottleneck, and the next candidate becomes chunking (now justified as the next thing to
try, with the top_k compensation reasoned through above) or direct investigation of the
`numerical_reasoning` judge weakness using the newly-captured raw annotation data.

**Status:** Implemented in the notebook (`v4` registered in `CONFIG_DEFINITIONS`,
`ACTIVE_CONFIG_NAME` set to `"v4"`, checkpoint enrichment added to section 10). Not yet run.

## Iteration: v4 (Embedding Model Swap + Enriched Checkpoint Logging)

### Status

Pending run. Verify directly against `G:\My Drive\ragbench_v2_run\v4\checkpoints\finqa_*.jsonl`
(should be a new hash, new folder) before logging, and confirm the `NOTEBOOK BUILD MARKER`
(patch-2026-07-11-v4) + `Active config: v4` print.

### Exact Config Used

Identical to `v3` except:

| Parameter | Value |
|---|---|
| Config name | `v4` |
| Embedding model | `BAAI/bge-large-en-v1.5` (was `BAAI/bge-small-en-v1.5`) |

### Results

Verified directly against `G:\My Drive\ragbench_v2_run\v4\checkpoints\finqa_f01aea86.jsonl`
(15/15 rows).

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| finqa | 15 | 0.122028 | 0.111656 | 0.186023 | 0.785714 |

| metric | v2 (last confirmed baseline) | v3 | v4 | v4 vs v2 | v4 vs v3 |
|---|---|---|---|---|---|
| relevance_rmse | 0.296268 | 0.097670 | 0.122028 | -59% | +25% (worse) |
| utilization_rmse | 0.346895 | 0.077866 | 0.111656 | -68% | +43% (worse) |
| completeness_rmse | 0.444160 | 0.248141 | 0.186023 | -58% | -25% (better) |
| adherence_aucroc | 0.428571 | 0.714286 | 0.785714 | into top of RAGBench's reported range | +10% (better) |
| adherence mismatches (of 15) | — | 8 | 6 | — | fewer, and a real accuracy gain this time, not just ranking |

### Row-Level Detail

Using the enriched logging (`judge_sentence_support_information`, captured at zero extra API cost),
all 6 adherence mismatches were read directly, not guessed at:

- **Rows 6, 8, 10** — the judge marks a harmless methodological/transition sentence ("to find X, we
  need to look at Y") as `fully_supported=False`, sometimes even while tagging it `"general"` in
  `supporting_sentence_keys` — its own category system implies this should count as supported. Since
  `adherence = ALL(fully_supported)`, this single mis-flagged sentence sinks the whole response even
  though every substantive claim is correctly marked supported.
- **Rows 11, 13** — the *generator* actually made real arithmetic mistakes (row 11: "$11.8M... not
  derived correctly"; row 13: "-3.08%... not the correct percentage change"), correctly caught by
  the judge. `gt_adherence=1.0` here reflects RAGBench's own (different, correct) reference response,
  not a claim that our response is right — the structural gt-response mismatch flagged since the
  `v2` entry.
- **Row 7** — a legitimate partial-support nuance flagged by the judge, not a scaffolding-sentence
  artifact.

**A rule-based automated fix was attempted and rejected**: forgiving any sentence whose
`supporting_sentence_keys` contains a recognized category tag (`general`, `supported_without_sentence`,
`well_known_fact`, `numerical_reasoning`) regardless of its `fully_supported` flag was tested against
this same checkpoint data (zero API cost) and made things *worse* — AUCROC dropped to 0.321 — because
it also forgave row 3 (the one genuinely hallucinated response, previously correctly identified),
destroying our only true-positive detection. The scaffolding-sentence issue is real but requires
either judge-prompt-level intervention (untestable without a new paid run) or case-by-case human
judgment, not a safe deterministic correction.

### Observations

1. `v4` is a decisive, across-the-board improvement over `v2` (the last fully-confirmed baseline):
   -59% relevance_rmse, -68% utilization_rmse, -58% completeness_rmse, adherence_aucroc into the top
   of RAGBench's own reported range (0.51-0.80).
2. `v4` vs. `v3` (isolating the embedding swap alone, judge held constant) is a genuine mixed result:
   completeness and adherence improved further, but relevance/utilization got moderately worse with
   `bge-large` than `bge-small`. Not a clean win on every axis, but a net positive given adherence's
   real accuracy gain (8→6 mismatches, not just an AUCROC ranking artifact).
3. The remaining adherence gap decomposes into two distinct, well-evidenced causes: judge
   inconsistency on non-factual scaffolding sentences (3 rows), and genuine generator arithmetic
   errors on multi-step calculations (2 rows) — the latter inherently unfixable via config, since
   it's compared against a different reference response.

### Inference

`v4` represents the strongest configuration found across this project's iteration history, and the
remaining gap is now explained by name rather than mysterious. Neither remaining failure mode has a
free, safe, or clearly-scoped next fix: the scaffolding-sentence issue would require judge-prompt
changes that can't be validated without spending more Groq budget on a new run with uncertain payoff,
and the arithmetic-error mode isn't a pipeline defect at all.

### Next-Config Decision — concluding the iteration here

**Decision: stop iterating, adopt `v4` as the final configuration for this project.** Rationale:
every metric is dramatically better than the last confirmed baseline; `adherence_aucroc=0.786` is a
legitimately strong, presentable number at the top of RAGBench's reported range; the two remaining
failure modes are identified and explainable (useful "future work" material for the presentation)
rather than pointing at an obvious missing lever; and further single-variable tests (chunking, a
different embedding, a different generator) would each cost a real ~10-20 minute run of non-refundable
Groq budget for uncertain, likely-marginal payoff against an already-strong result.

**Final configuration (`v4`):** hybrid (dense + BM25, RRF fusion) + MMR diversity + cross-encoder
rerank retrieval, whole-document units (no chunking), `BAAI/bge-large-en-v1.5` embeddings,
`llama-3.3-70b-versatile` generator, `openai/gpt-oss-120b` judge (independent from the generator),
`long_cot` prompt with the judge seeing full reasoning (`strip_cot_answer=False`), n=15 on `finqa`.

**Known limitations to state explicitly in the presentation, not hide:**
- The judge's sentence-keying still uses a naive regex splitter, not ground-truth-aligned boundaries
  (a fix that was built and validated in an earlier, since-retired notebook, but never ported here).
- `adherence_aucroc` at n=15 with a 14:1 (or similar) class imbalance is statistically fragile —
  a small number of row-level judge errors can swing it substantially.
- Ground-truth scores were computed against RAGBench's own reference response for each row, not this
  pipeline's generated response — some residual RMSE reflects two different responses being compared,
  not pipeline error.
- Two distinct, named remaining failure modes (judge leniency inconsistency on scaffolding sentences;
  occasional generator arithmetic mistakes) were identified but not resolved, and are natural
  candidates for future iteration beyond this project's scope.

---

## Second track: tatqa domain

Everything above (`baseline` through `v4`) is the `finqa` track. `tatqa` (RAGBench's TAT-QA source —
tabular + textual financial QA) is a separate, independently-tracked domain starting fresh from its
own Iteration 1, not a continuation of the finqa sequence, even though its starting config borrows
`v4`'s proven levers.

### Iteration: tatv1 (New Domain, finqa's Proven Config + Context-Volume Adaptation)

#### Status

Pending run.

#### Exact Config Used

Same core levers as `v4` (`judge_model="openai/gpt-oss-120b"`, `embed_model="BAAI/bge-large-en-v1.5"`,
hybrid+rerank+MMR retrieval, `chunking_strategy="none"`, `strip_cot_answer=False`, n=15), plus:

| Parameter | Value | Why |
|---|---|---|
| Domain | `tatqa` (RAGBench's actual identifier — an earlier draft used `"tat"`, which isn't valid) | — |
| `top_k_retrieve` | `25` (was `20`) | tatqa documents are much shorter than finqa's (verified offline: avg 388 chars / median 319 vs. finqa's avg 1342) |
| `top_k_final` | `8` (was `5`) | at the inherited `top_k_final=5`, the generator would see only ~1940 chars of context per question — under a third of what `v4` effectively gave finqa (~6710 chars) — risky for tabular reasoning, which often needs several short table-row documents combined |

Offline verification before running (zero API cost): `tatqa` test split has 3,338 rows, 1,272 unique
corpus documents (similar count to finqa's 1,097, but total corpus text volume only ~1/3 of finqa's
due to the shorter documents) — indexing time with `bge-large` is expected to be comparable to or
faster than finqa's, not worse, despite the larger row count.

#### Results

Verified directly against `G:\My Drive\ragbench_v2_run\tatv1\checkpoints\tatqa_5f2ac0fe.jsonl`
(15/15 rows).

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa | 15 | 0.170620 | 0.174072 | 0.201843 | **NaN — undefined** |

`adherence_aucroc` is undefined because all 15 `gt_adherence` values in this `first_n`-sampled slice
happen to be `1.0` — no hallucination example exists in the sample to measure detection against.
This is a sampling artifact, not a config or judge failure; no amount of config tuning fixes it.
Raw adherence accuracy (not AUCROC): 9/15 correct (60%), all 6 mismatches `pred_adherence=0.0` vs
`gt_adherence=1.0` — the same judge-too-harsh-on-scaffolding-sentences pattern seen in finqa `v3`/`v4`
(same judge model, `openai/gpt-oss-120b`).

Compared to finqa `v4` (0.122 / 0.112 / 0.186 / 0.786): relevance and utilization RMSE are both
noticeably worse for tatqa under the same judge/embedding levers.

#### Row-Level Detail

Verified offline (retrieved-document sentence counts, zero API cost) that every row retrieved
exactly 8 documents (`top_k_final=8` applied correctly) totaling 13-30 sentences depending on the
row. The rows with the largest relevance gaps have RAGBench's own implied candidate-document sentence
count far smaller than what was retrieved — e.g. row 0: `gt_relevance=0.500` (implying RAGBench's own
candidate pool for that question was ~2 sentences total) vs. our 24 retrieved sentences, giving
`pred_relevance=0.042`. Same structural corpus-scope-vs-candidate-set mismatch identified for finqa,
now confirmed to recur in a second domain — but the *magnitude* here (up to 30 total sentences per
row) is large enough to be worth directly testing against a smaller `top_k`.

#### Observations

1. `adherence_aucroc` is uncomputable for this specific n=15 sample — a sampling composition issue,
   separate from any config question.
2. relevance_rmse (0.171) and utilization_rmse (0.174) are meaningfully worse than finqa `v4`'s
   (0.122 / 0.112), while completeness_rmse (0.202) is roughly comparable (0.186).
3. The same judge-leniency-inconsistency pattern from finqa `v3`/`v4` recurs here (6/15 adherence
   mismatches, all in the harsh-on-scaffolding direction) — consistent with this being a genuine
   `gpt-oss-120b` behavior, not a finqa-specific artifact.
4. Every row retrieved the full `top_k_final=8` documents, and the rows with the worst relevance gaps
   show retrieved-sentence counts far exceeding RAGBench's own implied per-question candidate size.

#### Inference

The relevance/utilization RMSE gap vs. finqa is plausibly explained by `tatv1`'s `top_k_final=8`
(raised from finqa's proven 5, to compensate for tatqa's shorter documents) retrieving more total
sentences per question than the ratio-based metrics' denominators can absorb without dilution. This
is not proof the top_k increase was a mistake — a smaller top_k could equally hurt completeness by
missing needed evidence for multi-step tabular reasoning — so it needs to be tested directly, not
assumed either way.

The adherence-judge pattern recurring identically in a second domain strengthens the finqa `v4`
finding: this looks like a genuine, domain-independent `gpt-oss-120b` behavior (harsh on non-factual
scaffolding sentences) rather than something specific to finqa's phrasing patterns.

#### Next-Config Decision

**Hypothesis:** `top_k_final=8` is diluting the relevance/utilization denominators more than it's
helping completeness for tatqa — reverting to finqa's proven `top_k_retrieve=20`/`top_k_final=5`
should improve relevance/utilization RMSE.

**Proposed isolated test (`tatv2`):** Revert `top_k_retrieve`/`top_k_final` to `Config`'s bare
defaults (20/5). Every other field identical to `tatv1` (`judge_model="openai/gpt-oss-120b"`,
`embed_model="BAAI/bge-large-en-v1.5"`, hybrid+rerank+MMR, `domains=("tatqa",)`, n=15).

**Confirm criteria:** relevance_rmse and utilization_rmse decrease, without completeness_rmse
regressing substantially (i.e. the smaller top_k isn't now missing needed evidence).

**Falsify criteria:** completeness_rmse gets meaningfully worse — meaning the extra retrieved
documents in `tatv1` genuinely were needed for tabular reasoning, and the dilution is a cost worth
paying; in that case `top_k_final=8` should be kept and the next lever should target something else.

**Separately, not part of this test:** for a presentation-ready tatqa readout, `adherence_aucroc`
will need either a larger `n` or a sampling change away from `first_n` to guarantee at least one
hallucination example appears — treat this as a final scale-up step once other levers are settled,
not an iteration-by-iteration lever (it costs more tokens with no guarantee of resolving in any
single attempt).

**Status:** Implemented in the notebook (`tatv2` registered, `ACTIVE_CONFIG_NAME` set to `"tatv2"`).
Not yet run.

### Iteration: tatv2 (Revert top_k to finqa's Proven Values)

#### Status

Pending run.

#### Exact Config Used

Identical to `tatv1` except:

| Parameter | Value |
|---|---|
| Config name | `tatv2` |
| `top_k_retrieve` | `20` (was `25`) |
| `top_k_final` | `5` (was `8`) |

#### Results

Verified directly against `G:\My Drive\ragbench_v2_run\tatv2\checkpoints\tatqa_3b4e4e46.jsonl`
(15/15 rows).

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa | 15 | 0.135228 | 0.163446 | 0.279715 | NaN (same sampling issue as `tatv1`) |

| metric | tatv1 (top_k_final=8) | tatv2 (top_k_final=5) | winner |
|---|---|---|---|
| relevance_rmse | 0.171 | **0.135** | tatv2 |
| utilization_rmse | 0.174 | **0.163** | tatv2 (narrow) |
| completeness_rmse | **0.202** | 0.280 (+38%) | tatv1 |
| adherence mismatches (of 15) | **6** | 8 | tatv1 |

#### Row-Level Detail

Row 12 is the clearest single illustration of the tradeoff: `tatv2`'s narrower retrieval gives
`pred_relevance=0.556` (over-marked vs. `gt=0.333`) but `pred_completeness=0.000` — the smaller
candidate pool didn't contain what was needed, so nothing relevant ended up utilized at all, despite
the judge marking plenty as "relevant" in the abstract.

#### Observations

1. relevance_rmse and utilization_rmse both improved with the smaller top_k, as hypothesized.
2. completeness_rmse regressed substantially (+38%) and adherence accuracy also got worse
   (6/15 → 8/15 mismatches) — the smaller retrieval pool is missing needed evidence often enough to
   outweigh the denominator-dilution benefit on the other two metrics.
3. `adherence_aucroc` is still NaN, confirming this is a sampling-composition issue independent of
   top_k — the exact same 15 rows get drawn either way under `first_n`.

#### Inference

Per the pre-registered falsify criteria, this **falsifies the simple version** of the top_k
hypothesis ("smaller top_k straightforwardly helps") — completeness regressing meaningfully means
the extra documents in `tatv1` were genuinely load-bearing for some questions, not just dilution.
**`tatv1`'s wider retrieval (`top_k_retrieve=25`, `top_k_final=8`) is the stronger config on
balance** — it wins on completeness and adherence accuracy by larger margins than `tatv2` wins on
relevance/utilization. The original reasoning behind `tatv1` (short documents need more retrieved
units for tabular reasoning) holds up under direct test.

#### Next-Config Decision

**Decision: restore `tatv1`'s top_k values (not a new lever test) and add one genuinely new,
isolated variable — sampling strategy.** `tatv1` vs `tatv2` already answered the top_k question, so
`tatv3` reverts to `top_k_retrieve=25`/`top_k_final=8`, unchanged from `tatv1`.

**Hypothesis:** `adherence_aucroc` has been NaN in both `tatv1` and `tatv2` because `first_n`
sampling deterministically draws the same 15 rows every time, and those 15 happen to be 100%
`gt_adherence=1.0` — a sampling artifact, not a retrieval/judge quality issue. A differently-composed
sample (same size, different rows) has a real chance of including at least one hallucination case.

**Proposed isolated test (`tatv3`):** Added `sample_strategy`/`random_seed` fields to `Config`
(defaulting to `"first_n"`, so every already-logged config's reproducibility is unaffected).
`tatv3` sets `sample_strategy="random"`, `random_seed=42`, keeps `tatv1`'s top_k values, judge, and
embedding model unchanged, still `n=15`.

**Confirm criteria:** `adherence_aucroc` becomes computable (i.e. at least one `gt_adherence=0.0`
row appears in the new sample).

**Falsify criteria:** the new random sample still happens to be 100% `gt_adherence=1.0` — plausible
if tatqa's underlying hallucination rate is genuinely low; in that case `n` would need to increase
rather than just resampling, a real token-cost tradeoff to make deliberately, not by accident.

**Status:** Implemented in the notebook (`tatv3` registered, `ACTIVE_CONFIG_NAME` set to `"tatv3"`,
domain-coverage guard also added to section 10 after a stale-`RAW_ROWS` issue was diagnosed
separately). Not yet run.

### Iteration: tatv3 (Restore Proven top_k + Random Sampling) — FINAL tatqa configuration

#### Status

Completed. Verified directly against `G:\My Drive\ragbench_v2_run\tatv3\checkpoints\tatqa_75501464.jsonl`
(15/15 rows).

#### Exact Config Used

Identical to `tatv1` except:

| Parameter | Value |
|---|---|
| Config name | `tatv3` |
| `sample_strategy` | `"random"` (was `"first_n"`, new field) |
| `random_seed` | `42` (new field) |

#### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa | 15 | 0.156802 | 0.160025 | 0.397329 | **0.557692** |

**Caveat on comparing to `tatv1`/`tatv2`:** this is a *different* 15-question sample (random seed 42
vs. `first_n`'s fixed slice) — retrieval settings are otherwise identical to `tatv1`, so the
completeness_rmse increase (0.202 → 0.397) is not necessarily a config regression; see Inference.

#### Row-Level Detail

The sample includes 2 real hallucination cases for the first time (`gt_adherence=0.0`): row 3
(missed — judge said `pred_adherence=1.0`) and row 5 (caught correctly). 5 additional rows (0, 1, 6,
10, 12) are false "unsupported" flags on genuinely fine responses — the same scaffolding-sentence
leniency-inconsistency pattern documented in finqa `v3`/`v4` and tatqa `v1`, now confirmed a third
time. Row 10 (`pred_completeness=0.000` vs `gt=1.000`, a full 1.0 gap — a refusal on an
"average restricted stock" question despite the judge finding relevant material) is a single
large-magnitude outlier that, at n=15, disproportionately drives the completeness_rmse increase via
squared error.

#### Observations

1. `adherence_aucroc` is finally computable and real: 0.558 — modest, at the low end of RAGBench's
   reported 0.51-0.80 range for comparable judges, but a genuine signal for the first time.
2. Of the 2 true hallucinations in this sample, the judge caught 1 and missed 1 (50% recall), while
   also producing 5 false "unsupported" flags on correct responses.
3. The scaffolding-sentence leniency-inconsistency pattern first identified in finqa `v3`/`v4` has
   now recurred identically in **two independent tatqa samples** (`tatv1`'s `first_n` slice and
   `tatv3`'s random slice) — strong, repeated evidence this is a genuine, domain-independent
   `openai/gpt-oss-120b` behavior, not a finqa-specific or sample-specific artifact.
4. The completeness_rmse increase vs. `tatv1` is concentrated in one large-magnitude outlier row
   (row 10), not a broad-based regression — consistent with sample-composition variance rather than
   the retrieval config performing worse.

#### Inference

The dominant remaining source of adherence error is now well-characterized rather than mysterious:
a judge-prompt-level behavior (inconsistent leniency on non-factual scaffolding sentences),
confirmed across three independent samples spanning two domains. This was already investigated on
finqa's data — a rule-based automated fix was attempted and explicitly rejected after it degraded
overall accuracy (see the `v4` entry) — so re-investigating it on tatqa's data would very likely
reach the same conclusion for the cost of more analysis, not a new finding. Further retrieval or
embedding tuning is unlikely to move this number, since the failure mode is in how the judge
evaluates support, not in what evidence gets retrieved.

#### Next-Config Decision — concluding the tatqa track here

**Decision: stop iterating, adopt `tatv3` as the final configuration for the tatqa track.**
Rationale, applying the same standard used to conclude the finqa track at `v4`: the blocking
measurement issue (`adherence_aucroc=NaN`) is resolved; the dominant remaining gap is understood and
already investigated (not an open question); further single-variable tests (hybrid retrieval on/off
for tatqa specifically, the one axis never independently verified for this domain) carry a
reasonable prior toward "probably also no effect," given finqa's own hybrid-retrieval test found
zero measurable impact — real but likely small expected payoff against a real, non-refundable cost
in Groq budget and wall-clock time.

**Final configuration (`tatv3`):** hybrid (dense + BM25, RRF fusion) + MMR diversity + cross-encoder
rerank retrieval (inherited from finqa, not independently re-verified for tatqa), whole-document
units (no chunking — tatqa documents are already shorter than the chunk-size threshold),
`BAAI/bge-large-en-v1.5` embeddings, `top_k_retrieve=25`/`top_k_final=8` (directly tested and proven
against `top_k=20/5` — see `tatv1` vs `tatv2`), `llama-3.3-70b-versatile` generator,
`openai/gpt-oss-120b` judge (independent from the generator), `long_cot` prompt with full reasoning
shown to the judge, n=15 on `tatqa`, `random` sampling (seed 42).

**Known limitations, parallel to finqa's:**
- Hybrid/rerank/MMR retrieval and the embedding model were inherited from the finqa track, not
  independently re-verified for tatqa — only `top_k` and `sample_strategy` were directly tested on
  this domain.
- `adherence_aucroc` at n=15 with only 2 true-hallucination examples remains statistically fragile —
  a small number of row-level judge disagreements can swing it substantially.
- The judge's scaffolding-sentence leniency inconsistency (documented across finqa `v3`/`v4` and
  tatqa `v1`/`v3`) is a real, unresolved, cross-domain limitation — a natural candidate for future
  work (judge-prompt engineering), not fixable by the config changes tested in this project.
- Ground-truth scores are computed against RAGBench's own reference response for each row, not this
  pipeline's generated response — the same structural caveat documented for finqa.

---

## Revisiting finqa: Iteration v5 (Larger Sample, `first_n`)

The finqa track was closed at `v4` (see above). `v5` reopens it: the config registry's own comment
frames it as a fragility check, not a new lever — every `v4` field is kept identical, with
`max_samples_per_domain` raised from 15 to 20 to test whether `v4`'s strong `adherence_aucroc=0.786`
was a real result or an artifact of how sensitive that metric is at small n (already flagged as a
known limitation in the `v4` entry above). It was actually run after the tatqa track (`tatv1`–`tatv4`),
not immediately after `v4`.

**Methodology correction (second one under this file's opening discipline):** the previous version of
this entry treated `finqa_cc8da9e7.jsonl` (33 rows) as `v5`'s result. Re-checking directly against the
Drive folder's own file timestamps shows `cc8da9e7` was last written 2026-07-18 14:13 — its 33-row
shape doesn't match `CONFIG_DEFINITIONS`'s current `v5` entry (`max_samples_per_domain=20`) at all, and
is an interrupted attempt at an earlier, since-edited value (the in-notebook comment references an
originally-planned n=40). A **later** checkpoint, `finqa_6332890a.jsonl`, was written 2026-07-19
10:36 — a full day after `cc8da9e7` — has exactly 20 rows matching the current config precisely, and
has a complete, matching RGB robustness sweep (section 12) alongside it, whereas `cc8da9e7`'s RGB
checkpoints are themselves partial/abandoned. `6332890a` is authoritative; `cc8da9e7` is a superseded,
interrupted attempt at a config that no longer exists in the notebook. This entry replaces the earlier
`cc8da9e7`-based writeup in full — the same category of stale-checkpoint error this file's opening
note (and the `v2` entry) already warned about, caught here by checking file timestamps directly rather
than trusting whichever checkpoint was read first.

### Status

Completed. Verified directly against `G:\My Drive\ragbench_v2_run\v5\checkpoints\finqa_6332890a.jsonl`
(20/20 rows) and its five `rgb_noise_finqa_*_6332890a.jsonl` + one `rgb_negrej_finqa_1.0_6332890a.jsonl`
companion files (15/15 rows each — section 12's robustness sweep, its own subsection below).

### Exact Config Used

Identical to `v4` except:

| Parameter | Value |
|---|---|
| Config name | `v5` |
| `max_samples_per_domain` | `20` (was `15`) |

### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| finqa | 20 | 0.106497 | 0.060266 | 0.365624 | 0.166667 |

| metric | v4 (adopted, n=15) | v5 (n=20) | v5 vs v4 |
|---|---|---|---|
| relevance_rmse | 0.122028 | 0.106497 | -13% (better) |
| utilization_rmse | 0.111656 | 0.060266 | -46% (better) |
| completeness_rmse | 0.186023 | 0.365624 | +97% (much worse) |
| adherence_aucroc | 0.785714 | 0.166667 | collapses to *below* random, not just toward it |
| adherence mismatches | 6 of 15 | 14 of 20 | mismatch rate rises from 40% to 70% |

`gt_adherence` class balance at n=20: 18 supported / 2 hallucinated (rows 3 and 15). Both true
hallucinations were **missed** (`pred_adherence=1.0` on both), while 12 of the 18 genuinely-supported
rows were flagged `pred_adherence=0.0` — about as anti-correlated a ranking as this class balance can
produce, which is why AUCROC (0.167) falls *below* 0.5 rather than merely toward it.

### Row-Level Detail: isolating the cause via `v4`'s own 15 questions

`v5`'s config is identical to `v4`'s in every field except sample size, and `first_n` sampling means
`v5`'s first 15 rows are the *exact same 15 questions* `v4` was scored on. Diffing the two runs
question-by-question (both checkpoints read directly) isolates what actually changed between them:

| | count (of 15 shared questions) |
|---|---|
| Same question, **identical generated response text** | 1 |
| Same question, **different response text** | 14 |
| `pred_adherence` verdict flips between `v4` and `v5` | 8 |

**`generate_response`'s `temperature=0.2` (section 7) is not held constant across re-runs of an
identical config** — 14 of 15 identical (question, retrieved docs, config) inputs produced a different
generated response the second time around, which alone is sufficient to explain most of the
adherence/completeness/utilization swing above, with no new retrieval or judge-policy explanation
needed.

More strikingly, **row 10's response is byte-identical between `v4` and `v5`** (same retrieved docs,
same generated text) — yet `pred_adherence` still flips, from `0.0` in `v4` to `1.0` in `v5`. Reading
both judge calls' raw `judge_sentence_support_information` side by side, the input to the judge was
identical; only its verdict on one scaffolding sentence ("to find X we need to look at Y") changed —
`v4`'s judge call marked it `fully_supported=false` (sinking the row, since
`adherence = ALL(fully_supported)`), `v5`'s judge call marked the identical sentence
`fully_supported=true`, tagged `"general"`. The judge model (`openai/gpt-oss-120b`, called at
`temperature=0.0`) is itself not fully deterministic on byte-identical input via the Groq API — a more
specific finding than the "scaffolding-sentence *leniency inconsistency*" framing used since `v4`: it
isn't just that the judge is inconsistently strict across different sentences or rows, it can reach a
different verdict on the literal same sentence, same call shape, `temperature=0.0`, with nothing else
different.

### Observations

1. `v5` is a headline win on relevance_rmse (-13%) and utilization_rmse (-46%) — but per the row-level
   diff above, "win" here means "attributable to sample size plus generation noise," not necessarily
   "the larger sample is intrinsically more accurate." With 14 of 15 shared questions' generated text
   differing between runs, `v4` and `v5` are less a controlled A/B pair and more two independent draws
   from a noisy process that happens to share a config.
2. completeness_rmse more than doubles (0.186 → 0.366) and adherence_aucroc collapses from 0.786 to
   0.167 (below-random) — both explained by the response-level rewrite documented above, not by a new
   judge or retrieval issue.
3. This is the **first direct, controlled demonstration** in this project that `generation_prompt_style`'s
   `temperature=0.2` sampling is a first-order noise source for every response-dependent metric
   (utilization, completeness, adherence) — potentially larger than some of the config deltas this
   project has attributed causal meaning to. relevance_rmse is comparatively insulated, since it scores
   retrieved-document relevance against the question, independent of what the generator actually wrote.
4. The judge's own non-determinism on byte-identical input (row 10) means even if generation were
   pinned to `temperature=0.0`, adherence would likely still show some irreducible run-to-run noise on
   scaffolding-sentence-bearing rows specifically — this doesn't overturn the scaffolding-sentence
   finding documented since `v4`, it sharpens it: the inconsistency isn't just judge-policy-level, it
   can occur on a single fixed call shape.

### Inference

Every metric delta reported between configs in this project (`v2`→`v3`→`v4`, and the tatqa track) was
computed from a single run per config, with `temperature=0.2` generation and non-fully-deterministic
`temperature=0.0` judging. `v5`'s controlled same-question diff against `v4` shows this run-to-run
noise is large enough, on its own, to produce swings of the same order of magnitude as this project's
smaller confirmed effects. The large, one-directional swings this project attributed to config changes
(`v2`→`v3`'s judge-model swap in particular: -67%/-78%/-44% RMSE, AUCROC 0.429→0.714) are almost
certainly still real — effects of that size and directional consistency are implausible as pure
sampling noise — but single-run deltas of `v3`→`v4`'s magnitude (a mixed result, some metrics up some
down) should be read with this noise floor explicitly in mind, not as precise point estimates. This is
a genuine limitation to disclose, not a reason to distrust the project's headline findings.

### Next-Config Decision

**Decision: keep `v4` as the presented/adopted configuration; do not promote `v5`.** Same rationale as
before this correction — `v5`'s completeness/adherence losses are larger in magnitude than its
relevance/utilization gains, and the row-level diff above shows the comparison is confounded by
generation noise rather than a clean sample-size effect either way.

**What actually resolves the sample-size question:** repeated runs of the *identical* config, not a
single `v4`-vs-single-`v5` comparison — the noise floor demonstrated above means one run of each isn't
enough to attribute a metric delta to sample size specifically. This project's remaining Groq budget
went toward completing the tatqa track's own fragility check (`tatv4`, below) and the RGB robustness
sweep (this entry's own subsection, next) instead of a `v5` repeat-run for this reason.

**Status: `v6` (below) was started as the planned `random`-sampling follow-up but was not completed** —
see its own entry for what's actually on disk.

---

### RGB robustness testbed (finqa, `v5`'s config) — section 12

A separate benchmark from RAGBench's TRACe framework above (Chen et al. 2023, "RGB" — Retrieval-
Augmented Generation Benchmark): **Noise Robustness** (does accuracy hold as the context fills with
real-but-irrelevant documents?) and **Negative Rejection** (does the model correctly refuse when
*none* of the retrieved documents contain the answer, instead of hallucinating one?). Uses `v5`'s
config (independent judge, `bge-large` embeddings) but its own sampling
(`rgb_max_samples_per_domain=15`, `rgb_context_size=5`) and its own correctness judge (section 8.5 — a
separate LLM call checking whether the generated answer conveys the same key fact(s) as RAGBench's
reference response, since RGB's own exact-match scoring doesn't apply to RAGBench's free-text answers).

**Verified directly against all six `*_6332890a.jsonl` checkpoint files (15/15 rows each):**

| noise_ratio (nominal) | n | accuracy | mean achieved ratio |
|---|---|---|---|
| 0.0 | 15 | 0.600 | 0.000 |
| 0.2 | 15 | 0.467 | 0.000 |
| 0.4 | 15 | 0.600 | 0.467 |
| 0.6 | 15 | 0.667 | 0.520 |
| 0.8 | 15 | 0.533 | 0.760 |

| Negative rejection | n | rejection rate |
|---|---|---|
| finqa | 15 | **0.933** |

**Achieved-ratio caveat** (documented in the notebook's own `build_noise_context` docstring, confirmed
live here): finqa rows average only ~1.28 positive documents each, so a nominal target ratio of 0.2 is
usually not realizable at `rgb_context_size=5` — both the 0.0 and 0.2 nominal rows land at an
*achieved* ratio of 0.000 (an all-positive-plus-padding context, not a true 20%-noise context). Only
the 0.4/0.6/0.8 rows show a real, distinct achieved ratio.

**Observation:** accuracy does *not* monotonically decline with noise ratio the way the RGB paper's own
Table 1 shows for its search-derived corpora (0.600 → 0.467 → 0.600 → 0.667 → 0.533). At n=15 per ratio
this is plausibly within noise, and it's also consistent with the achieved-ratio caveat above — two of
the five nominal "ratios" are actually the same near-zero-noise condition, and RAGBench's much smaller
per-question document pool means even the higher nominal ratios pad with a small, fixed number of extra
documents rather than RGB's own paper-scale corpus dilution. **Negative rejection is strong** (93.3%):
when the context is entirely irrelevant filler, the model reliably recognizes it can't answer rather
than fabricating a response — a meaningfully different, and better, result than the TRACe adherence
numbers above, which measure a different failure mode (partial support within an otherwise-relevant
context, not total absence of relevant content).

---

## Iteration: v6 (Random-Sampling Follow-up to `v5`) — incomplete, not usable for comparison

### Status

**Incomplete / stalled.** `finqa_8bcd58fa.jsonl` has 10 of the intended 15 rows
(`max_samples_per_domain=15`, `sample_strategy="random"`, `random_seed=42` — mirroring `tatv3`/`tatv4`'s
precedent for isolating sampling method from sample size). Its RGB companion sweep is further behind:
only `rgb_noise_finqa_0.0_8bcd58fa.jsonl` exists, itself only 6 of 15 rows, and no other noise ratio or
the negative-rejection file exists at all. This looks like a run that was started and interrupted (a
Colab disconnect, rate limit, or a manually stopped session) rather than one that hit an error — nothing
on disk indicates *why* it stopped, only that it did.

### Exact Config Used

Identical to `v5` except:

| Parameter | Value |
|---|---|
| Config name | `v6` |
| `sample_strategy` | `random` (was `first_n`) |
| `random_seed` | `42` |

### What's on disk (diagnostic only — not `v6`'s result)

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| finqa (partial) | 10 | 0.100066 | 0.107355 | 0.158114 | NaN |

`adherence_aucroc` is undefined at this partial n because the 10 rows drawn so far happen to be a
single class — not informative about whether `random` sampling itself would resolve anything. Given
`v5`'s finding above (generation noise alone drives large swings run-to-run), even this partial number
should not be read as evidence either way about `first_n` vs. `random` sampling.

### Next-Config Decision

**Resume `v6`, don't re-run it from scratch.** The checkpoint/resume logic already in the notebook
(section 10) will continue from the 10 rows already on disk under the same `8bcd58fa` fingerprint —
registering a new config name would only lose progress. Complete `v6` (and its RGB sweep) before
drawing any conclusion about whether `first_n` vs. `random` sampling, independent of the generation-noise
floor documented in the `v5` entry above, changes the adherence picture. This is the one open
finqa-track question this file does not yet have an answer to.

---

## Revisiting tatqa: Iteration tatv4 (Larger Sample, `random` — parallel fragility check to `v5`)

The tatqa track was closed at `tatv3` (see above). `tatv4` reopens it with the same fragility-check
motivation as finqa's `v5`: every `tatv3` field is kept identical, with `max_samples_per_domain` raised
from 15 to 40 — a larger jump than `v5`'s (15→20), and `sample_strategy="random"`/`random_seed=42` is
kept **unchanged** from `tatv3` rather than reverted to `first_n`, so any shift is attributable to
sample size alone, not a different draw method or seed.

### Status

Completed. Verified directly against
`G:\My Drive\ragbench_v2_run\tatv4\checkpoints\tatqa_354f1791.jsonl` (40/40 rows) and its five
`rgb_noise_tatqa_*_354f1791.jsonl` + one `rgb_negrej_tatqa_1.0_354f1791.jsonl` companion files (15/15
rows each).

### Exact Config Used

Identical to `tatv3` except:

| Parameter | Value |
|---|---|
| Config name | `tatv4` |
| `max_samples_per_domain` | `40` (was `15`) |

### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa | 40 | 0.204771 | 0.164190 | 0.388990 | 0.617117 |

| metric | tatv3 (adopted, n=15) | tatv4 (n=40) | tatv4 vs tatv3 |
|---|---|---|---|
| relevance_rmse | 0.156802 | 0.204771 | +31% (worse) |
| utilization_rmse | 0.160025 | 0.164190 | +3% (~flat) |
| completeness_rmse | 0.397329 | 0.388990 | -2% (~flat) |
| adherence_aucroc | 0.557692 | 0.617117 | +11% (better) |
| adherence mismatches | 6 of 15 | 17 of 40 | 40% → 42.5%, ~flat |

`gt_adherence` class balance at n=40: 37 supported / 3 hallucinated — a larger and slightly richer
positive-class count than `tatv3`'s 2-of-15, one contributor to the AUCROC improvement.

### Row-Level Detail: same overlap check as `v5`

`tatv4` uses the identical `random_seed=42` as `tatv3`, and empirically its first-drawn rows are the
same 15 questions `tatv3` was scored on (verified by direct question-text match, not assumed from the
seed alone — `random.sample`'s output isn't guaranteed to nest across different sample sizes even with
a fixed seed).

| | count (of 15 shared questions) |
|---|---|
| Same question, **identical generated response text** | 1 |
| `pred_adherence` verdict flips between `tatv3` and `tatv4` | not identical row-for-row, but the *same total mismatch count* (6 of 15) recurs on a **different subset of rows** |

This independently reproduces the `v5` finding in a second domain: `generate_response`'s
`temperature=0.2` rewrites 14 of 15 identical-input responses between runs of the same config, and the
resulting adherence verdicts move with the rewritten text rather than staying pinned to the question.
The `v5`→`tatv4` pair together is stronger evidence than either alone that this is a property of the
pipeline's sampling settings, not something specific to finqa or to one run.

### Observations

1. Unlike `v5`, `tatv4`'s adherence_aucroc *improves* over its n=15 predecessor (0.558 → 0.617) rather
   than collapsing — the opposite direction from the finqa track's large-n result. Given the same
   generation-noise floor applies here too (confirmed by the row-level overlap check above), this is
   consistent with the noise being a source of variance in *either* direction, not a one-way
   degradation — `tatv4`'s larger, richer 3-hallucination sample happened to land favorably where `v5`'s
   larger sample happened to land unfavorably.
2. relevance_rmse regresses meaningfully (+31%) while utilization/completeness stay roughly flat — a
   different metric-level pattern than `v5`'s (where relevance improved and completeness regressed
   sharply). This further supports treating each of these single-run deltas as noisy point estimates
   rather than reliable per-metric effects of sample size specifically.
3. Of the 17 adherence mismatches, 15 are the familiar "judge marks a genuinely-supported response
   unsupported" direction (`gt=1`, `pred=0`) and only 2 are the reverse — consistent with the
   scaffolding-sentence-leniency-inconsistency pattern's dominant direction throughout this project,
   though only a couple of the flagged sentences carry an explicit `general`/`well_known_fact` category
   tag this time; a full per-row causal breakdown (as done for `v4` and `v5`) was not repeated at n=40
   given the diminishing marginal value already established by four prior confirmations.

### Inference

`tatv4` does **not** overturn `tatv3` as the adopted tatqa configuration — its relevance regression and
the now twice-confirmed generation-noise floor mean this single run isn't a clean "n=40 is better"
result either. Its main contribution is corroborating, in a second domain, the central methodological
finding from the `v5` entry: single-run deltas in this project carry a real noise floor from
`temperature=0.2` generation (and non-fully-deterministic `temperature=0.0` judging) that is large
enough to flip individual metrics in either direction, independent of any config change.

### Next-Config Decision — concluding both fragility checks here

**Decision: keep `tatv3` (and `v4`, on the finqa side) as the adopted configurations.** Both large-n
follow-ups (`v5`, `tatv4`) are retained as diagnostic evidence about measurement noise, not as
candidate replacements — promoting either would mean chasing single-run noise rather than a real
config improvement. Given the Groq-budget cost already spent establishing this noise floor in two
domains, further sample-size-only re-runs are not planned; the higher-value remaining lever (per the
`v4`/`tatv3` entries) is still judge-prompt engineering targeted at the scaffolding-sentence
inconsistency, now additionally motivated by the finding that the judge itself is not perfectly
deterministic at `temperature=0.0` on identical input.

---

### RGB robustness testbed (tatqa, `tatv4`'s config) — section 12

Same methodology as the finqa RGB sweep above, run against `tatv4`'s config.

**Verified directly against all six `*_354f1791.jsonl` checkpoint files (15/15 rows each):**

| noise_ratio (nominal) | n | accuracy | mean achieved ratio |
|---|---|---|---|
| 0.0 | 15 | 0.800 | 0.000 |
| 0.2 | 15 | 0.867 | 0.000 |
| 0.4 | 15 | 0.867 | 0.500 |
| 0.6 | 15 | 0.800 | 0.500 |
| 0.8 | 15 | 0.800 | 0.800 |

| Negative rejection | n | rejection rate |
|---|---|---|
| tatqa | 15 | **0.867** |

**Observation:** tatqa's noise-robustness accuracy is both higher (0.80–0.87 vs. finqa's 0.47–0.67) and
flatter across noise ratios than finqa's — consistent with tatqa's shorter, more tightly-scoped
documents (see the `tatv1` entry's corpus-length comparison) making it easier for the generator to
locate the one relevant table row/sentence even as filler documents are added. Negative rejection
(86.7%) is strong and in the same range as finqa's (93.3%) — both domains show the model reliably
recognizing when it has no basis to answer, a materially cleaner result than either domain's TRACe
adherence numbers. The same achieved-ratio coarseness caveat from the finqa sweep applies here (0.0 and
0.2 nominal both land at 0.000 achieved).

---

## Second wave, tatqa track: pinning generation temperature + judge-prompt fix (`tatv5_best` → `tatv8`)

`tatv3`/`tatv4` closed the tatqa track above, carrying two open items into "Suggested future work": pin
`generate_response`'s `temperature` to `0.0`, and target judge-prompt engineering at the
scaffolding-sentence leniency inconsistency. This wave is a fresh `CONFIG_DEFINITIONS` sequence,
re-declaring `tatv3`/`tatv4`'s proven levers (independent judge, `bge-large`, tuned top_k) explicitly
from `Config()`'s bare defaults rather than layering on top of `tatv4` directly, and bundling three
changes at once in its first entry (`temperature=0.0`, a judge-prompt change, and `n`/sampling changes)
rather than this project's usual single-variable discipline — flagged here as a real confound, not
hidden, consistent with this file's own stated methodology.

### Iteration: tatv5_best (temp=0.0 + judge-prompt change + n=30 random)

#### Status

Completed, 30/30 rows. Verified directly against
`G:\My Drive\ragbench_v2_run\tatv5_best\checkpoints\tatqa_e563da02.jsonl`.

#### Exact Config Used

Re-declared from `Config()` defaults, reinstating `tatv3`/`tatv4`'s proven levers plus two new ones:

| Parameter | Value |
|---|---|
| Config name | `tatv5_best` |
| Domain | `tatqa` |
| `max_samples_per_domain` | `30` |
| `strip_cot_answer` | `False` (judge sees full reasoning) |
| `judge_model` | `openai/gpt-oss-120b` (independent — matches `tatv3`/`tatv4`) |
| `embed_model` | `BAAI/bge-large-en-v1.5` (matches `tatv3`/`tatv4`) |
| `top_k_retrieve` / `top_k_final` | `25` / `8` (matches `tatv3`/`tatv4`) |
| `generation_temperature` | **`0.0`** (new — every prior iteration in this file used `0.2`) |
| `sample_strategy` / `sample_seed` | `random` / `42` (matches `tatv3`/`tatv4`) |
| Judge prompt | scaffolding-sentence rule widened to also force `fully_supported=True` on `numerical_reasoning`-tagged sentences (new — this is the change `tatv7` later finds a bug in) |

**Confound:** three levers changed relative to `tatv4` at once (generation temperature, judge prompt,
`n` 40→30) — no delta from this run can be cleanly attributed to a single cause.

#### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa | 30 | 0.224959 | 0.241415 | 0.386101 | **NaN — undefined** |

`gt_adherence` class balance: **30 supported / 0 hallucinated** — this particular `random`/seed-42 draw
at n=30 contains zero true-hallucination rows, so AUCROC is structurally undefined again (the same
sampling-composition issue first seen in `tatv1`/`tatv2`, recurring at a different n/seed combination).
Raw adherence accuracy: 27/30 correct (90%).

#### Row-Level Detail

All 3 mismatches are the familiar direction (`pred_adherence=0.0` on a `gt_adherence=1.0` row) — rows
15, 22, 26 — consistent with the scaffolding-sentence leniency-inconsistency pattern documented since
finqa `v3`/`v4`. No hallucinated row exists in this sample, so `tatv5_best` cannot test the one thing
its judge-prompt change was built for.

#### Observations

1. relevance/utilization/completeness RMSE are all worse than `tatv3` (adopted, n=15) and `tatv4`
   (n=40) — 0.225/0.241/0.386 vs. `tatv4`'s 0.205/0.164/0.389 — though with three confounded variables
   changed at once, no single cause is attributable from this run alone.
2. Zero true-hallucination rows in this sample means `tatv5_best` answers neither of the two questions
   it was built to test: whether `temperature=0.0` reduces run-to-run noise, or whether the judge-prompt
   change helps hallucination detection.
3. 90% raw adherence accuracy, entirely explained by the already-characterized scaffolding-sentence
   pattern.

#### Inference

`tatv5_best` mainly re-confirms the known scaffolding-leniency pattern at a new sample; its two
motivating questions remain untested because this specific draw contains no hallucination example.

#### Next-Config Decision

**Hypothesis:** a larger `n` has a better chance of drawing at least one hallucination row to actually
exercise the judge-prompt change.

**Proposed test (`tatv6`):** same config, `max_samples_per_domain` raised 30→80.

**Status:** run — did not complete; see below.

---

### Iteration: tatv6 (n=80, same config) — incomplete, 56/80 rows

#### Status

**Incomplete.** `tatqa_690ef8a9.jsonl` has 56 of the intended 80 rows (24 missing, 30% dropout). Per
the notebook's own inline comment, checkpoint verification showed generation/judge calls exhausting
Groq retries under sustained load at the then-current `request_min_interval_s=4.0`/`max_retries=4` — a
rate-limit dropout, not a data or logic error.

#### Exact Config Used

Identical to `tatv5_best` except:

| Parameter | Value |
|---|---|
| Config name | `tatv6` |
| `max_samples_per_domain` | `80` (was `30`) |

#### Results (partial — dropout-biased, not a clean n=80 read)

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa (56/80) | 56 | 0.238071 | 0.288265 | 0.331849 | 0.427 |

`gt_adherence` class balance: 55 supported / **1 hallucinated** (row 54). 9/56 adherence mismatches
(16%).

#### Row-Level Detail

The single true-hallucination row (54 — "What is the percentage change between inventory purchases from
Supplier A from 2017 to 2018") was **missed**: `pred_adherence=1.0` against `gt_adherence=0.0`. The
response reasons through a multi-step percentage calculation ("To find the percentage change... we need
to first find the percentages..."). The other 8 mismatches are the familiar reverse-direction pattern
(`pred=0.0`, `gt=1.0` — scaffolding-sentence leniency inconsistency).

#### Observations

1. This is the first row in this wave where the judge-prompt change's target failure mode
   (numerical-reasoning hallucination) is actually present in the sample — and the judge missed it,
   marking a wrong calculation as fully supported.
2. Because only 56/80 rows landed via rate-limit dropout (likely correlated with later-processed rows,
   not a random subsample), this checkpoint is diagnostic only, not a valid comparison point against
   `tatv3`/`tatv4`.

#### Inference

The missed hallucination on row 54 is the first concrete evidence that this wave's judge-prompt change
(forcing `fully_supported=True` whenever a sentence carries a `numerical_reasoning`, `general`,
`well_known_fact`, or `supported_without_sentence` tag) may have gone too far — genuinely wrong
arithmetic tagged `numerical_reasoning` now gets waved through instead of caught. This needed a
completed, unbiased run to confirm, motivating `tatv7`.

#### Next-Config Decision

**Hypothesis:** the dropout is a retry-budget problem, not a config problem — raising `max_retries` and
`request_min_interval_s` should let the same n=80 draw complete cleanly.

**Proposed test (`tatv7`):** same config, `max_retries` 4→6, `request_min_interval_s` 4.0→6.0.

**Confirm criteria:** row count reaches at or near 80/80.

---

### Iteration: tatv7 (retry/backoff fix) — 77/80 rows, reveals a real judge-prompt regression

#### Status

Completed (near-fully), 77/80 rows — the retry-budget fix worked: dropout fell from 24/80 (`tatv6`) to
3/80. Verified directly against
`G:\My Drive\ragbench_v2_run\tatv7\checkpoints\tatqa_d91a8aa7.jsonl`.

#### Exact Config Used

Identical to `tatv6` except:

| Parameter | Value |
|---|---|
| Config name | `tatv7` |
| `max_retries` | `6` (was `4`) |
| `request_min_interval_s` | `6.0` (was `4.0`) |

#### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa | 77 | 0.267712 | 0.295281 | 0.387521 | **0.440** |

`gt_adherence` class balance: 75 supported / **2 hallucinated** (rows 54, 71). **`adherence_aucroc=0.44`
— worse than random guessing, and worse than `tatv4`'s 0.617 (the last full-sample comparison point).**

#### Row-Level Detail

**Both true-hallucination rows were missed** — the judge marked both `fully_supported=True`
(`pred_adherence=1.0`) despite `gt_adherence=0.0`:

- Row 54 (repeat from `tatv6`): percentage-change calculation, Supplier A inventory purchases.
- Row 71: "How much did the operating income grow in December 2018 a sequential..." — again a
  multi-step percentage/growth calculation.

Both are **arithmetic/percentage-change questions** — exactly the failure category this wave's
widened judge-prompt rule was risking: sentences tagged `numerical_reasoning` are forced
`fully_supported=True` regardless of whether the arithmetic is actually correct. The remaining 9
mismatches keep the familiar reverse pattern (`pred=0`, `gt=1`, scaffolding-sentence leniency
inconsistency).

#### Observations

1. **This is a regression, not noise:** `tatv7`'s AUCROC (0.44) is below random, and both real
   hallucinations in the sample were missed — a clean, 2-for-2 failure on the exact failure mode this
   wave's judge-prompt change was meant to help with.
2. Root cause, read directly from the two missed rows: both are percentage/growth-change arithmetic
   questions, and the current judge prompt forces `fully_supported=True` whenever a sentence's
   `supporting_sentence_keys` includes `numerical_reasoning` — regardless of whether the arithmetic is
   correct. This blinds the judge to exactly the "generator did the math wrong" failure mode this
   project's earlier entries (finqa `v4`, rows 11/13) already identified as a real, recurring
   hallucination cause.
3. relevance/utilization/completeness RMSE are all worse than `tatv3`/`tatv4` too (0.268/0.295/0.388
   vs. `tatv4`'s 0.205/0.164/0.389) — per the `v5`/`tatv4` generation-noise-floor finding, some of this
   is expected run-to-run variance, not necessarily config-attributable.

#### Inference

The judge-prompt change made for this wave has a specific, identified bug: the `numerical_reasoning`
forced-true carve-out is too broad — it should confirm a sentence *is reasoning about numbers*, not
assert that the reasoning is *correct*. This is a clean, isolable fix (narrow the forced-true rule to
exclude `numerical_reasoning`), not a sign the judge-prompt-engineering approach itself is wrong.

#### Next-Config Decision

**Hypothesis:** narrowing the forced-`fully_supported=True` rule so it no longer applies to
`numerical_reasoning`-tagged sentences (only to genuinely non-factual scaffolding — `general`,
`well_known_fact`, `supported_without_sentence`) will let the judge catch arithmetic-based
hallucinations like rows 54/71 while still forgiving harmless scaffolding sentences.

**Proposed test (`tatv8`):** identical config to `tatv7`; only the judge-prompt template (section 8)
changes. Section 10's checkpoint logging now also captures `judge_sentence_support_information` per
row, so this class of issue can be verified directly from the checkpoint on the next run instead of
inferred from question text alone.

**Confirm criteria:** hallucination cases like rows 54/71 (if redrawn) get caught
(`pred_adherence=0.0`), and `adherence_aucroc` recovers toward or past `tatv4`'s 0.617, without a large
increase in false "unsupported" flags on genuinely correct responses.

**Falsify criteria:** the narrowed rule still misses genuine arithmetic hallucinations, or over-corrects
by flagging many more genuinely correct `numerical_reasoning` answers as unsupported (mirroring the
`v4`-era rejected fix that forgave too much in the opposite direction).

**Status: registered in the notebook (`tatv8`, `ACTIVE_CONFIG_NAME="tatv8"`) — not yet run.** `tatv8`'s
checkpoint folder exists on Drive but is empty (created 2026-08-02, no `.jsonl` written yet).

---

### Iteration: tatv8 (narrowed `numerical_reasoning` judge-prompt rule) — fix did not work; root cause mis-diagnosed in `tatv7`

#### Status

Completed, 77/80 rows (same 3-row dropout pattern as `tatv7`). Verified directly against
`G:\My Drive\ragbench_v2_run\tatv8\checkpoints\tatqa_2de4f28d.jsonl`.

#### Exact Config Used

Identical to `tatv7` except the judge prompt (section 8): the forced-`fully_supported=True` rule for
non-factual scaffolding tags now explicitly excludes `numerical_reasoning` — "a calculation can be
arithmetically wrong even when it is clearly a numerical-reasoning step, so the Judge must still be
free to mark a `numerical_reasoning` sentence as `fully_supported=false` when the math itself does not
check out against the documents."

#### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa | 77 | 0.282839 | 0.266931 | 0.353713 | **0.4267** |

| metric | tatv7 (n=77) | tatv8 (n=77) | change |
|---|---|---|---|
| relevance_rmse | 0.267712 | 0.282839 | +5.7% (worse) |
| utilization_rmse | 0.295281 | 0.266931 | -9.6% (better) |
| completeness_rmse | 0.387521 | 0.353713 | -8.7% (better) |
| adherence_aucroc | 0.440 | 0.427 | ~flat, still below-random |
| adherence mismatches | 11/77 | 13/77 | slightly worse |

`gt_adherence` class balance: 75 supported / **2 hallucinated** — the same two questions as `tatv7`
(Supplier A inventory-purchase percentage change; December-2018 operating-income growth).

#### Row-Level Detail: the fix never engaged on either target row

Read directly from `judge_sentence_support_information` for both hallucination rows (row 57, row 72):

- **Row 57** (Supplier A %, still `pred_adherence=1.0`): the sentence performing the actual arithmetic
  ("the arithmetic yields -2.94%, which is correct") is tagged `supporting_sentence_keys: ["0a"]` —
  **not** `"numerical_reasoning"`. The judge cites the raw source sentence directly and independently
  affirms the derived figure as correct, rather than routing through the tag `tatv8`'s fix targeted. The
  narrowed rule can only stop the judge from *rubber-stamping* a `numerical_reasoning`-tagged sentence —
  it does nothing when the judge never applies that tag in the first place and instead (incorrectly)
  verifies the derived claim by citing the underlying data sentence.
- **Row 72** (operating income, still `pred_adherence=1.0`): the response is actually a refusal — it
  concludes the needed Sep-2018/Dec-2017 figures are missing and growth "cannot be calculated." Every
  sentence is tagged `"general"` or `"supported_without_sentence"`, which the prompt's own rule requires
  to be `fully_supported=true` unconditionally (this exact category is explicitly defined for "expressing
  inability to answer... due to lack of relevant information"). The judge is behaving exactly as
  instructed here; the disagreement with `gt_adherence=0` is a **ground-truth/reference-response
  mismatch**, not a judge error to fix.

#### Observations

1. **The targeted fix had zero effect on its target failure mode** — both known hallucination cases are
   still missed, identically to `tatv7`.
2. Root cause is more specific than `tatv7`'s hypothesis: on row 57 the judge's fact-verification is the
   actual failure (accepting a cited source sentence as support without checking whether the response's
   restated numeric premise literally matches it), not a `numerical_reasoning`-tag rubber-stamp — the tag
   was never applied. Row 72 is a separate, likely-unfixable case: correct judge behavior per its own
   instructions, disagreeing with RAGBench's own ground truth for structural reasons already documented
   (gt scored against RAGBench's own different reference response).
3. **New finding: `temperature=0.0` does not eliminate generation noise.** Comparing `tatv7` and `tatv8`
   (identical config apart from the judge prompt, both `temperature=0.0`, same `random_seed=42`; the code
   correctly threads `cfg.generation_temperature` into the Groq call, confirmed in section 7 — this isn't
   a wiring bug) — only **20 of 77 (26%)** shared questions produced byte-identical response text between
   the two runs. This extends the `v5` entry's finding (the judge alone isn't fully deterministic at
   `temperature=0.0`) to the generator as well: pinning `temperature=0.0` reduced but did **not**
   eliminate the run-to-run noise floor this project has been managing since `v5`/`tatv4`.
4. relevance regressing slightly and mismatches rising (11→13) are both consistent with being generation
   noise (per Observation 3) rather than a judge-prompt regression — utilization and completeness moving
   the *opposite* direction (better) on the same pair of runs supports reading these deltas as noisy, not
   as a clean effect of the prompt change either way.

#### Inference

The `tatv7` hypothesis (a `numerical_reasoning` tag being rubber-stamped) was a plausible but incorrect
diagnosis for these two specific rows — the actual failure is the judge trusting a response's restated
numeric premise without cross-checking it against the literal cited source text. Row 72 is not a judge
bug at all. And the newly confirmed generation non-determinism at `temperature=0.0` means the last two
iterations' RMSE deltas (`tatv7`→`tatv8`) are noisier than they'd appear from the numbers alone — a
third, independent confirmation (after `v5`, `tatv4`) that this pipeline's single-run deltas need a noise
caveat, now extended to the generator itself, not just the judge.

#### Next-Config Decision

**Hypothesis:** catching row-57-style failures needs a fact-verification instruction — explicitly telling
the judge to cross-check any specific numeric values a response sentence claims are "from" a cited
document sentence against that sentence's literal text — not another adjustment to the
`numerical_reasoning` tag rule, which this entry shows the judge doesn't reliably apply to this failure
class in the first place.

**Worth considering as a code-level alternative, not another prompt tweak:** a cheap deterministic
post-hoc check — extract numeric values from each response sentence and verify they appear in (or are a
one-step arithmetic derivation from) the numbers in that sentence's cited `supporting_sentence_keys` text
— as a supplementary hard signal alongside the LLM judge, since LLM judges are known to be unreliable at
precise numeric cross-checking. This would need the same validation-before-trusting discipline as the
`v4`-era rule-based fix that was tested and explicitly rejected, not assumed to work.

**Falsify criteria:** if a fact-verification-instruction change still fails to catch row-57-style cases,
the more likely explanation is that Doc 0's actual text doesn't contain the exact figures being cited at
all (a retrieval/generation fabrication, not a judge-verification gap) — worth checking directly against
the retrieved document text before writing a third prompt iteration.

**Status: diagnosis only, not yet implemented.** `tatv8` remains the active config in the notebook; no
further run yet.

---

### Iteration: tatv9 (fact-verification judge-prompt rule) — engaged correctly, but the real cause was RAGBench's own ground truth, not the judge

#### Status

Completed, 76/80 rows. Verified directly against
`G:\My Drive\ragbench_v2_run\tatv9\checkpoints\tatqa_a383d556.jsonl`.

#### Exact Config Used

Identical to `tatv8` except the judge prompt (section 8): added a rule, independent of any
`supporting_sentence_keys` tag, requiring the judge to verify any quantitative value a response sentence
attributes to a cited document sentence against that sentence's literal text before marking
`fully_supported=true`.

#### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa | 76 | 0.264564 | 0.271560 | 0.376794 | **0.439189** |

`gt_adherence` class balance: 74 supported / **2 hallucinated** — the same two questions as `tatv7`/`tatv8`
(Supplier A %, operating-income growth). 11/76 adherence mismatches.

#### Row-Level Detail: the fix worked as designed, and that's exactly what exposed the real cause

Read directly from `judge_sentence_support_information` for row 53 (Supplier A): unlike `tatv8`, the judge
now genuinely performs the verification instruction — its explanation for the sentence restating the
premise values states: *"These values are exactly those shown in sentence 0a of Document 0, which records
Supplier A as 34% (2017), 33% (2018) and 31% (2019)."* This is a real literal cross-check, not a
rubber-stamp — and it's **correct**: pulling the actual RAGBench tatqa document for this question directly
from the `galileo-ai/ragbench` dataset (zero API cost) confirms the source table says exactly that:
Supplier A inventory purchases 2017=34%, 2018=33%, 2019=31%. Our pipeline's response computed
`(33-34)/34 = -2.94%` from those figures — correct arithmetic on a correct, verified premise.

Pulling RAGBench's own reference response for this exact question (the response `gt_adherence=0` was
actually scored against) reveals the real problem:

> "...taking the difference between the percentages in 2017 (34%) and 2018 (31%)... (31%-34%)/34% =
> -0.0882 = -8.82%. So, the percentage change is -3%."

This reference response uses **31% (2019's value) where it needed 33% (2018's value)**, and its own final
stated answer (-3%) doesn't even match its own displayed arithmetic (-8.82%) — it's internally broken. The
`gt_adherence=0` label on this row was computed against *that* reference, not a correct one — our
pipeline's actually-correct response was never going to score as "supported" against ground truth built on
a wrong answer.

Row 70 (operating income) has its own version of the same problem: the real document states Q4 2019
operating income ($460M) grew sequentially from Q3 2019 ($336M, +$124M) and year-over-year from Q4 2018
($443M, +$17M). RAGBench's own reference response states growth was "$17 million" on **both** a sequential
and year-over-year basis — repeating the YoY figure for the sequential one, arithmetically wrong (sequential
should be $124M). This row is messier than row 53, though: our pipeline's own response is *also* not fully
correct here — it misreads the confusingly-worded question ("December 2018... sequential and
year-over-year") and concludes the needed comparison quarters are "missing" from the documents when they're
actually present under different labels (Sep 2019, Dec 2018) — a genuine retrieval/reasoning slip on our
side, not just a ground-truth artifact. But even a hypothetically perfect response here would still be
scored against a broken reference.

#### Observations

1. **The fact-verification instruction worked exactly as designed** — the judge is now genuinely comparing
   cited numbers against source text (confirmed from its own explanation text), a real improvement in judge
   behavior over `tatv8`, where the instruction never engaged at all.
2. **`adherence_aucroc` still didn't move (0.439, essentially flat vs. `tatv7`'s 0.440 and `tatv8`'s 0.427)**
   — not because the fix failed, but because there was nothing left on these two specific rows for it to
   fix. Row 53's `gt_adherence=0` label is verifiably wrong: our response is correct, and RAGBench's own
   reference response (which the label was scored against) contains a real arithmetic/data error (wrong
   year's value, and a final answer that doesn't match its own shown math).
3. Row 70 is a genuine mixed case: RAGBench's reference is also demonstrably wrong (repeats one growth
   figure for both bases), but our own pipeline's response has an independent, real flaw (misidentifying
   which quarters the question needs) — not something a judge-prompt fix could address, since it's a
   retrieval/reasoning issue in the generator's own response, not a judge-scoring issue.
4. This is a stronger, directly-evidenced version of this project's long-standing documented limitation
   ("ground truth scored against RAGBench's own different reference response") — previously framed as "two
   valid-but-different responses can disagree"; this entry proves the reference itself can simply be
   **wrong**, not just different.

#### Inference

Three consecutive iterations (`tatv7`, `tatv8`, `tatv9`) chased "the judge misses these two hallucinations"
as if it were a fixable judge-prompt problem. It wasn't, at least for row 53: verified directly against
RAGBench's own source documents and reference responses (zero API cost), our pipeline's response was
correct and RAGBench's own ground-truth reference was not. No judge-prompt change could ever have flipped
this row correctly, because doing so would require the judge to call a correct, source-verified response
"unsupported" to match a broken reference — the opposite of what a good judge should do. Row 70 is
genuinely mixed (a real flaw in our response, plus an also-wrong reference), so it doesn't cleanly support
or refute the judge-prompt approach either way.

At n=76-80 in this specific `random_seed=42` draw, both of the only two available `gt_adherence=0` examples
rest on questionable ground truth — meaning `adherence_aucroc` on this exact sample was never capable of
validating or invalidating the judge-prompt fixes tested across `tatv7`-`tatv9`, independent of how good or
bad the judge actually is. This is a sample-composition problem at the ground-truth level, not a judge,
retrieval, or generation defect.

#### Next-Config Decision — concluding the second tatqa wave here

**Decision: stop iterating on the judge prompt for this specific failure mode.** Further prompt engineering
targeting these two rows would be chasing dataset noise, not a real weakness — confirmed by direct
primary-source verification, not assumed.

**`tatv3` (n=15) remains the adopted, presented tatqa configuration**, unchanged by this entire second wave
(`tatv5_best`-`tatv9`). The temperature-pinning and judge-prompt-fix experiments are retained as valuable
diagnostic history (they surfaced the generation-noise-floor finding and the fact-verification behavior
improvement), not as configuration changes that beat `tatv3`.

**If a cleaner read of judge quality on hallucination detection specifically is still wanted:** the real
lever is a much larger `n` (to dilute the chance that *every* available hallucination example in a sample
happens to rest on a broken RAGBench reference), not further prompt iteration. This should be weighed
against the real, non-refundable Groq-budget cost already spent across five runs (`tatv5_best` through
`tatv9`) chasing this same gap.

**Known limitation, upgraded from earlier entries:** "ground truth scored against RAGBench's own reference
response" is not merely a two-valid-answers-can-differ caveat — confirmed here with a concrete example,
RAGBench's own reference responses can themselves contain real arithmetic and data errors, so a low raw
`adherence_aucroc` at small `n` should never be read as proof of pipeline or judge failure without checking
the underlying reference response first.

---

### Iteration: tatv9 (checkpoint completed further — 78/80 rows, up from 76/80) — plateau confirmed at higher completion

#### Status

Same `tatv9` config and checkpoint file, resumed and picked up 2 more rows since the entry above (76→78/80;
2 rows still never complete even after 6 retries — a small residual dropout, not a new run). Verified by
loading `G:\My Drive\ragbench_v2_run\tatv9\checkpoints\tatqa_a383d556.jsonl` directly (78 lines) and
recomputing every metric independently with the same `rmse`/`roc_auc_score` formulas as section 11, rather
than transcribing the notebook's printed summary table — consistent with this file's standing methodology
correction (see top of file). The recomputed figures match the notebook's own printed output exactly.

#### Exact Config Used

Unchanged from the `tatv9` entry above — same config name, same fingerprint (`a383d556`), no code or
prompt change. This is the same run reaching a later checkpoint state, not a new iteration.

#### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| tatqa | 78 | 0.263829 | 0.269132 | 0.404937 | **0.440789** |

| metric | tatv9 (76/80) | tatv9 (78/80) | Δ |
|---|---|---|---|
| relevance_rmse | 0.264564 | 0.263829 | −0.0007 (flat) |
| utilization_rmse | 0.271560 | 0.269132 | −0.0024 (flat) |
| completeness_rmse | 0.376794 | 0.404937 | +0.028 (worse) |
| adherence_aucroc | 0.439189 | 0.440789 | +0.0016 (flat) |
| adherence_mismatches | 11/76 (14.5%) | 11/78 (14.1%) | flat |

`gt_adherence` class balance unchanged: 76 supported / 2 hallucinated — the same two rows as every prior
entry (index 53, Supplier A %; index 70, operating-income growth), same wrong-direction judge call on both
(`pred_adherence=1.0` against `gt_adherence=0`), same root cause already established above (broken RAGBench
reference responses, not a judge defect).

#### Row-Level Detail

Re-extracted all 11 mismatches directly from the 78-row checkpoint: the same 9 `gt=1.0`/`pred=0.0`
scaffolding-leniency mismatches as the 76-row entry, plus the same 2 `gt=0.0`/`pred=1.0` rows (53, 70). The
2 newly-completed rows are both `gt_adherence=1.0` and scored correctly by the judge — they add signal to
relevance/utilization/completeness only, which is why `adherence_aucroc` is flat to the 3rd decimal while
`completeness_rmse` moved.

#### Observations

1. Completing 2 more rows changed nothing about the central finding: `adherence_aucroc` stays pinned at
   ~0.44 because the only 2 hallucination examples available in this seed-42 draw rest on verifiably-wrong
   RAGBench ground truth (already primary-source-verified above), not judge quality. Additional rows
   landing on the `gt_adherence=1.0` side can't move that number.
2. `completeness_rmse`'s move (0.377→0.405) is fully attributable to the 2 newly-completed rows' own
   completeness gap — at n≈77-80 this metric still swings by ~0.03 from 2 rows, i.e. it hasn't stabilized
   even at over 5x `tatv3`'s sample size.
3. relevance_rmse/utilization_rmse (~0.26/~0.27) are now confirmed stable across three independent
   checkpoints at this sample size (`tatv7` 77-row: 0.268/0.295; `tatv8` 77-row: 0.283/0.267; `tatv9`
   76-and 78-row: 0.265/0.272 and 0.264/0.269) — a real, sample-composition characteristic of the larger,
   less cherry-picked n≈80 draw, not run-to-run noise, and not directly comparable to `tatv3`'s n=15
   (0.157/0.160), which reflects a much smaller slice.
4. Dropout is now down to 2/80 (2.5%) vs. `tatv6`'s 24/80 (30%) and `tatv7`/`tatv8`'s 3/80 — the
   `max_retries=6`/`request_min_interval_s=6.0` fix from `tatv7` continues to hold. But this exact config
   has never reached a clean 80/80 across three consecutive completions (77, 77, 78), which points to a
   small residual (~2-3 row) *hard* failure — e.g. a persistently malformed judge response for those
   specific rows — distinct from the transient rate-limiting `tatv7` fixed, and not addressed by more
   retries alone.

#### Inference

A confirmation, not a new finding: every metric moves by noise-level amounts except `completeness_rmse`,
whose ~0.03 shift is fully explained by the 2 specific new rows. The `tatv7`→`tatv9` conclusion stands at
higher completion: `adherence_aucroc` on this exact seed-42 draw was never capable of validating or
invalidating the judge-prompt line of fixes, because both available positive (hallucinated) examples rest
on broken reference responses — and no amount of further checkpoint completion on *this* sample changes
that, since the hallucinated-row set is fixed by the sampling seed, not by how many rows finish.

#### Next-Config Decision

**No change.** Reconfirms the standing decision: **`tatv3` (n=15) remains the adopted TatQA
configuration**; `tatv5_best`→`tatv9` stays diagnostic history, not a replacement. Two forward options
remain open and untaken (no further Groq budget spent chasing this specific gap since `tatv9`):
- Scale `n` well beyond 80 with a *different* seed to get hallucination examples not resting on broken
  references — the only way left to get a clean read on judge quality specifically.
- Investigate the residual ~2-3/80 hard-failure rate (consistent across `tatv7`, `tatv8`, `tatv9`) as a
  distinct, smaller issue from the `tatv6` rate-limit dropout already fixed — worth inspecting what those
  specific rows' API responses actually return before spending more budget on blind resumes.

---

## Revisiting FinQA: Testing the Evolved Judge Prompt + Random Sampling at Scale (`finqa_v10`)

`v4` closed the finqa track at n=15 (`first_n` sampling, default `generation_temperature=0.2`, the
pre-`tatv7` judge prompt). Since then, the tatqa second wave changed three things never tested on
finqa: `generation_temperature` pinned to `0.0`, `sample_strategy` switched to `random` (seed 42) at
much larger `n`, and the judge prompt itself (section 8) evolved through `tatv7`→`tatv9`. `finqa_v10`
applies all of these to finqa for the first time, bundling **four** changes from `v4` at once —
**flagged as a real confound, not hidden**, the same discipline applied to `tatv5_best`'s bundled
changes.

### Iteration: finqa_v10 (random n=40, temp=0.0, evolved judge prompt) — `adherence_aucroc` collapses to 0.375, three distinct causes identified

#### Status

Completed, 39/40 rows. Verified directly against
`G:\My Drive\ragbench_v2_run\finqa_v10\checkpoints\finqa_b5b941e6.jsonl` (39 lines) — every headline
figure recomputed independently from the raw checkpoint (same `rmse`/`roc_auc_score` formulas as
section 11) and matched the notebook's printed summary exactly.

#### Exact Config Used

| Parameter | `v4` (adopted) | `finqa_v10` |
|---|---|---|
| `max_samples_per_domain` | 15 | 40 (39 completed) |
| `sample_strategy` | `first_n` | `random`, seed 42 |
| `generation_temperature` | 0.2 (default) | **0.0** (pinned) |
| Judge prompt (section 8) | pre-`tatv7` version | **`tatv9`-evolved version** (fact-verification rule added) |
| `top_k_retrieve` / `top_k_final` | 20 / 5 | 20 / 5 (unchanged) |
| `embed_model` / `judge_model` | `bge-large` / `gpt-oss-120b` | unchanged |

**Confound: four variables changed simultaneously relative to `v4`.** No single number below can be
cleanly attributed to one cause from this run alone.

#### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|
| finqa | 39 | 0.162128 | 0.162032 | 0.308706 | **0.375** |

| metric | `v4` (n=15, adopted) | `finqa_v10` (n=39) | Δ |
|---|---|---|---|
| relevance_rmse | 0.122 | 0.162 | +33% worse |
| utilization_rmse | 0.112 | 0.162 | +45% worse |
| completeness_rmse | 0.186 | 0.309 | +66% worse |
| adherence_aucroc | 0.786 | 0.375 | −52%, now **below random guessing** |

`gt_adherence` class balance: 36 supported / 3 hallucinated. **All 3 hallucinated rows were missed**
— `pred_adherence=1.0` on every one (0% recall) — plus 9 further `gt=1.0`/`pred=0.0` mismatches
(12/39 mismatches total, 31%).

#### Row-Level Detail — three hallucinated rows, three distinct causes

Read directly from `judge_sentence_support_information` for all 3 `gt_adherence=0` rows:

- **Row 7 (London Market share of net reserves), `gt_completeness=1.0`:** the response computes
  57/367 ("total gross reserves"); the question asks for the portion of *net* reserves (320, stated
  in the same source sentence). Every individual number cited is real and judge-verified against
  source text — the response silently uses the wrong base figure for what was asked. The `tatv9`
  fact-verification rule checks "does this number appear in the cited sentence," not "is this the
  right quantity for the question" — a premise-selection error invisible to text-matching
  verification.
- **Row 34 (2007→2008 rent increase), `pred_relevance=pred_utilization=0.0`:** the response is an
  outright refusal ("no document mentions rent increase... cannot calculate"). The judge correctly
  verifies this claim is true of *what was retrieved* — no retrieved sentence mentions rent — and
  marks every response sentence `fully_supported=True`. But `gt_adherence=0` implies the reference
  answer does derive a value, meaning the needed fact exists in the corpus but fell outside
  `top_k_final=5` for this question. This is a **retrieval-recall miss disguised as a
  hallucination-detection failure**: a judge grounded only in retrieved context is structurally
  unable to flag "the retriever failed to surface the answer," independent of prompt quality.
- **Row 38 (nuclear realized price as % of net revenue decrease):** the response computes
  (−194)/(−191)×100 = 101.57%; the judge verifies both source figures ($2045M/$1854M and −$194M, both
  in doc 1a) and confirms the arithmetic is correct. This matches the signature already documented for
  tatqa `tatv9` (rows 53/70): a judge-verified-correct response scored against `gt_adherence=0` —
  most likely because RAGBench's own reference computed this differently (and, per the tatqa
  precedent, possibly incorrectly). Not independently re-verified against the raw RAGBench reference
  text here, unlike the tatqa deep-dive, but close enough to flag rather than dismiss.

#### Observations

1. **0% hallucination recall (0/3) is a real, severe number, but not proof of a single failure** — the
   3 misses split three ways: a genuine premise-selection blind spot in the fact-verification rule
   (row 7, arguably fixable), a retrieval-recall failure the judge cannot see by construction (row 34,
   not a judge problem at all), and a suspected broken-reference case matching the tatqa `tatv9`
   pattern (row 38, likely not fixable via judge or pipeline changes).
2. **The retrieval-miss-as-invisible-hallucination pattern (row 34) is new** — not previously
   catalogued in either track. It's structural: any judge grounded only in retrieved context will
   rubber-stamp a truthful-given-its-inputs refusal, even when the true cause is a retrieval failure.
   Distinct from the scaffolding-sentence leniency issue, which is about tone/genericity, not
   retrieval recall.
3. **relevance/utilization/completeness RMSE are all substantially worse than `v4`** (+33% / +45% /
   +66%) — this closely mirrors TatQA's move from `tatv3` (n=15) to the n≈80 second wave
   (relevance/utilization roughly doubled there too). The pattern now holds in **both domains**:
   small, `first_n` or small-random samples produce meaningfully more optimistic RMSE than larger
   random samples. Both "final" scoreboards (`v4`, `tatv3`) are best read as encouraging results on a
   favorable, undersized slice, not settled corpus-wide performance.
4. **Four-variable confound means the specific 0.786→0.375 delta can't be attributed to one change.**
   Given tatqa's own second wave already showed the evolved judge prompt alone doesn't explain its
   AUCROC plateau (broken ground truth did), and finding 3 above plausibly explains the RMSE moves via
   sampling alone, the most defensible reading is that sampling composition (small favorable slice →
   larger, harder, unfiltered one) dominates this collapse — but that is inference, not proof, without
   an isolating re-run.

#### Inference

The same class of finding as tatqa's second wave, arriving from a different angle:
`adherence_aucroc` measured at small, favorable-sample `n` does not reliably predict performance at
scale, and low-`n` "final" configs (`v4`, `tatv3`) should be read as encouraging early signals, not
settled performance numbers. This run also surfaces a genuinely new, structurally distinct failure
mode — retrieval-miss-as-invisible-hallucination — that no amount of judge-prompt tuning can fix,
since the judge only ever sees what retrieval handed it.

#### Next-Config Decision

**Do not adopt `finqa_v10` over `v4`** — the four-variable confound makes it unusable as a direct
comparison, and `v4` remains the presented FinQA configuration. Two isolating follow-ups, in priority
order:
1. **Re-run at `v4`'s exact settings except `sample_strategy=random`/`max_samples_per_domain=40`**
   (temperature and judge prompt held at `v4`'s original values) — isolates whether the RMSE/AUCROC
   drop is sampling-driven (this project's leading hypothesis) or judge-prompt-driven.
2. **Investigate row 34's class specifically (retrieval-miss-driven refusals) across a larger sample**
   — if this pattern recurs, it argues for either raising `top_k_final` for FinQA (previously untested
   as a lever, unlike TatQA, where it was directly tested and proven) or explicitly flagging refusal
   responses for separate handling, since the judge cannot resolve them by design.

---

## Root-Cause Finding: `sample_seed` Does Not Actually Reproduce a Sample (`tatv3_recheck`)

### Iteration: tatv3_recheck (same seed=42, n=15, domain=tatqa as `tatv3`) — drew a completely different 15 questions

#### Status

Completed, 15/15 rows. Verified directly against
`G:\My Drive\ragbench_v2_run\tatv3_recheck\checkpoints\tatqa_06e0dd5d.jsonl`, compared row-by-row
against the original `tatv3` checkpoint
(`G:\My Drive\ragbench_v2_run\tatv3\checkpoints\tatqa_75501464.jsonl`).

#### Exact Config Used

Nominally identical sampling parameters to `tatv3`: `domains=("tatqa",)`, `max_samples_per_domain=15`,
`sample_strategy="random"`, `sample_seed=42`. (`generation_temperature=0.0` differs from `tatv3`'s
default 0.2 — irrelevant to *which* rows get sampled, only to what the generator writes for them.)

#### Results

| domain | n | relevance_rmse | utilization_rmse | completeness_rmse | n_hallucinated | n_supported | adherence_aucroc | adherence_mismatches |
|---|---|---|---|---|---|---|---|---|
| tatqa | 15 | 0.186158 | 0.115479 | 0.298752 | 0 | 15 | NaN (0 hallucinated rows drawn) | 2 |

#### The actual finding: this was never a reproducibility check

Comparing the 15 questions in `tatv3_recheck` against `tatv3`'s original 15 (both `seed=42`, both
`n=15`, both `domain="tatqa"`) shows **zero overlap** — every single question differs:

| # | `tatv3` (original, 12 Jul) | `tatv3_recheck` (7 Aug) |
|---|---|---|
| 0 | "What is the percentage change in capitalized interest..." | "How much are the total compensations for Richard S. Hill..." |
| 3 | "Which assets had non-material adjusted carrying values?" (`gt_adherence=0`) | "What is the increase / (decrease) in 28 nanometers..." |
| 5 | "What is the average age of the company's Vice Presidents?" (`gt_adherence=0`) | "What was the percentage change in Accruals..." |
| ... | (all 15 differ) | (all 15 differ) |

`select_eval_rows` (section 10) implements sampling as
`random.Random(cfg.sample_seed).shuffle(rows); rows[:n]`. `random.Random(42)` is deterministic —
given the *same input list order*, this always produces the same permutation. The only way two runs
with identical `seed`/`n`/`domain` draw disjoint samples is if the **input `rows` list itself had a
different order** at shuffle time between the two runs. Two candidate causes, not yet distinguished:

1. `select_eval_rows`'s implementation (or whatever preceded it) changed between `tatv3`'s original run
   (12 Jul, per the `tatv3/` folder timestamp) and now (7 Aug) — very plausible given how much this
   notebook has been edited in that window (the `tatv5_best`→`tatv9`→`finqa_v10` sequence alone
   rewrote section 8's judge prompt three times and added `judge_sentence_support_information`
   logging to section 10, the same cell `select_eval_rows` lives next to).
2. `load_dataset("galileo-ai/ragbench", "tatqa", split="test")`'s row order isn't guaranteed stable
   across HF Hub/`datasets`-library cache states (re-downloads, cache eviction, library version
   changes) — if the *upstream* row order shifted, a byte-identical `shuffle(42)` call still yields a
   different final sample.

Both are consistent with the same underlying gap: **`sample_seed` alone does not pin a sample.**
Nothing in this pipeline captures or persists *which specific question IDs* were drawn for a given
named config — only the seed integer, which is necessary but not sufficient for reproducibility
across time.

#### Observations

1. **This invalidates "same seed" as a reproducibility guarantee for any comparison spanning a code
   change** — which is most of this project's history. Within a tight cluster of runs on unchanged
   code (`tatv7`→`tatv8`→`tatv9`, all within the same session/day range) the shared seed did keep
   drawing the same 2 hallucination rows, consistent with each other — so the mechanism isn't broken
   in general, only across the kind of notebook edits that happened between `tatv3` (12 Jul) and
   `tatv3_recheck` (7 Aug).
2. **This is now the 4th independent tatqa sample to draw zero `gt_adherence=0` rows** (`tatv1`,
   `tatv2` — both `first_n` — `tatv5_best` n=30, and now `tatv3_recheck` n=15) — further evidence that
   tatqa's true hallucination rate (~7-13%, per the second-wave entries) makes `adherence_aucroc`
   structurally fragile at n≤30 regardless of sampling strategy, independent of today's reproducibility
   finding.
3. **The 2 mismatches found are a mixed bag, not a clean repeat of the scaffolding pattern:** row 6
   ("diluted net income per share") is the familiar pattern — the judge demands a near-literal source
   quote for a correctly-paraphrased formula. Row 13 ("valuation allowance for deferred tax assets") is
   different: the judge's objection is that the response falsely claims the documents lack a figure
   that document `0b` actually contains, and pads the final answer with several unrelated dollar
   figures beyond what was asked — plausibly a genuine generator imprecision, not judge over-strictness,
   despite `gt_adherence=1.0`. Worth noting as a caution against assuming every gt=1/pred=0 mismatch is
   automatically "the judge being too strict" without reading it.
4. **relevance/utilization/completeness RMSE moved in mixed directions vs. `tatv3`** (relevance worse:
   0.186 vs 0.157; utilization better: 0.115 vs 0.160; completeness better: 0.299 vs 0.397) — exactly
   what's expected from two disjoint, unrelated 15-question samples, not evidence of anything about the
   config itself. These numbers say nothing about whether `tatv3`'s config is good or bad; they're
   incomparable, full stop.
5. **Secondary finding, section 9.1's free formula-validation cell:** re-run alongside this, comparing
   RAGBench's own stored TRACe scores to values recomputed directly from RAGBench's own annotation
   fields (no LLM involved) across all rows in whatever `RAW_ROWS` held at execution time. Mean absolute
   difference was small across the board (relevance 0.013, utilization 0.006, completeness 0.007,
   adherence exact-match 99.9%) — the formula implementation itself remains validated. But
   `relevance`'s **max** abs diff was 8.0 — a single-row outlier far outside `relevance`'s valid [0,1]
   range on the *dataset*-stored side (the recomputed side is mathematically bounded to [0,1] by
   construction, so the anomaly is in RAGBench's own stored `relevance_score` for that one row, not in
   this pipeline's formula). Low priority (doesn't affect any reported RMSE, since actual pipeline runs
   regress against per-row ground truth as stored, not this validation path) but worth a five-minute
   look to identify the offending row id. Separately, the printed table's `id` column showed
   `finqa_test_*` rows while `ACTIVE_CONFIG_NAME="tatv3_recheck"` (tatqa-only) was active — `RAW_ROWS`
   still held a prior config's finqa data at the moment this cell executed, i.e. **stale global state
   from running notebook cells out of order**, not a data bug. A reminder that any cell reading
   `RAW_ROWS`/`CONFIG` globals needs the Ingestion cells re-run after switching `ACTIVE_CONFIG_NAME`,
   or its output describes the wrong config.

#### Inference

The original goal of `tatv3_recheck` — checking whether `tatv3`'s adopted numbers are stable/reproducible
— was never actually tested, because the "recheck" silently became a fifth independent random draw
instead of a repeat of the same one. This reframes a meaningful slice of this project's history: any
comparison between two named configs that weren't run back-to-back on the same notebook code should be
treated as **comparing two different samples**, not as an isolated single-variable test, unless the
actual sampled question IDs were captured and verified to match (as was done here, after the fact, by
diffing checkpoints directly — which is what caught this).

#### Next-Config Decision

**Priority fix, before any further metric-chasing runs:** persist the actual sampled question list
(e.g., write `[row["id"] for row in eval_rows]` to a small JSON file per config at the top of
`run_domain_evaluation`) so any future "recheck" loads that exact list instead of re-deriving it from
`seed`. This is now the single highest-leverage change available: without it, every future
reproducibility check is at risk of silently repeating today's failure mode, and every historical
comparison between configs separated by more than a few hours of unedited code should be treated as
provisional. This is a code-reliability fix, not a metric-tuning one — it doesn't require any further
Groq budget and directly addresses why recent runs have felt like they're "not converging on
anything": several of them were never comparable to begin with.

**Status update — implemented (code only, not yet run):** `Config` gained a new field,
`sample_source_config: str | None = None`. `select_eval_rows` (section 10) now: (1) writes
`<checkpoint_dir>/<domain>_sample_ids.json` the first time a named config draws its sample, and
reloads from that file on every later run of the same config, instead of re-deriving from
`sample_seed` each time; (2) if `sample_source_config` is set to another config's name, loads *that*
config's manifest instead, for deliberate cross-config isolation reruns (e.g. a future
`tatv3_recheck2` could set `sample_source_config="tatv3"` to genuinely repeat `tatv3`'s 15
questions). The core selection/reuse/error-path logic was verified with an in-memory simulation
(same-config reuse survives simulated upstream row-order drift; cross-config reuse via
`sample_source_config` returns the identical sample; both error paths — missing source manifest, a
previously-sampled id no longer present — raise clearly). The `mkdir`/`write_text`/`read_text` calls
themselves could not be dry-run end-to-end in the local editing environment (a sandbox restriction
unrelated to this logic), but use the same pattern already proven across every checkpoint file this
project has produced (`ensure_dirs`, `append_checkpoint`). **This does not retroactively fix any
past config** (`tatv1`-`tatv9`, `finqa_v10`, `tatv3_recheck` all ran before this existed) — it only
pins samples for configs run from this point forward. Needs a real Colab run to confirm end-to-end
before being treated as fully verified.

---

## Full Checkpoint Audit (Every `.jsonl` on Drive, Both Domains) — Cross-Referenced Against `Task-2 Model Implementation on RAGBench.pdf`

Every checkpoint file under `G:\My Drive\ragbench_v2_run\*\checkpoints\` was loaded directly and every
metric recomputed independently (not read from prior notebook printouts), against the assignment brief
(`Task-2 Model Implementation on RAGBench.pdf`). Two things came out of this pass that hadn't been
written up anywhere before now.

### Task-compliance check (against the PDF brief)

| Requirement (from the brief) | Status | Evidence |
|---|---|---|
| Retriever built over **all** documents in the domain's test split, not just per-question candidates | **Met** | `build_corpus` (section 5) hash-dedupes `row["documents"]` across *every* row in `RAW_ROWS[domain]` before indexing — confirmed by reading the code directly, not assumed. |
| Judge LLM output schema matches RAGBench's own field names (`relevance_explanation`, `all_relevant_sentence_keys`, `overall_supported_explanation`, `overall_supported`, `sentence_support_information`, `all_utilized_sentence_keys`) | **Met, exactly** | `JudgeAnnotation`/`SentenceSupportInfo` Pydantic models (section 8) use these exact field names; `JUDGE_PROMPT_TEMPLATE` is labeled "RAGBench's own annotation prompt (arXiv:2407.11005, Appendix), verbatim" in its own comment. |
| TRACe formulas (Context Relevance, Context Utilization, Completeness, Adherence) match the paper | **Met** | `compute_trace_metrics` (section 9) implements exactly relevance=\|relevant∩valid\|/\|valid\|, utilization=\|utilized∩valid\|/\|valid\|, completeness=\|relevant∩utilized\|/\|relevant\| (1.0 if vacuous), adherence=all sentences `fully_supported`. Independently re-validated against RAGBench's own stored scores in section 9.1 (mean abs diff ≤0.013 across all metrics, 99.9% adherence exact-match). |
| RMSE/AUCROC vs. dataset ground truth | **Met** | `sklearn.metrics.mean_squared_error`/`roc_auc_score`, section 11. |
| Open-source/non-proprietary models only, via Groq (or Ollama) | **Met** | `llama-3.3-70b-versatile` (Meta, open-weight), `openai/gpt-oss-120b` (OpenAI's Apache-2.0 **open-weight** release — not the proprietary GPT API — served via Groq), `BAAI/bge-large-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2` — all open weights, no proprietary/subscription API called anywhere in the pipeline. |
| "Best Practices in RAG" techniques incorporated | **Partially, and documented as tested, not assumed** | Hybrid dense+BM25 with RRF fusion, MMR diversification, cross-encoder reranking are all implemented and were A/B tested (found to make no measurable difference vs. plain dense retrieval in this project — a *tested* negative result, not an omission). Chunking is implemented but unused (`chunking_strategy="none"`) since both domains' documents are short enough to pass through whole. Query rewriting/expansion, HyDE, and re-ranking-model alternatives were never tried — genuine open lever, not yet explored either domain. |
| **"Do not rely on LLM-generated code for evaluation metrics or other critical components — develop and validate independently"** | **⚠ Needs your attention, not mine to certify** | `compute_trace_metrics`, `rmse`, and the `roc_auc_score` usage are the graded-critical pieces — I have *read and cross-checked* them against the RAGBench paper's formulas (section 9.1's self-validation cell agrees with the dataset's own scores to ~0.01), but per the brief you should personally confirm you wrote/understand every line of that cell independently of any AI assistance, since that's what's being assessed. Everything I've *edited* this session (the `sample_source_config` reproducibility fix) is pipeline infrastructure, not a metrics/scoring cell — worth keeping that distinction clear if asked about it. |

### Full metrics table, every tatqa and finqa checkpoint on disk

**tatqa** (chronological):

| Config | n | relevance_rmse | utilization_rmse | completeness_rmse | n_halluc/n_sup | adherence_aucroc | judge prompt era |
|---|---|---|---|---|---|---|---|
| `tatv1` | 15 | 0.171 | 0.174 | 0.202 | 0/15 | NaN | original |
| `tatv2` | 15 | 0.135 | 0.163 | 0.280 | 0/15 | NaN | original |
| `tatv3` (adopted) | 15 | 0.157 | 0.160 | 0.397 | 2/13 | **0.558** | original |
| `tatv4` | 40 | 0.205 | 0.164 | 0.389 | 3/37 | **0.617** | original |
| `tatv3_recheck` | 15 | 0.186 | 0.115 | 0.299 | 0/15 | NaN | *modified* (different sample than `tatv3`, see above) |
| `tatv5_best` | 30 | 0.225 | 0.241 | 0.386 | 0/30 | NaN | *modified* (+scaffolding forced-true) |
| `tatv6` | 56 (of 80, incomplete) | 0.238 | 0.288 | 0.332 | 1/55 | 0.427 | *modified* |
| `tatv7` | 77 | 0.268 | 0.295 | 0.388 | 2/75 | 0.440 | *modified* (narrowed carve-out) |
| `tatv8` | 77 | 0.283 | 0.267 | 0.354 | 2/75 | 0.427 | *modified* |
| `tatv9` | 78 | 0.264 | 0.269 | 0.405 | 2/76 | 0.441 | *modified* (+fact-verification) |
| `tatv10` | 77 | 0.264 | 0.261 | 0.318 | 2/75 | **0.577** | *reverted to original* |

**finqa** (chronological):

| Config | n | relevance_rmse | utilization_rmse | completeness_rmse | n_halluc/n_sup | adherence_aucroc |
|---|---|---|---|---|---|---|
| `v2` (self-judged baseline) | 15 | 0.296 | 0.347 | 0.444 | 1/14 | 0.429 |
| `v3` | 15 | 0.098 | 0.078 | 0.248 | 1/14 | 0.714 |
| `v4` (adopted) | 15 | 0.122 | 0.112 | 0.186 | 1/14 | **0.786** |
| `v5` (run A, n=20) | 20 | 0.107 | 0.060 | 0.366 | 2/18 | 0.167 |
| `v5` (run B, n=33) | 33 | 0.089 | 0.068 | 0.388 | 4/29 | 0.496 |
| `v6` | 10 (incomplete) | 0.100 | 0.107 | 0.158 | 0/10 | NaN |
| `finqa_v10` | 39 | 0.162 | 0.162 | 0.309 | 3/36 | 0.375 |

`v5` and `v6` were previously **not written up anywhere** in this file or in `Final Results Summary.md`
— they exist only as raw checkpoints. Same finding as tatqa's second wave, a fourth time over: large-n
`v5`/`v6` runs score dramatically worse on `adherence_aucroc` (0.167–0.496) than the adopted `v4`
(0.786) — consistent with the small-sample-optimism + broken-ground-truth pattern already established,
not re-litigated in full detail here to avoid duplicating the `finqa_v10` writeup above, which covers
the same territory at a comparable n.

Also present on disk but out of scope for RMSE/AUCROC (different schema, a RAGBench **robustness**
testbed, not the TRACe eval): `rgb_noise_*`/`rgb_negrej_*` files under `tatv4/` and `v5/`/`v6/` — the
tatqa half is already written up above ("RGB robustness testbed"); the finqa `v5`/`v6` noise/negrej
checkpoints are incomplete (6–15 of 15 rows each) and not yet analyzed in this file.

### The one pattern in this table that matters most for "why isn't tatqa improving"

Look at the **judge prompt era** column. Every tatqa run using the *original*, unmodified RAGBench
prompt (`tatv1`-`tatv4`) that has a computable AUCROC scores **0.558–0.617**. Every run using the
*modified* prompt (`tatv5_best` onward — first the scaffolding forced-true rule, then narrowed, then
the fact-verification rule added) scores **0.427–0.441** — a full, consistent band lower, across five
separate runs and three different prompt-editing attempts, none of which closed the gap back to
`tatv4`'s level.

**This has never been isolated.** `tatv4`→`tatv5_best` also changed `generation_temperature` (0.2→0.0,
pinned) and jumped `n` (40→30, then up to 80) at the exact same boundary as the prompt change — three
variables moved together, and the existing `tatv4` writeup (above) already correctly attributes some of
this gap to temperature-driven generation noise (confirmed: 14/15 identical-input responses came back
reworded between `tatv3` and `tatv4` at temp=0.2). But that explanation was reasoned about *before* the
prompt was ever modified — it was never re-examined against the specific, later finding that every
*post*-modification run independently clusters in the same lower band regardless of further prompt
edits (`tatv6` through `tatv9` all land 0.427–0.441 despite two different prompt-fix attempts between
them). Sample size alone doesn't obviously explain it either — `tatv4` (n=40) and `tatv6` (n=56) are
comparable sizes with very different scores. The prompt-modification boundary is the strongest
remaining unexplained variable, and no run in this project's entire history isolates it.

**Update: isolated by `tatv10` below.** Reverting the prompt (holding everything else identical to
`tatv9`) recovered `adherence_aucroc` to 0.577 — back inside this band. See the `tatv10` entry for the
full result and an important caveat: the recovery came with a large new false-positive cost, not a clean
win.

### Iteration: tatv10 (reverted judge prompt) — AUCROC recovers, but at the cost of a new false-positive spike

**Status: completed, 77/80 rows** (same 3-row dropout pattern as `tatv7`/`tatv8`). Checkpoint:
`G:\My Drive\ragbench_v2_run\tatv10\checkpoints\tatqa_c99ece57.jsonl`. Config identical to `tatv9`
(n=80, `temperature=0.0` pinned, `top_k` 25/8, `sample_seed=42`); the only change was reverting
`JUDGE_PROMPT_TEMPLATE` (section 8) to its pre-`tatv5_best`, RAGBench-verbatim form — removing the
scaffolding forced-true rule, the `numerical_reasoning` carve-out, and the fact-verification rule added
across `tatv5_best`→`tatv9`.

**Result:**

| metric | tatv9 (78/80) | tatv10 (77/80) | change |
|---|---|---|---|
| relevance_rmse | 0.264 | 0.264 | flat |
| utilization_rmse | 0.269 | 0.261 | flat |
| completeness_rmse | 0.405 | 0.318 | −22% |
| n_hallucinated / n_supported | 2/76 | 2/75 | same class balance |
| adherence_aucroc | 0.441 | **0.577** | **+31%, recovers toward `tatv3`/`tatv4`'s 0.558–0.617 band** |
| adherence_mismatches | 11/76 (14%) | **27/77 (35%)** | **+145%, worst mismatch rate of any tatqa run in this project** |

**Confirm/falsify verdict: confirmed, with a caveat the confirm criteria didn't anticipate.**
`adherence_aucroc` recovered to 0.577 — inside the original-prompt band (`tatv3` 0.558, `tatv4` 0.617),
clearly out of the modified-prompt band (`tatv6`-`tatv9`, 0.427–0.441). Nine iterations of *adding*
rules to the judge prompt never closed this gap; the first iteration to *remove* them did, on the first
try. The judge-prompt-modification boundary first flagged as an unexplained pattern two entries above is
now isolated as a real, dominant cause of the AUCROC band-split — not temperature or sample composition,
both of which were held constant here versus `tatv9`.

**But the mechanism is not "the original prompt judges better" — read the row-level detail:**
- Of the same two seed-42 gt-hallucinated rows this project has tracked since `tatv7` (the Supplier A
  inventory-purchase-percentage question and the December-2018 operating-income-growth question), `tatv10`
  catches **one** (`pred_adherence=0.0` on operating-income-growth, correctly) where `tatv7`/`tatv8`/`tatv9`
  caught **zero**. The Supplier A row is still missed (`pred_adherence=1.0`) — consistent with this
  project's existing finding that it rests on a broken RAGBench reference, not a judge defect.
- That one extra true positive is the entire reason AUCROC moved — with only 2 positive examples,
  AUCROC is coarse enough that catching 1-of-2 instead of 0-of-2 alone explains most of the jump.
- Of the 27 mismatches, **26 are false positives**: genuinely-supported answers the judge now scores
  `pred_adherence=0.0`. Sample rows: *"What does Cash and Cash Equivalents include?"*, *"What was the
  valuation allowance for certain deferred tax assets in 2019?"*, *"What information does note 3
  provide?"* — short, direct, source-grounded answers, exactly the kind of "procedural/scaffolding"
  sentence the `tatv5_best` rule was originally added to stop the judge from wrongly penalizing. Removing
  that rule didn't just remove leniency on the two real hallucinations — it reintroduced strictness
  against 26 correct answers that have nothing to do with hallucination.

**Key findings:**
1. **The judge-prompt-modification boundary is now a confirmed, not just suspected, cause of the
   `tatv5_best`→`tatv9` AUCROC band-drop** — the first clean isolation of this variable in ten
   iterations.
2. **AUCROC and raw agreement (mismatches) moved in opposite directions for the first time in this
   project's history.** Every prior run had them roughly track together; here, AUCROC improved 31% while
   mismatches worsened 145%. This is only possible because AUCROC is a ranking statistic over 2 positive
   examples, not an accuracy measure over all 77 — a concrete illustration of the fragility this project
   has flagged since `tatv1`/`tatv2`, now shown to bite in the opposite direction (rewarding a config that
   is *less* accurate overall).
3. **Neither prompt version is unambiguously better** — the modified prompt (`tatv6`-`tatv9`) was too
   lenient (missed both real hallucinations, low mismatches elsewhere); the original prompt (`tatv10`) is
   too strict (catches one hallucination, but at 26 false-positive costs). The true fix is narrower than
   either extreme: keep the scaffolding leniency for genuinely non-factual statements, without extending
   it to `numerical_reasoning`-tagged or quantitative content — closer to `tatv8`'s intent, but `tatv8`'s
   specific implementation of that carve-out is already proven (in the `tatv8` entry above) not to have
   engaged on the rows that mattered.

**Decision: not adopted.** `tatv3` (n=15) remains the adopted TatQA configuration — `tatv10`, like the
rest of the `tatv5_best`-onward second wave, is diagnostic history. It closes the "does the prompt
modification explain the AUCROC gap" question (yes) but opens a new, better-specified one: a prompt rule
that protects only *non-factual* scaffolding sentences without either (a) rubber-stamping
`numerical_reasoning` content or (b) penalizing short, correctly-grounded factual answers. That rule has
not yet been written or tested — it is the concrete next step, not another blanket revert or blanket
addition.

**Cost incurred:** one full n=80 run (77/80 completed), Groq budget comparable to `tatv7`/`tatv8`/`tatv9`.
