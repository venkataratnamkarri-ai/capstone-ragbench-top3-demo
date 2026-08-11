# RAG Pipeline Results — FinQA and TatQA

## Overview

A Retrieval-Augmented Generation pipeline built and iteratively evaluated against
[RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) across two financial-domain
sources — **FinQA** (financial-report QA over tables and text) and **TatQA** (tabular + textual
QA) — using RAGBench's own **TRACe** framework:

- **Relevance** — fraction of retrieved-context sentences that were actually relevant to the question
- **Utilization** — fraction of retrieved-context sentences the generated response actually used
- **Completeness** — of the relevant sentences, how many were used (relevant ∩ utilized / relevant)
- **Adherence** — is every sentence in the response fully supported by the context (hallucination check)

Predicted scores are regressed against RAGBench's own ground-truth annotations via **RMSE** (lower
is better, for the three continuous metrics) and **AUCROC** (higher is better, 0.5 = random guessing,
for adherence/hallucination detection). RAGBench's own paper reports 0.51–0.80 AUCROC for comparable
judges.

**Methodology:** one configuration change tested at a time, each justified by a specific hypothesis
with pre-registered confirm/falsify criteria, decided only after reading the previous run's actual
results.

---

## How It Works, In Plain Terms

Think of it as three characters working on every single question, plus a final report-card step.

**1. A Librarian (retrieval) searches.** There's a huge pile of documents — over a thousand, for
finqa — far too many to hand over at once. The librarian searches it two ways at the same time: one
search looks for documents using *similar words* to the question (a keyword search), the other looks
for documents that *mean something similar* even with different wording (a search that understands
meaning, not just words). Both results get combined, and the top handful of documents (5, in the
final configuration) get pulled out.

**2. The student's worksheet gets packed (the "prompt").** Those few documents, the question, and a
set of instructions get glued together into one block of text — literally the exact words sent to
the AI. The instructions say, roughly: *"Here are some documents. Answer the question using only
what's in them. If you can't find the answer, say so — don't make anything up."* The documents
themselves are the **context**; the whole package — instructions + context + question — is the
**prompt**.

**3. The Student writes an answer (generation).** The AI reads the worksheet and writes a response
grounded, hopefully, in only what it was given — e.g. *"Doc 1 shows the value was $100.00 in 2010
and $137.90 in 2011, so the return was 37.9%."*

**4. The Grader checks the work (annotation).** Crucially, this is a *different* AI than the one that
answered — the same way you wouldn't want a student grading their own exam. To make grading precise
instead of vague, every sentence in the documents *and* every sentence in the answer gets numbered
(`1a`, `1b`, `2a`, ...). The grader then answers four questions using those numbers: which document
sentences were actually relevant? Of those, which did the answer actually use? For every sentence the
student wrote, is it backed by something specific in the documents, or invented? And overall — is any
part of this answer made up (hallucinated)?

**5. That grading becomes four numbers per question:**
- **Relevance** = useful document sentences ÷ all document sentences — did the librarian bring back a
  tight, focused set of pages, or pad it with junk?
- **Utilization** = used sentences ÷ all document sentences — how much of what was handed over
  actually got put to work?
- **Completeness** = used-*and*-relevant sentences ÷ relevant sentences — of the stuff that mattered,
  did the student use all of it, or leave something important on the table?
- **Adherence** = did the student make anything up? Yes/no.

**6. Checking against an answer key.** The dataset (RAGBench) doesn't just give questions — for every
one, it already comes with expert-graded versions of these same four scores, done ahead of time when
the dataset was built. So we can directly check: did *our* grader's scores match the *expert's*
scores for the same question?

**7. One report card across all questions.** Do this for every sampled question, then boil it down to
two kinds of summary numbers: **RMSE** for Relevance/Utilization/Completeness — roughly "how far off
was our grader, on average" (lower is better) — and **AUCROC** for Adherence — since it's a yes/no
hallucination flag, this checks whether sorting all answers from "most suspicious" to "least
suspicious" by our grader's opinion actually puts the truly-hallucinated ones near the top (0.5 = no
better than a coin flip, 1.0 = perfect sorting).

**What the `v2` → `v3` → `v4` iterations actually changed**, in this story's terms:
- `v2`: the same AI played both **Student** and **Grader** — like grading your own homework. Turned
  out badly (adherence score below a coin flip).
- `v3`: swapped in a genuinely different AI as the **Grader** — like bringing in an outside teacher
  instead of self-grading. Scores jumped dramatically.
- `v4`: upgraded the **Librarian**'s search tool to a smarter one — helped further, with a small
  tradeoff on two of the four scores.

---

## FinQA Track

### Scoreboard

| Run | top_k retrieve/final | Embedding model | Judge model | Judge sees | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|---|---|---|
| `v2` (baseline) | 20 / 5 | `bge-small-en-v1.5` | `llama-3.3-70b-versatile` (same as generator) | full reasoning | 0.296 | 0.347 | 0.444 | 0.429 |
| `v3` | 20 / 5 | `bge-small-en-v1.5` | `openai/gpt-oss-120b` (independent) | full reasoning | 0.098 | 0.078 | 0.248 | 0.714 |
| **`v4` (final)** | 20 / 5 | `bge-large-en-v1.5` | `openai/gpt-oss-120b` (independent) | full reasoning | **0.122** | **0.112** | **0.186** | **0.786** |

All three runs also share: hybrid dense+BM25 retrieval with RRF fusion, MMR diversity, cross-encoder
reranking (all on), whole-document retrieval units (no chunking), `llama-3.3-70b-versatile`
generator, `long_cot` generation prompt, n=15.

**Net improvement, `v2` → `v4`:** relevance_rmse −59%, utilization_rmse −68%, completeness_rmse
−58%, adherence_aucroc from below-random (0.429) to the top of RAGBench's reported range (0.786).

### `v2` — Baseline

**What was tested:** hybrid+BM25+RRF fusion, MMR diversity, and cross-encoder reranking all enabled
simultaneously, generator and judge sharing the same model (`llama-3.3-70b-versatile`), judge shown
the full chain-of-thought reasoning (not just a parsed final answer).

**Results:** relevance_rmse=0.296, utilization_rmse=0.347, completeness_rmse=0.444,
adherence_aucroc=0.429 (below random).

**Observations:**
- Enabling hybrid search + reranking + MMR simultaneously produced results statistically
  indistinguishable from a plain dense-only retrieval baseline tested separately — retrieval-ranking
  sophistication had no measurable effect here.
- `adherence_aucroc` below 0.5 means the predicted adherence signal was actively anti-correlated
  with ground truth, not just noisy.
- `completeness_rmse` (0.444) was the worst of the three continuous metrics.

**Inference:** Generator and judge sharing a model is a known self-evaluation bias risk, and was
identified as the most likely driver of the adherence gap — prioritized as the next test since it's
the most direct, cleanly isolable explanation.

### `v3` — Independent Judge Model

**What changed:** `judge_model` swapped to `openai/gpt-oss-120b` (a different model family from the
generator). Every other field held identical to `v2`.

**Results:** relevance_rmse=0.098 (−67%), utilization_rmse=0.078 (−78%), completeness_rmse=0.248
(−44%), adherence_aucroc=0.714 (+67%, into RAGBench's reported good-judge range).

**Observations:**
- A single field change (judge model) produced the largest improvement of any test in this project —
  strong, direct evidence of self-evaluation bias in the `v2` configuration.
- Row-level review (using the judge's full annotation, captured at zero extra API cost) showed the
  new judge is inconsistent about marking harmless, non-factual "scaffolding" sentences (e.g. "to
  find X, we need to look at Y") as supported — a distinct, secondary issue from self-bias.

**Inference:** Judge independence was the dominant lever for adherence specifically. The residual
scaffolding-sentence issue was investigated further in `v4`.

### `v4` — Larger Embedding Model (Final)

**What changed:** `embed_model` swapped to `BAAI/bge-large-en-v1.5` (same model family as the
`bge-small` baseline, larger/more accurate). Every other field held identical to `v3`.

**Results:** relevance_rmse=0.122, utilization_rmse=0.112, completeness_rmse=0.186 (−25% vs. `v3`),
adherence_aucroc=0.786 (+10% vs. `v3`, real accuracy improvement — adherence mismatches dropped from
8/15 to 6/15, not just an AUCROC ranking artifact).

**Observations:**
- Completeness and adherence improved further; relevance/utilization RMSE were moderately *worse*
  than `v3` (0.122 vs 0.098, 0.112 vs 0.078) — a genuine tradeoff, not a uniform win.
- Using the judge's captured reasoning, all 6 remaining adherence mismatches were read directly:
  3 were the scaffolding-sentence leniency-inconsistency issue (judge marks "we need to look at Y"
  as unsupported even while tagging it as a general/non-factual statement); the other 3 were cases
  where the *generator itself* made genuine arithmetic errors on multi-step calculations — correctly
  caught by the judge, but scored against RAGBench's own (different, correct) reference response, a
  structural evaluation property rather than a pipeline defect.
- A rule-based automated fix for the scaffolding-sentence issue was tested (trusting the judge's own
  category tags over its `fully_supported` boolean) and explicitly rejected — it degraded AUCROC from
  0.786 to 0.321 by also forgiving a genuinely-hallucinated response.

**Inference / final decision:** `v4` was adopted as the final finqa configuration. Every metric is
dramatically better than `v2`; the remaining gap is explained by two named, distinct causes rather
than being mysterious; further tuning (chunking, another embedding, another generator) would each
cost a real run for uncertain, likely-marginal payoff against an already-strong result.

### Second Wave — Testing at Scale with Random Sampling (`finqa_v10`, Diagnostic Only)

`v4` was measured at n=15 with `first_n` sampling. `finqa_v10` re-ran finqa for the first time with
three levers proven or adopted on the TatQA track since: `generation_temperature` pinned to `0.0`,
`random` sampling (seed 42) at n=40, and the judge prompt as it stood after TatQA's `tatv7`-`tatv9`
fixes — four changes from `v4` bundled at once (flagged as a confound, not hidden; see
Observations and Inference.md for the full isolating-test discipline this project follows).

| metric | `v4` (n=15, adopted) | `finqa_v10` (n=39) | Δ |
|---|---|---|---|
| relevance_rmse | 0.122 | 0.162 | +33% worse |
| utilization_rmse | 0.112 | 0.162 | +45% worse |
| completeness_rmse | 0.186 | 0.309 | +66% worse |
| adherence_aucroc | 0.786 | **0.375** | −52%, now below random guessing |

**Key finding:** all 3 true hallucinations in the n=39 sample were missed (0% recall), but row-level
reading of the judge's own reasoning shows this wasn't one failure — it was three distinct causes:
one row used a judge-verified-real number as the wrong quantity for the question (a premise-selection
error the fact-verification rule can't catch, since it only checks "is this number in the source,"
not "is this the number the question asked for"); one row was an outright refusal where the retrieved
context genuinely didn't contain the answer — the judge correctly confirmed the response was truthful
*given what retrieval handed it*, but the real problem was upstream, a retrieval-recall miss the judge
has no way to see; and one row was a judge-verified-correct calculation scored against a
`gt_adherence=0` label, matching the same "RAGBench's own reference response is wrong" pattern already
confirmed on TatQA.

**Decision:** not adopted — the four-variable confound makes `finqa_v10` unusable as a direct
replacement for `v4`, which remains the presented FinQA configuration. But two things from this run
are real and load-bearing regardless of the confound: (1) relevance/utilization/completeness RMSE
degrading substantially at larger, randomly-sampled `n` closely mirrors what TatQA's second wave
already showed, reinforcing that both "final" scoreboards below reflect a small, favorable slice
rather than settled corpus-wide performance; and (2) the retrieval-miss-as-invisible-hallucination
failure mode is newly identified and structural — no judge-prompt fix can address it, since a
context-grounded judge can only ever evaluate consistency with what retrieval provided.

---

## TatQA Track

### Scoreboard

| Run | top_k retrieve/final | Sampling | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc |
|---|---|---|---|---|---|---|
| `tatv1` (baseline) | 25 / 8 | `first_n` | 0.171 | 0.174 | 0.202 | NaN* |
| `tatv2` | 20 / 5 | `first_n` | 0.135 | 0.163 | 0.280 | NaN* |
| **`tatv3` (final)** | 25 / 8 | `random`, seed 42 | 0.157 | 0.160 | 0.397** | **0.558** |

All three runs share: `openai/gpt-oss-120b` judge, `bge-large-en-v1.5` embeddings, hybrid+rerank+MMR
retrieval, whole-document units (no chunking), `llama-3.3-70b-versatile` generator, `long_cot`
prompt with full reasoning shown to judge, n=15 — the proven finqa `v4` levers carried over as
TatQA's starting point.

\* `tatv1`/`tatv2` used deterministic `first_n` sampling, which happened to draw 15 rows that were
all `gt_adherence=1.0` — no hallucination case existed in either sample to measure detection
against, making AUCROC undefined regardless of config.

\** `tatv3` evaluates a different 15-question sample than `tatv1`/`tatv2` (random vs. `first_n`), so
completeness_rmse isn't directly comparable to the other two rows — see below.

### `tatv1` — Baseline (Adapted from FinQA `v4`)

**What was tested:** FinQA `v4`'s proven levers (independent judge, `bge-large` embeddings,
hybrid+rerank+MMR) applied to a new domain, TatQA, with one deliberate adaptation: `top_k_retrieve`
raised 20→25 and `top_k_final` raised 5→8, because TatQA's documents are much shorter than FinQA's
(offline-verified: avg 388 chars / median 319 vs. FinQA's avg 1,342 chars) — at the inherited
`top_k_final=5`, the generator would see under a third of the context volume FinQA's proven config
effectively provided.

**Results:** relevance_rmse=0.171, utilization_rmse=0.174, completeness_rmse=0.202,
adherence_aucroc=NaN (sampling artifact, not a config issue).

**Observations:**
- Relevance/utilization RMSE were somewhat worse than FinQA `v4`'s, traced to retrieved-sentence-count
  dilution: FinQA's own reference candidate document sets are much smaller per-question than what
  this pipeline's domain-wide corpus retrieval pulls in.
- The same judge scaffolding-sentence leniency pattern from FinQA `v3`/`v4` recurred here — first
  cross-domain confirmation that it's a genuine model behavior, not FinQA-specific.

### `tatv2` — Reduced top_k (Isolation Test)

**What changed:** `top_k_retrieve`/`top_k_final` reverted to FinQA's original proven values (20/5).
Everything else identical to `tatv1`, same 15 questions (same `first_n` sample).

**Results:** relevance_rmse=0.135 (better), utilization_rmse=0.163 (better), completeness_rmse=0.280
(38% worse), adherence mismatches 8/15 (worse than `tatv1`'s 6/15).

**Observations:** A clean, direct A/B test on the identical question set. Smaller top_k reduced
denominator dilution (helping relevance/utilization) but missed needed evidence often enough to hurt
completeness and adherence accuracy by a larger margin. Row 12 illustrates this concretely:
`pred_relevance` over-marked relative to ground truth, but `pred_completeness=0.000` — the narrower
candidate pool didn't contain what was actually needed.

**Inference:** Falsifies the simple "smaller top_k helps" hypothesis. `tatv1`'s wider retrieval is
the stronger overall configuration — the original reasoning for raising top_k (short documents need
more retrieved units) held up under direct test.

### `tatv3` — Restored top_k + Random Sampling (Final)

**What changed:** `top_k` restored to `tatv1`'s proven values (25/8) — not re-tested, already
settled by the `tatv1`/`tatv2` comparison. The one new, isolated variable: `sample_strategy`
switched from `first_n` to `random` (seed 42), added specifically because `first_n` was structurally
incapable of producing a computable `adherence_aucroc` for TatQA.

**Results:** relevance_rmse=0.157, utilization_rmse=0.160, completeness_rmse=0.397,
adherence_aucroc=0.558 (finally computable — modest, at the low end of RAGBench's reported range).

**Observations:**
- The new random sample included 2 real hallucination examples for the first time. The judge caught
  1 and missed 1 (50% recall), while also producing 5 false "unsupported" flags on genuinely correct
  responses — the scaffolding-sentence leniency pattern, now confirmed a third time across two
  domains and two different samples.
- The completeness_rmse increase vs. `tatv1` traces largely to a single large-magnitude outlier row
  (a refusal on an "average restricted stock" question, `pred_completeness=0.000` vs `gt=1.000`) —
  at n=15, RMSE is sensitive to individual outliers; this reads as sample-composition variance, not
  a broad-based regression, since retrieval settings are unchanged from `tatv1`.

**Inference / final decision:** `tatv3` was adopted as the final TatQA configuration, applying the
same standard used to conclude the FinQA track. The blocking measurement issue is resolved; the
dominant remaining gap (judge scaffolding-sentence leniency) is already investigated and understood
from the FinQA track, not a new open question; the one genuinely untested axis for TatQA (whether
hybrid+rerank+MMR retrieval helps this domain — FinQA found no effect) carries a reasonable prior
toward low expected payoff for the cost of another real run.

### Second Wave — Temperature Pinning + Judge-Prompt Fix (`tatv5_best` → `tatv9`, Diagnostic Only)

Five further runs tested two items `tatv3` carried into "suggested future work": pinning
`generation_temperature` to `0.0`, and judge-prompt fixes targeting the scaffolding-sentence
leniency pattern. **None replaced `tatv3` as the adopted config**, but they closed out both
questions and surfaced a deeper, dataset-level finding.

| Run | n (completed) | relevance_rmse | utilization_rmse | completeness_rmse | adherence_aucroc | What changed |
|---|---|---|---|---|---|---|
| `tatv5_best` | 30/30 | 0.225 | 0.241 | 0.386 | NaN (0 hallucinated rows drawn) | temp=0.0 + widened judge-prompt rule + n 15→30 (3 confounded changes at once) |
| `tatv6` | 56/80 | — | — | — | — | n raised to 80; run incomplete (30% API dropout under retry budget) |
| `tatv7` | 77/80 | 0.268 | 0.295 | 0.388 | 0.440 (below random) | retry budget fixed (dropout 30%→4%); exposed a judge-prompt regression |
| `tatv8` | 77/80 | 0.283 | 0.267 | 0.354 | 0.427 | narrowed the forced-`fully_supported` carve-out (no help — root cause mis-diagnosed) |
| `tatv9` | 78/80 | 0.264 | 0.269 | 0.405 | 0.441 | added a literal-value fact-verification rule to the judge prompt |

**Key finding:** from `tatv7` onward, every run on this fixed `random_seed=42` sample drew the exact
same 2 `gt_adherence=0` (hallucinated) rows — one about a Supplier A inventory-purchase percentage
change, one about operating-income growth. Reading RAGBench's own reference response for both
(zero extra API cost) showed **both references contain real arithmetic/data errors** — one uses the
wrong year's percentage, the other repeats a year-over-year figure as if it were sequential. `tatv9`'s
judge-prompt fix worked exactly as designed (confirmed from its own stated reasoning against the
source document) and correctly judged the Supplier A response as supported — which scored as "wrong"
only because it was measured against a `gt_adherence=0` label built on a broken reference. No prompt
engineering could have flipped this correctly, since doing so would require the judge to call a
verified-correct, source-checked response "unsupported."

**Decision:** stop iterating the judge prompt for this specific pair of rows — confirmed dataset
noise, not a fixable judge weakness. **`tatv3` (n=15) remains the adopted, presented TatQA
configuration.** The temperature-pinning and prompt-fix work is retained as diagnostic history: it
established that `adherence_aucroc` on this exact 80-row sample can never validate judge quality
(both available positive examples rest on broken ground truth), that RAGBench's own reference
responses can contain real errors and not just validly differ from a correct one, and that
relevance/utilization RMSE naturally run higher (~0.26–0.28 vs. `tatv3`'s 0.157/0.160) at n≈80 than
at n=15 due to sample composition, not regression.

---

## Cross-Domain Key Findings

1. **Judge and embedding quality mattered more than retrieval-ranking sophistication.** Hybrid
   search, RRF fusion, MMR diversity, and cross-encoder reranking — all enabled simultaneously in
   the baseline — produced no measurable improvement over plain dense-only retrieval. Every real
   gain in this project came from changing *who judges* the output or *what embeds* the documents.
2. **Self-evaluation bias is real and large.** Using the same model as generator and judge suppressed
   FinQA's hallucination detection to below-random (0.429 AUCROC). Swapping to an independent judge
   alone nearly doubled it to 0.714, with no other change.
3. **Embedding quality trades off across metrics, not uniformly.** A larger embedding model improved
   completeness and adherence further but slightly worsened relevance/utilization — a genuine cost,
   not a free lunch.
4. **The judge's scaffolding-sentence leniency inconsistency generalizes across domains.** Confirmed
   independently in FinQA (`v3`, `v4`) and TatQA (`tatv1`, `tatv3` — two different question samples)
   — strong evidence this is a real, domain-independent `openai/gpt-oss-120b` behavior. A rule-based
   automated fix was attempted and explicitly rejected after it degraded overall accuracy.
5. **Retrieval-scale tuning (top_k) is domain-specific and needs direct testing, not assumption.**
   TatQA's much shorter documents motivated raising top_k beyond FinQA's proven value, and a direct
   A/B test (`tatv1` vs `tatv2`) confirmed this was the right call — reverting regressed completeness
   and adherence accuracy by a larger margin than it improved relevance/utilization.
6. **Some residual error is structural, not fixable by configuration.** Ground-truth scores are
   computed against RAGBench's own reference response for each question, not this pipeline's
   generated response — occasional generator arithmetic mistakes get correctly flagged by the judge
   but scored against a different, correct reference answer, inflating apparent error without
   reflecting a retrieval or judge defect. Confirmed more severely than originally framed: RAGBench's
   own reference responses can themselves contain real arithmetic/data errors (verified directly
   against primary source documents for TatQA `tatv9`, and suspected on the same signature for
   `finqa_v10`) — a low raw `adherence_aucroc` at small `n` should never be read as proof of judge or
   pipeline failure without checking the underlying reference first.
7. **Both tracks' "final" scoreboards are measured on small, favorable samples, not corpus-wide
   performance.** Re-testing both domains at larger, randomly-sampled `n` (TatQA `tatv9`, n=78; FinQA
   `finqa_v10`, n=39) showed relevance/utilization/completeness RMSE substantially worse than the
   adopted `v4`/`tatv3` numbers in both cases — the same direction and rough magnitude in both
   domains, which argues this is a real small-sample-optimism effect rather than noise or a
   domain-specific issue.
8. **A context-grounded judge cannot detect hallucinations caused by retrieval failure.** Newly
   identified on FinQA `finqa_v10`: when the retriever fails to surface the needed document and the
   generator correctly refuses to answer, the judge verifies the refusal is truthful *relative to what
   it was given* and marks it fully supported — even though RAGBench's ground truth expects a derived
   answer. This is structurally distinct from the scaffolding-sentence leniency issue and cannot be
   fixed by judge-prompt engineering, since the judge never sees what the retriever missed.

## Final Configurations

| Parameter | FinQA (`v4`) | TatQA (`tatv3`) |
|---|---|---|
| Retrieval | Hybrid dense+BM25, RRF fusion, MMR, cross-encoder rerank | Same (inherited, not independently re-verified for TatQA) |
| Chunking | None (whole-document units) | None (documents already shorter than chunk-size threshold) |
| `top_k_retrieve` / `top_k_final` | 20 / 5 | 25 / 8 (directly tested and proven for this domain) |
| Embedding model | `BAAI/bge-large-en-v1.5` | Same (inherited) |
| Generator model | `llama-3.3-70b-versatile` | Same |
| Judge model | `openai/gpt-oss-120b` (independent) | Same |
| Generation prompt | `long_cot`, judge sees full reasoning | Same |
| Sample size | 15 | 15 |
| Sampling strategy | `first_n` | `random`, seed 42 (needed to get a computable AUCROC) |
| **relevance_rmse** | **0.122** | **0.157** |
| **utilization_rmse** | **0.112** | **0.160** |
| **completeness_rmse** | **0.186** | **0.397** |
| **adherence_aucroc** | **0.786** | **0.558** |

## Known Limitations

- The judge's sentence-keying uses a naive regex splitter, not ground-truth-aligned sentence
  boundaries — a fix that was validated offline in an earlier version of this pipeline but never
  carried into the final notebook. Applies to both domains.
- `adherence_aucroc` is statistically fragile at n=15 with only a handful of true-hallucination
  examples (1 for FinQA, 2 for TatQA) — a small number of row-level judge disagreements can swing it
  substantially.
- Ground-truth scores are computed against RAGBench's own reference response, not this pipeline's
  generated response — some residual RMSE reflects two different responses being compared. **This is
  more severe than "differ validly":** for TatQA's `random`-seed-42 draw, direct primary-source
  verification (`tatv7`-`tatv9`) confirmed RAGBench's own reference responses for both available
  `gt_adherence=0` rows contain real arithmetic/data errors. A low raw `adherence_aucroc` at small
  `n` should never be read as proof of judge or pipeline failure without checking the underlying
  reference response first.
- The judge's scaffolding-sentence leniency inconsistency is a real, unresolved, cross-domain
  limitation, not fixable by the configuration changes tested in this project (including a dedicated
  fact-verification judge-prompt rule tested in TatQA `tatv9`, which improved judge behavior in
  general but had no effect on the two rows it was aimed at, since their gap was ground-truth-side).
- For TatQA specifically: hybrid/rerank/MMR retrieval and the embedding model were inherited from
  FinQA's proven config, not independently re-verified on this domain — only top_k and sampling
  strategy were directly tested here.
- TatQA's `tatv9` checkpoint has plateaued at 78/80 rows across three separate completions (77, 77,
  78) despite `tatv7`'s retry-budget fix — a small (~2-3 row) residual hard-failure rate distinct
  from the transient rate-limiting that fix resolved, not yet root-caused.
- FinQA `v4`'s headline numbers (n=15, `first_n`) and TatQA `tatv3`'s (n=15, `random`) are both
  measured on small samples; larger-`n` re-tests on both tracks (TatQA `tatv9`, FinQA `finqa_v10`)
  show meaningfully worse RMSE, so the adopted configs' reported numbers should be read as
  best-case/favorable-sample results, not corpus-wide guarantees.
- The judge cannot detect hallucinations caused by retrieval recall failures (a correct-given-its-
  inputs refusal gets marked fully supported even when the true answer existed in the corpus but
  wasn't retrieved) — confirmed once on FinQA `finqa_v10`, not yet checked for prevalence.

## Suggested Future Work

- Port the ground-truth-aligned sentence-keying fix into the production notebook (both domains).
- ~~Judge-prompt engineering targeted at the scaffolding-sentence leniency inconsistency~~ — **done,
  inconclusive on TatQA's current sample:** three consecutive judge-prompt iterations (`tatv7`-`tatv9`)
  were tested; the final one demonstrably improved judge fact-checking behavior but couldn't move
  `adherence_aucroc` because both hallucination examples in the tested sample rest on broken RAGBench
  reference responses, not judge error. Re-testing needs a different sample, not another prompt tweak.
- For TatQA: verify whether hybrid+rerank+MMR retrieval actually helps this domain, the one
  configuration axis never independently tested here.
- Chunking strategy (`sentence_window`) as an untested retrieval-side lever for FinQA.
- Scale TatQA sample size well beyond n=80, **with a fresh random seed**, specifically to get
  hallucination examples that aren't the same two broken-reference rows every seed-42 run keeps
  drawing — the only remaining way to get a clean read on judge quality for adherence specifically.
- Root-cause TatQA `tatv9`'s persistent 2/80 hard-failure rows (stable across 3 completions) — inspect
  their raw API responses directly rather than relying on more retries, which haven't closed the gap.
- Isolate `finqa_v10`'s four bundled changes: re-run FinQA at `v4`'s original temperature and judge
  prompt with only `sample_strategy=random`/`n=40` changed, to confirm whether the RMSE/AUCROC drop is
  sampling-driven (current leading hypothesis, consistent with TatQA's own pattern) rather than a
  judge-prompt regression.
- Check how common retrieval-miss-driven refusals are on FinQA at scale (the `finqa_v10` row 34
  pattern) — if it recurs, it's a case for either raising FinQA's `top_k_final` (never independently
  tested, unlike TatQA) or explicitly separating refusal responses from the adherence metric, since
  the judge structurally cannot resolve them either way.
