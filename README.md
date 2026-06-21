# AdaptiveCP

**Conformal prediction sets for open-ended LLM question answering, with an adaptive sampling budget and built-in abstention.**

AdaptiveCP wraps a frozen, black-box LLM with a calibrated procedure that does three things at once:

1. **Samples adaptively** — it keeps drawing answers from the model only until the *structure* of the answers stabilizes, instead of using a fixed (and usually wasteful) number of calls.
2. **Builds a conformal prediction set** — using split conformal calibration, it returns a small set of answers that contains a correct answer with a user-chosen probability `1 − α`.
3. **Knows when to abstain** — when the model is internally incoherent, expresses ignorance, or fragments into too many distinct answers, AdaptiveCP returns an empty set (abstains) rather than guessing.

The method needs **no access to logits, no fine-tuning, and no white-box internals** — only the ability to sample text from the model and to embed strings with a sentence encoder.

---

## The core idea

For a given question we sample several short answers at temperature, then ask: *do these answers agree?*

We answer that question with **spectral graph analysis** of the responses:

1. **Embed & connect.** Clean each response (strip explanations, lowercase, truncate) and embed it with MiniLM. Build a graph where responses are nodes and an edge connects two responses whose cosine similarity exceeds `sim_threshold`.
2. **Find the structure.** Compute the normalized graph Laplacian `L = I − D^{-1/2} W D^{-1/2}` and look at its eigenvalues. The **eigengap heuristic** gives `k*`, the number of well-separated answer clusters, and the **spectral gap** measures how confident that split is.
3. **Stop adaptively.** After each batch of samples we recompute the structure. As soon as `k*` is stable *and* the spectral gap stops moving (`structure_converged`), we stop sampling. Easy questions converge in 2 batches; hard ones spend the full budget.
4. **Score by frequency.** Each cluster gets a nonconformity score `s = 1 − (cluster size / total samples)`. A cluster the model overwhelmingly favors has a low score; a rare cluster has a high one.

### Calibration and prediction sets

This is standard **split conformal prediction** on top of the cluster scores:

- On the **calibration split**, for each question we record the score of the cluster that contains the *gold* answer (`true_cluster_score`). If the model never surfaced the truth, the score is `1.0` (the worst possible) — so the sampler's own failures are baked into the threshold.
- The calibration scores are **α-independent**: we sample the model **once**, then for any target `α` we read off the quantile `q̂ = sorted_scores[⌈(n+1)(1−α)⌉ − 1]`. Changing `α` costs nothing.
- At **test time**, the prediction set is every cluster representative whose score `≤ q̂`. If no cluster qualifies, the set is empty — a **score-based abstention**.

### When does it abstain?

A question yields an empty prediction set (abstention) for any of these reasons:

| Reason | Meaning |
|---|---|
| `budget_exhausted` | Answer structure never stabilized within the sampling budget. |
| `uncertainty_responses` | The model converged, but mostly emitted "unknown / unclear / cannot determine"-style answers. |
| `too_many_clusters` | `k*` exceeds `max_clusters` — the model fragmented into too many distinct answers. |
| empty set | Converged with clear structure, but no cluster was frequent enough to pass `q̂`. |

---

## What the experiments show

The headline finding (visualized in `plots/`) is that AdaptiveCP's empirical coverage **tracks an "answerability ceiling"** — the fraction of questions the model can actually answer — rather than the requested target `1 − α`. Pushing `α` lower does not buy more coverage; it only buys more abstention. Abstention, in turn, is **selective**: the questions AdaptiveCP drops are disproportionately the ones it would have gotten wrong.

### Metrics

- **ECR** — Empirical Coverage Rate over *all* questions (abstentions count as misses). Target is `(1 − α)·100`.
- **SSC** — Selective Set Coverage: coverage among the questions the system *did* answer.
- **APSS** — Average Prediction Set Size over all questions (0 for abstentions). Lower is more efficient/decisive.
- **api_calls** — mean model calls per question; this is AdaptiveCP's efficiency claim against fixed-budget baselines like LofreeCP (which use ~20–30 calls regardless of difficulty).

### The plots

| File | What it shows |
|---|---|
| `plots/ceiling_bars.png` | Best achieved ECR sits right at the operating ceiling (calibration coverage), and the ceiling barely moves from Mistral-7B → Qwen-7B → Qwen-14B. **Scale does not lift the ceiling.** |
| `plots/coverage_validity.png` | ECR vs. target coverage. The curves flatten at each model's ceiling (dotted lines) instead of following the `ECR = target` diagonal — beyond the ceiling, lowering `α` no longer raises coverage. |
| `plots/risk_coverage.png` | Selective risk–coverage. Moving up-and-right (higher SSC as abstain rate rises) means abstention is **selective** — the dropped questions are the harder, more error-prone ones. |
| `plots/qhat_staircase.png` | `q̂` vs. `α`. Long flat steps show that many `α` values map to the same threshold — a direct consequence of the score distribution below. |
| `plots/score_distributions.png` | Calibration nonconformity scores pile up at 0 (model nailed it) and 1 (model never surfaced the truth), with little mass in between — so `q̂` snaps between a few discrete thresholds. |
| `plots/efficiency.png` | Prediction-set size (APSS) and abstain rate as functions of `α`, per model. |

---

## Repository layout

```
AdaptiveCP/
├── adaptive_cp/
│   ├── core.py       # Method: scoring, spectral analysis, adaptive sampler, calibration
│   ├── model.py      # 4-bit quantized causal LM loading + short-answer generation; MiniLM encoder
│   ├── data.py       # Dataset loaders: TriviaQA, WebQuestions, MMLU
│   ├── evaluate.py   # Test-time eval loop, metrics (ECR/SSC/APSS), saving, comparison tables
│   └── main.ipynb    # End-to-end driver: load data + model → calibrate → evaluate → compare
└── plots/            # Result figures (see table above)
```

### `core.py` — the method

The heart of the project. Key pieces:

- `SamplerConfig` — all knobs: batch size, max/min batches, similarity & eigengap thresholds, `max_clusters`, convergence tolerance.
- `clean_response`, `is_uncertainty_response`, `responses_are_substantive` — normalize raw generations and detect "I don't know"-style answers.
- `analyze_graph_structure` — the spectral pipeline: similarity graph → normalized Laplacian → eigengap → `k*`, spectral gap, clusters, and per-cluster frequency scores.
- `structure_converged` — the adaptive stopping rule.
- `adaptive_sample` — the sampling loop that ties it together and decides abstention.
- `build_prediction_set`, `check_coverage`, `true_cluster_score` — prediction-set construction and (semantic + token-F1) coverage matching.
- `compute_qhat`, `run_calibration` — split-conformal calibration; scores are sampled once and reused across all `α`.

### `model.py` — generation backend

Loads a chat LLM in 4-bit (NF4) via `bitsandbytes` and prompts it for short factual answers using the tokenizer's chat template (so it adapts to any instruct model). Swap `MODEL_NAME` to change models — the paper compares Mistral-7B, Qwen2.5-14B, and Llama-2-13B. Also loads the `all-MiniLM-L6-v2` sentence encoder used for clustering and coverage.

### `data.py` — datasets

Loaders for **TriviaQA** (`rc.nocontext`), **WebQuestions**, and **MMLU** (multiple-choice reformatted as open-ended). Each is split deterministically by index into 50% calibration / 25% validation / 25% test.

### `evaluate.py` — evaluation

`run_full_evaluation` calibrates once, samples the test set once, then scores every `α` cheaply. Produces per-question JSONL records, a per-`α` summary JSON with LofreeCP numbers side-by-side, and printable comparison tables.

---

## Quickstart

Requires a CUDA GPU (4-bit quantized inference) and gated-model access on Hugging Face for some models.

```bash
pip install -r requirements.txt
```

Then run `adaptive_cp/main.ipynb` top to bottom, or use the API directly:

```python
from core import SamplerConfig
from model import load_model
from data import load_triviaqa, load_webquestions, load_mmlu
from evaluate import run_full_evaluation

model, tokenizer = load_model()
DATASETS = {
    "triviaqa": load_triviaqa(seed=42),
    "webq":     load_webquestions(seed=42),
}

LOFREE_TRIVIAQA = {  # baseline reference numbers, per alpha
    0.20: {"ECR": 80.1, "SSC": 79.0, "APSS": 2.19},
    0.30: {"ECR": 70.3, "SSC": 76.7, "APSS": 1.08},
    0.40: {"ECR": 60.0, "SSC": 81.0, "APSS": 0.75},
}

results = run_full_evaluation(
    model, tokenizer, DATASETS,
    dataset_name="triviaqa",
    alphas=[0.20, 0.30, 0.40],
    lofree_ref=LOFREE_TRIVIAQA,
    cal_samples=500, test_samples=500,
    save_dir="results/run1", model_name="mistral7b",
)
```

### Tuning behavior

All behavior lives in `SamplerConfig`:

```python
config = SamplerConfig(
    batch_size=3, max_batches=6,   # sampling budget = batch_size * max_batches
    temperature=0.9,
    sim_threshold=0.60,            # graph edge threshold
    min_gap=0.30,                  # min eigengap to trust the cluster structure
    max_clusters=3,                # abstain above this many answer clusters
)
```

---

## Method at a glance

```
question
   │
   ▼
sample a batch of short answers  ◄────────────┐
   │                                          │ not converged & budget left
   ▼                                          │
embed → similarity graph → Laplacian → eigengap
   │  (k* clusters, spectral gap)             │
   ▼                                          │
structure stable? ────────────────────────────┘
   │ yes
   ▼
substantive? & k* ≤ max_clusters? ── no ──► ABSTAIN (empty set)
   │ yes
   ▼
score clusters (1 − frequency); keep those ≤ q̂
   │
   ▼
prediction set  (empty ⇒ score-based abstention)
```

Calibration sets `q̂` from the gold-cluster scores on held-out data, once, for every `α`.
