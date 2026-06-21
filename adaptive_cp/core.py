"""
AdaptiveCP core: config, response cleaning, spectral scoring, adaptive sampling,
calibration, and prediction-set construction.
"""

import math
import time
import json
import numpy as np
from collections import Counter
from dataclasses import dataclass
from sklearn.cluster import KMeans

from model import generate_batch, load_sentence_encoder


# ── Config 

@dataclass
class QASample:
    question_id:  str
    question:     str
    gold_answers: list   # list[str]
    source:       str    # "triviaqa" | "webq" | "mmlu"
    split:        str    # "calibration" | "validation" | "test"


@dataclass
class SamplerConfig:
    batch_size:    int   = 3     # responses per batch
    max_batches:   int   = 6     # hard budget = batch_size * max_batches
    temperature:   float = 0.9
    min_batches:   int   = 2     # never stop before this many batches
    sim_threshold: float = 0.60  # min cosine similarity to draw a graph edge
    min_gap:       float = 0.30  # min eigengap to consider structure real
    max_clusters:  int   = 3     # abstain if k* exceeds this
    gap_tolerance: float = 0.10  # spectral gap change tolerance for convergence


UNCERTAINTY_TOKENS = {
    "unknown", "uncertain", "unclear", "unspecified", "unavailable",
    "undetermined", "unverified", "undefined", "unanswerable",
    "insufficient", "incomplete", "inconclusive",
    "no data", "no record", "no records", "no reliable", "no precise",
    "no accurate", "no specific", "no consensus", "no information",
    "not known", "not available", "not recorded", "not specified",
    "not enough", "not clear", "not certain", "not determined",
    "varies", "vary", "variable", "disputed", "debated", "contested",
    "estimates", "estimate", "approximat", "roughly", "around",
    "cannot", "can't", "could not", "impossible to",
    "sources differ", "sources vary", "sources disagree",
    "limited records", "limited data", "limited historical",
    "lack of", "lacking", "absence of",
    "historians disagree", "scholars disagree",
}

COVERAGE_SIM_THRESHOLD = 0.65

MMLU_SUBJECTS = [
    "high_school_mathematics",
    "college_medicine",
    "high_school_world_history",
    "professional_law",
    "abstract_algebra",
]


# ── Sentence encoder (lazy + cache) ───────────────────────────────────────────

_ENCODER = None
_EMBED_CACHE: dict = {}


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = load_sentence_encoder()
    return _ENCODER


def embed_cached(text: str) -> np.ndarray:
    """Embed text using MiniLM with caching. Returns L2-normalized vector."""
    key = text.strip().lower()
    if key not in _EMBED_CACHE:
        _EMBED_CACHE[key] = get_encoder().encode(
            key, normalize_embeddings=True, show_progress_bar=False,
        )
    return _EMBED_CACHE[key]


def clear_embed_cache():
    """Call between questions to prevent unbounded memory growth."""
    _EMBED_CACHE.clear()


# ── Split helpers ─────────────────────────────────────────────────────────────

def assign_split(i: int, n_total: int) -> str:
    """50% calibration / 25% validation / 25% test (deterministic by index)."""
    frac = i / n_total
    if frac < 0.50:
        return "calibration"
    if frac < 0.75:
        return "validation"
    return "test"


def print_split_counts(samples: list) -> None:
    counts = Counter(s.split for s in samples)
    for split in ["calibration", "validation", "test"]:
        print(f"  {split}: {counts.get(split, 0)}")


def filter_split(samples: list, split: str) -> list:
    return [s for s in samples if s.split == split]


# Response cleaning / uncertainty detection 

def clean_response(response: str, max_tokens: int = 8) -> str:
    response = response.strip()

    for char in [".", "\n", ";", "[", "]"]:
        if char in response:
            response = response[:response.index(char)].strip()

    for word in [" because", " since", " which", " who", " as ", " but "]:
        if word in response.lower():
            idx = response.lower().index(word)
            response = response[:idx].strip()

    tokens = response.split()
    if len(tokens) > max_tokens:
        response = " ".join(tokens[:max_tokens])

    return response.lower().strip()


def is_uncertainty_response(response: str) -> bool:
    r = response.lower().strip()
    return any(token in r for token in UNCERTAINTY_TOKENS)


def responses_are_substantive(
    cleaned: list,
    uncertainty_threshold: float = 0.6,
    min_length: int = 2,
) -> bool:
    """
    True if the majority of responses are substantive answers
    (not uncertainty tokens and not trivially short).
    """
    if not cleaned:
        return False
    n_bad = sum(
        1 for r in cleaned
        if is_uncertainty_response(r) or len(r.strip()) < min_length
    )
    return (n_bad / len(cleaned)) < uncertainty_threshold


def token_f1(a: str, b: str) -> float:
    """Token-level F1 with multiset overlap. Used in coverage matching."""
    a_tokens = Counter(a.lower().split())
    b_tokens = Counter(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    overlap   = sum((a_tokens & b_tokens).values())
    precision = overlap / sum(a_tokens.values())
    recall    = overlap / sum(b_tokens.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Spectral graph analysis ───────────────────────────────────────────────────

def _make_cluster_info(clusters: list, representatives: list, n_total: int) -> list:
    """
    Pair each cluster with its frequency-based conformity score.

      frequency = (responses in the cluster) / (total responses sampled)
      score     = 1 - frequency      # low score = model strongly favored it

    Deduplicates by representative, summing the frequencies of any merged
    clusters (edge case: two clusters elect the same representative string).
    Returns a list of dicts: {representative, members, frequency, score}.
    """
    merged, order = {}, []
    for members, rep in zip(clusters, representatives):
        if not rep:
            continue
        if rep not in merged:
            merged[rep] = {"representative": rep, "members": list(members),
                           "frequency": 0.0}
            order.append(rep)
        else:
            merged[rep]["members"].extend(members)
        merged[rep]["frequency"] += (len(members) / n_total) if n_total else 0.0

    info = []
    for rep in order:
        c = merged[rep]
        c["score"] = round(max(0.0, 1.0 - c["frequency"]), 6)
        info.append(c)
    return info


def analyze_graph_structure(
    responses: list,
    sim_threshold: float = 0.60,
    min_gap:       float = 0.30,
) -> dict:
    """
    Analyze cluster structure of LLM responses via spectral graph analysis.

    Builds a thresholded cosine-similarity graph, computes the normalized
    Laplacian, and uses the eigengap heuristic to find k* (number of clusters)
    and the spectral gap. The gap drives adaptive stopping only. The conformal
    nonconformity score is per-cluster and frequency-based (score = 1 - freq) —
    see cluster_info / _make_cluster_info.

    Returns dict with: k, spectral_gap, clusters, representatives,
    cluster_info (list of {representative, members, frequency, score}).
    """
    cleaned = [clean_response(r) for r in responses]
    n = len(cleaned)

    if n == 1:
        info = [{"representative": cleaned[0], "members": cleaned,
                 "frequency": 1.0, "score": 0.0}]
        return {
            "k": 1, "spectral_gap": 2.0,
            "clusters": [cleaned], "representatives": [cleaned[0]],
            "cluster_info": info,
        }

    # Embed unique texts in one forward pass
    encoder = get_encoder()
    unique_texts = list(dict.fromkeys(cleaned))
    emb_matrix = encoder.encode(
        unique_texts, normalize_embeddings=True, show_progress_bar=False,
    )
    embed_map = {t: emb_matrix[i] for i, t in enumerate(unique_texts)}
    embeddings = np.array([embed_map[c] for c in cleaned])

    # Thresholded similarity matrix
    W = np.clip(embeddings @ embeddings.T, 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    W[W < sim_threshold] = 0.0

    # Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    degree      = W.sum(axis=1)
    safe_degree = np.where(degree > 0, degree, 1.0)
    d_inv_sqrt  = np.where(degree > 0, 1.0 / np.sqrt(safe_degree), 0.0)
    L = np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])

    eigenvalues = np.clip(np.linalg.eigvalsh(L), 0.0, None)

    # Eigengap heuristic
    max_k = min(n - 1, 6)
    gaps  = np.diff(eigenvalues[:max_k + 1])
    k_star       = int(np.argmax(gaps)) + 1
    spectral_gap = float(gaps[k_star - 1])

    # Diffuse: gap too small to trust structure
    if spectral_gap < min_gap:
        return {
            "k": n, "spectral_gap": spectral_gap,
            "clusters": [[c] for c in cleaned], "representatives": [],
            "cluster_info": [],
        }

    # Single cluster
    if k_star == 1:
        best = int(np.argmax(degree)) if degree.max() > 0 else 0
        rep  = cleaned[best]
        info = [{"representative": rep, "members": cleaned,
                 "frequency": 1.0, "score": 0.0}]
        return {
            "k": 1, "spectral_gap": spectral_gap,
            "clusters": [cleaned], "representatives": [rep],
            "cluster_info": info,
        }

    # Multi-cluster: spectral embedding → k-means
    _, eigenvectors = np.linalg.eigh(L)
    spectral_coords = eigenvectors[:, :k_star]
    row_norms       = np.linalg.norm(spectral_coords, axis=1, keepdims=True)
    spectral_coords = spectral_coords / np.where(row_norms > 0, row_norms, 1.0)

    labels = KMeans(n_clusters=k_star, n_init=10, random_state=42).fit_predict(
        spectral_coords,
    )

    clusters, representatives = [], []
    for cid in range(k_star):
        indices = np.where(labels == cid)[0]
        if len(indices) == 0:
            continue
        clusters.append([cleaned[i] for i in indices])
        best = indices[int(np.argmax(degree[indices]))]
        representatives.append(cleaned[best])

    # Pair each cluster with its frequency-based score (handles dedup).
    info = _make_cluster_info(clusters, representatives, n)

    return {
        "k": len(info), "spectral_gap": spectral_gap,
        "clusters":        [c["members"] for c in info],
        "representatives": [c["representative"] for c in info],
        "cluster_info":    info,
    }


def structure_converged(prev: dict, curr: dict, gap_tolerance: float = 0.10) -> bool:
    """
    Convergence between two consecutive batches requires:
      1. k* unchanged (categorical stability)
      2. |Δgap| < tolerance, OR gap already > 1.0 (complete-graph shortcut)
    """
    if prev is None:
        return False
    if prev["k"] != curr["k"]:
        return False
    gap_stable  = abs(curr["spectral_gap"] - prev["spectral_gap"]) < gap_tolerance
    gap_certain = curr["spectral_gap"] > 1.0
    return gap_stable or gap_certain


# ── Adaptive sampler ──────────────────────────────────────────────────────────

DEFAULT_CONFIG = SamplerConfig()


def adaptive_sample(model, tokenizer, question: str,
                    config: SamplerConfig = DEFAULT_CONFIG) -> dict:
    """
    Sample until the response graph structure stabilizes or budget is exhausted.

    Returns a result dict with keys:
      abstain, reason, k, spectral_gap, representatives,
      clusters, cluster_info, cleaned, gap_history, n_batches, n_samples.

    Abstention reasons:
      1. budget_exhausted      — structure never converged within max_batches
      2. uncertainty_responses — converged but responses express ignorance
      3. too_many_clusters     — k* exceeds max_clusters
      4. empty prediction set  — no cluster's score <= q_hat (downstream,
                                 in build_prediction_set)
    """
    all_responses  = []
    gap_history    = []
    prev_structure = None
    last_structure = None

    for batch_num in range(1, config.max_batches + 1):
        new = generate_batch(
            model, tokenizer, question,
            n=config.batch_size, temperature=config.temperature,
        )
        all_responses.extend(new)

        structure = analyze_graph_structure(
            all_responses,
            sim_threshold=config.sim_threshold,
            min_gap=config.min_gap,
        )
        gap_history.append(structure["spectral_gap"])
        last_structure = structure

        if batch_num >= config.min_batches and structure_converged(
            prev_structure, structure, gap_tolerance=config.gap_tolerance,
        ):
            cleaned = [clean_response(r) for r in all_responses]

            if not responses_are_substantive(cleaned):
                return _abstain_result(
                    "uncertainty_responses", all_responses, cleaned,
                    structure, gap_history, batch_num,
                )

            if structure["k"] > config.max_clusters:
                return _abstain_result(
                    "too_many_clusters", all_responses, cleaned,
                    structure, gap_history, batch_num,
                )

            return {
                "abstain":         False,
                "reason":          "converged",
                "responses":       all_responses,
                "cleaned":         cleaned,
                "k":               structure["k"],
                "spectral_gap":    structure["spectral_gap"],
                "representatives": structure["representatives"],
                "clusters":        structure["clusters"],
                "cluster_info":    structure["cluster_info"],
                "gap_history":     gap_history,
                "n_batches":       batch_num,
                "n_samples":       len(all_responses),
            }

        prev_structure = structure

    # Budget exhausted
    cleaned = [clean_response(r) for r in all_responses]
    return _abstain_result(
        "budget_exhausted", all_responses, cleaned,
        last_structure, gap_history, config.max_batches,
    )


def _abstain_result(reason, responses, cleaned, structure, gap_history, n_batches):
    return {
        "abstain":         True,
        "reason":          reason,
        "responses":       responses,
        "cleaned":         cleaned,
        "k":               structure["k"] if structure else None,
        "spectral_gap":    structure["spectral_gap"] if structure else 0.0,
        "representatives": [],
        "clusters":        structure["clusters"] if structure else [],
        "cluster_info":    structure["cluster_info"] if structure else [],
        "gap_history":     gap_history,
        "n_batches":       n_batches,
        "n_samples":       len(responses),
    }


# ── Prediction set + coverage 

def build_prediction_set(result: dict, q_hat: float = None) -> list:
    """
    Return prediction set for a test question, or [] for abstention.

    Includes the representative of EVERY cluster whose frequency-based score
    is <= q_hat (i.e. frequency >= 1 - q_hat). If no cluster qualifies the set
    is empty — a score-based abstention. Structural abstentions return []
    regardless. With q_hat=None, returns all representatives (diagnostic).
    """
    if result["abstain"]:
        return []

    info = result.get("cluster_info", [])
    if q_hat is None:
        return result["representatives"]

    preds = []
    for c in info:
        if c["score"] <= q_hat:
            preds.append(c["representative"])

    seen = []
    for p in preds:
        if p and p not in seen:
            seen.append(p)
    return seen


def check_coverage(prediction_set: list, gold_answers: list) -> bool:
    """
    True if any prediction is semantically similar to any gold answer.
    Primary: MiniLM cosine >= COVERAGE_SIM_THRESHOLD. Fallback: token F1 >= 0.5.
    Empty prediction set (abstention) returns False.
    """
    if not prediction_set:
        return False

    for pred in prediction_set:
        if not pred:
            continue
        for gold in gold_answers:
            gold_clean = clean_response(gold, max_tokens=12)
            if not gold_clean:
                continue
            try:
                sim = float(np.dot(embed_cached(pred), embed_cached(gold_clean)))
                if sim >= COVERAGE_SIM_THRESHOLD:
                    return True
            except Exception:
                pass
            if token_f1(pred, gold_clean) >= 0.5:
                return True
    return False


def true_cluster_score(result: dict, gold_answers: list) -> float:
    """
    Conformal calibration score for one question: the score (1 - frequency) of
    the cluster that CONTAINS the gold answer. Returns 1.0 if no cluster's
    representative matches the gold answer (the model never surfaced the truth).

    Matching reuses check_coverage, so the calibration score is consistent with
    how test-time coverage is judged: the true cluster lands in the prediction
    set  iff  its score <= q_hat  iff  this calibration score <= q_hat.
    """
    best = 1.0
    for c in result.get("cluster_info", []):
        if check_coverage([c["representative"]], gold_answers):
            best = min(best, c["score"])
    return best


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_qhat(n_scores: list, alpha: float) -> float:
    """
    Standard split-CP threshold:
      position = ceil((n + 1) * (1 - alpha))
      q_hat    = sorted_scores[position - 1]
    Returns inf if position > n (conservative fallback).
    """
    n = len(n_scores)
    if n == 0:
        print("  WARNING: no scored calibration examples. q_hat = inf.")
        return float("inf")

    position = math.ceil((n + 1) * (1 - alpha))
    if position > n:
        print(f"  NOTE: position={position} > n={n}. q_hat = inf (conservative).")
        return float("inf")

    return sorted(n_scores)[position - 1]


def run_calibration(
    model, tokenizer, DATASETS, dataset_name: str,
    alpha:       float         = None,
    config:      SamplerConfig = DEFAULT_CONFIG,
    max_samples: int           = 500,
    save_path:   str           = None,
) -> dict:
    """
    Run conformal calibration on the calibration split of a dataset.

    The calibration scores (n_scores) are ALPHA-INDEPENDENT — they depend only
    on (model, data, score function). Alpha selects a quantile from them, which
    is cheap. So calibrate ONCE and call compute_qhat(n_scores, alpha) per alpha
    instead of re-sampling the model for every alpha. If alpha is None, q_hat is
    left None and only the scores are returned.

    Structural abstentions contribute a calibration score of 1.0 (worst
    possible) so they push q_hat upward — the sampler's failure rate is baked
    into the threshold.
    """
    samples = filter_split(DATASETS[dataset_name], "calibration")
    if len(samples) > max_samples:
        import random
        samples = random.sample(samples, max_samples)

    print(f"\nCalibrating on {dataset_name} (n={len(samples)}) "
          f"— scores are alpha-independent, sampled once")
    print(f"Config: {config}")
    print("-" * 55)

    results, n_scores = [], []
    n_abstained = 0
    t_start = time.time()

    for i, sample in enumerate(samples):
        clear_embed_cache()

        result = adaptive_sample(model, tokenizer, sample.question, config=config)
        result["question"]     = sample.question
        result["gold_answers"] = sample.gold_answers
        result["source"]       = sample.source

        covered = check_coverage(
            result["representatives"] if not result["abstain"] else [],
            sample.gold_answers,
        )
        result["covered"] = covered

        if result["abstain"]:
            n_abstained += 1
            n_scores.append(1.0)
        else:
            n_scores.append(true_cluster_score(result, sample.gold_answers))

        results.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            elapsed  = time.time() - t_start
            per_q    = elapsed / (i + 1)
            eta      = per_q * (len(samples) - i - 1)
            coverage = sum(r["covered"] for r in results) / len(results)
            print(f"  [{i+1:4d}/{len(samples)}] "
                  f"abstain_rate={n_abstained/(i+1)*100:.1f}%  "
                  f"coverage={coverage*100:.1f}%  "
                  f"eta={eta/60:.1f}min")

    q_hat        = compute_qhat(n_scores, alpha) if alpha is not None else None
    abstain_rate = n_abstained / len(samples)
    coverage     = sum(r["covered"] for r in results) / len(results)

    cal_dict = {
        "dataset":      dataset_name,
        "alpha":        alpha,
        "n_total":      len(samples),
        "n_scored":     len(n_scores),
        "n_abstained":  n_abstained,
        "abstain_rate": abstain_rate,
        "q_hat":        q_hat,
        "n_scores":     n_scores,
        "coverage":     coverage,
        "config":       config.__dict__,
        "results":      results,
    }

    print(f"\nCalibration complete — {dataset_name}")
    print(f"  n_total      : {len(n_scores)}")
    print(f"  n_abstained  : {n_abstained} ({abstain_rate*100:.1f}%)")
    if q_hat is not None:
        print(f"  q_hat        : {q_hat:.4f}")
    print(f"  cal SSC      : {coverage*100:.1f}%  (coverage among non-abstained)")

    if save_path:
        save_dict = {k: v for k, v in cal_dict.items() if k != "results"}
        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=2)
        print(f"  Saved to {save_path}")

    return cal_dict
