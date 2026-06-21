"""
Test-time evaluation: runs AdaptiveCP on the test split, computes metrics,
and saves per-question records + summary.
"""

import os
import json
import time

from core import (
    SamplerConfig, adaptive_sample, build_prediction_set, check_coverage,
    filter_split, clear_embed_cache, run_calibration, compute_qhat,
)


DEFAULT_CONFIG = SamplerConfig()


# ── Core evaluation loop ──────────────────────────────────────────────────────

def sample_test_set(
    model, tokenizer, DATASETS, dataset_name: str,
    config:      SamplerConfig = DEFAULT_CONFIG,
    max_samples: int           = 500,
) -> list:

    test_samples = filter_split(DATASETS[dataset_name], "test")
    if len(test_samples) > max_samples:
        import random
        test_samples = random.sample(test_samples, max_samples)

    print(f"\nSampling test set: {dataset_name} | n={len(test_samples)} "
          f"(sampled once, reused across alphas)")

    results = []
    for i, sample in enumerate(test_samples):
        clear_embed_cache()

        result = adaptive_sample(model, tokenizer, sample.question, config=config)
        result["question"]     = sample.question
        result["gold_answers"] = sample.gold_answers
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:4d}/{len(test_samples)}] sampled")

    return results


def score_results(test_results: list, dataset_name: str,
                  alpha: float, q_hat: float) -> dict:
    """
    Metrics:
      ECR  — Empirical Coverage Rate (%): fraction covered, abstentions count
             as NOT covered. Target: >= (1 - alpha) * 100.
      SSC  — Selective Set Coverage: coverage among non-abstained questions.
      APSS — Average Prediction Set Size over ALL questions (0 for abstentions).
      api_calls — mean model calls per question (AdaptiveCP's efficiency claim).
    """
    n_covered = n_ssc_covered = n_ssc_total = 0
    total_set_size = total_api_calls = 0
    results = []

    for result in test_results:
        pred_set  = build_prediction_set(result, q_hat)
        covered   = check_coverage(pred_set, result["gold_answers"])
        abstained = len(pred_set) == 0   # empty set = structural OR score-based abstention

        if covered:
            n_covered += 1
        if not abstained:
            n_ssc_total += 1
            if covered:
                n_ssc_covered += 1

        total_set_size  += len(pred_set)
        total_api_calls += result["n_samples"]

        # shallow copy so per-alpha pred_set/covered don't clobber the cached result
        r = dict(result)
        r["pred_set"]  = pred_set
        r["covered"]   = covered
        r["abstained"] = abstained
        results.append(r)

    n = len(test_results)
    return {
        "dataset":      dataset_name,
        "alpha":        alpha,
        "q_hat":        q_hat,
        "n":            n,
        "ECR":          round(n_covered / n * 100, 1),
        "SSC":          round(n_ssc_covered / n_ssc_total * 100, 1) if n_ssc_total else 0.0,
        "APSS":         round(total_set_size / n, 2),
        "api_calls":    round(total_api_calls / n, 1),
        "abstain_rate": round(sum(1 for r in results if r["abstained"]) / n * 100, 1),
        "results":      results,
    }


def evaluate(
    model, tokenizer, DATASETS, dataset_name: str,
    alpha:       float,
    q_hat:       float,
    config:      SamplerConfig = DEFAULT_CONFIG,
    max_samples: int           = 500,
) -> dict:
    """Backward-compatible single-alpha evaluation: sample then score."""
    test_results = sample_test_set(
        model, tokenizer, DATASETS, dataset_name,
        config=config, max_samples=max_samples,
    )
    print(f"\nEvaluating {dataset_name} | alpha={alpha} | q_hat={q_hat:.4f} "
          f"| n={len(test_results)}")
    return score_results(test_results, dataset_name, alpha, q_hat)


# ── Saving helpers ────────────────────────────────────────────────────────────

def _save_detailed(results: list, dataset: str, alpha: float, q_hat: float,
                   save_dir: str, model_name: str) -> str:
    """Per-question JSONL — streamable, greppable, audit-friendly."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_{dataset}_alpha{alpha:.2f}_detailed.jsonl")

    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "question":        r["question"],
                "gold_answers":    r["gold_answers"],
                "pred_set":        r.get("pred_set", []),
                "covered":         r["covered"],
                "abstained":       r["abstained"],
                "reason":          r.get("reason"),
                "k":               r.get("k"),
                "spectral_gap":    r.get("spectral_gap"),
                "n_batches":       r.get("n_batches"),
                "n_samples":       r.get("n_samples"),
                "gap_history":     r.get("gap_history", []),
                "representatives": r.get("representatives", []),
                "clusters":        r.get("clusters", []),
                "cluster_info":    [
                    {"representative": c["representative"],
                     "frequency":     round(c["frequency"], 4),
                     "score":         c["score"]}
                    for c in r.get("cluster_info", [])
                ],
                "alpha":           alpha,
                "q_hat":           q_hat,
                "dataset":         dataset,
            }) + "\n")
    return path


def _save_summary(all_results: list, dataset: str, save_dir: str,
                  model_name: str, lofree_ref: dict) -> str:
    """Per-alpha summary JSON with AdaptiveCP + LofreeCP side-by-side."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_{dataset}_summary.json")

    rows = []
    for r in all_results:
        alpha = r["alpha"]
        ref   = lofree_ref.get(alpha, {})
        rows.append({
            "dataset":      dataset,
            "model":        model_name,
            "alpha":        alpha,
            "target_ECR":   round((1 - alpha) * 100, 1),
            "ECR":          r["ECR"],
            "SSC":          r["SSC"],
            "APSS":         r["APSS"],
            "api_calls":    r["api_calls"],
            "abstain_rate": r["abstain_rate"],
            "utility":      round(r["SSC"] * (1 - r["abstain_rate"] / 100), 1),
            "q_hat":        r["q_hat"],
            "n":            r["n"],
            "lofree_ECR":   ref.get("ECR"),
            "lofree_SSC":   ref.get("SSC"),
            "lofree_APSS":  ref.get("APSS"),
        })

    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return path


# ── Main entry point ──────────────────────────────────────────────────────────

def run_full_evaluation(
    model, tokenizer, DATASETS,
    dataset_name: str,
    alphas:       list,
    lofree_ref:   dict,
    cal_samples:  int           = 500,
    test_samples: int           = 500,
    config:       SamplerConfig = DEFAULT_CONFIG,
    save_dir:     str           = None,
    model_name:   str           = "model",
) -> list:
    """
    For each alpha: calibrate → evaluate → save → print comparison.
    Returns a list of result dicts (one per alpha).
    """
    all_results = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"FULL EVALUATION: {dataset_name.upper()}  |  model={model_name}")
    print(f"{'='*70}")

    # 1) Calibrate ONCE — n_scores are alpha-independent; only the quantile changes.
    cal = run_calibration(
        model, tokenizer, DATASETS, dataset_name,
        config=config, max_samples=cal_samples,
        save_path=(os.path.join(
            save_dir, f"{model_name}_{dataset_name}_calibration.json",
        ) if save_dir else None),
    )
    n_scores = cal["n_scores"]

    # 2) Sample the TEST set ONCE — adaptive_sample is alpha-independent.
    test_results = sample_test_set(
        model, tokenizer, DATASETS, dataset_name,
        config=config, max_samples=test_samples,
    )

    # 3) Per alpha: pick the quantile + threshold + score (all cheap, no model calls).
    for alpha in sorted(alphas):
        print(f"\n--- alpha={alpha} (target ECR >= {(1-alpha)*100:.0f}%) ---")

        q_hat = compute_qhat(n_scores, alpha)
        print(f"  q_hat = {q_hat:.4f}")

        result = score_results(test_results, dataset_name, alpha, q_hat)
        result["cal_coverage"] = cal["coverage"]
        all_results.append(result)

        if save_dir:
            path = _save_detailed(
                result["results"], dataset_name, alpha, q_hat, save_dir, model_name,
            )
            print(f"  Saved detailed → {path}")

        _print_alpha_comparison(result, lofree_ref.get(alpha, {}))

    if save_dir:
        path = _save_summary(all_results, dataset_name, save_dir, model_name, lofree_ref)
        print(f"\n  Saved summary → {path}")

    return all_results


def _print_alpha_comparison(r: dict, ref: dict) -> None:
    target  = (1 - r["alpha"]) * 100
    ecr_ok  = "✓" if r["ECR"] >= target else "✗"
    print(f"\n  Results vs LofreeCP (alpha={r['alpha']}):")
    print(f"  {'Metric':<12} {'AdaptiveCP':>12} {'LofreeCP':>12} {'Delta':>10}")
    print(f"  {'-'*48}")
    for k, label, fmt in [("ECR", f"ECR {ecr_ok}", ".1f"),
                          ("SSC", "SSC ↑", ".1f"),
                          ("APSS", "APSS ↓", ".2f")]:
        ref_val = ref.get(k, "-")
        delta   = (r[k] - ref.get(k, r[k])) if ref else 0
        ref_str = f"{ref_val:>11}" if ref_val == "-" else f"{ref_val:>11.1f}"
        print(f"  {label:<12} {r[k]:>11{fmt}} {ref_str}  {delta:>+9{fmt}}")
    print(f"  {'API calls':<12} {r['api_calls']:>12.1f} {'~20-30':>11}  {'(adaptive)':>10}")
    print(f"  {'Abstain%':<12} {r['abstain_rate']:>11.1f}%")


def print_comparison_table(all_results: list, lofree_ref: dict,
                           dataset_name: str) -> None:
    """Clean comparison table across all alphas (no LaTeX block)."""
    alphas = sorted(set(r["alpha"] for r in all_results))

    print(f"\n{'='*75}")
    print(f"COMPARISON TABLE: {dataset_name.upper()}")
    print(f"{'='*75}")

    header = f"\n{'Method':<22}" + "".join(f"  α={a}" for a in alphas)
    print(header)

    for metric, arrow in [("ECR", ""), ("SSC", "↑"), ("APSS", "↓")]:
        own  = f"  {metric} {arrow}".ljust(22)
        ref  = f"  LofreeCP {metric}".ljust(22)
        for a in alphas:
            r       = next(x for x in all_results if x["alpha"] == a)
            ref_val = lofree_ref.get(a, {}).get(metric, "-")
            own_str = f"  {r[metric]:.2f}" if metric == "APSS" else f"  {r[metric]:.1f}"
            ref_str = f"  {ref_val}" if ref_val == "-" else (
                f"  {ref_val:.2f}" if metric == "APSS" else f"  {ref_val:.1f}"
            )
            own += own_str
            ref += ref_str
        print(own)
        print(ref)
        print()

    api = "  API calls (new)".ljust(22)
    for a in alphas:
        r = next(x for x in all_results if x["alpha"] == a)
        api += f"  {r['api_calls']:.1f}"
    print(api)
    print(f"  LofreeCP API calls    {'~25 (fixed)':>}")

    util = "  Utility (new)".ljust(22)
    for a in alphas:
        r = next(x for x in all_results if x["alpha"] == a)
        util += f"  {r['SSC'] * (1 - r['abstain_rate'] / 100):.1f}"
    print(util)
