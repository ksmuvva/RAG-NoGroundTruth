"""Statistics calculator for evaluation reports."""
import statistics
from typing import Any, Dict, List, Optional

from src.api.schemas import (
    ComparisonSummary,
    DiscrepancyRecord,
    ReportStatistics,
)
from src.evaluation.models import BatchEvaluationOutput, EvaluationOutput


def calculate_statistics(
    results: List[EvaluationOutput],
    thresholds: Optional[Dict[str, float]] = None
) -> ReportStatistics:
    """
    Calculate aggregate statistics from evaluation results.

    Args:
        results: List of evaluation outputs
        thresholds: Pass/fail thresholds per metric

    Returns:
        ReportStatistics with aggregated metrics
    """
    if not results:
        return ReportStatistics(total_evaluations=0)

    thresholds = thresholds or {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_relevancy": 0.7,
    }

    metrics = ["faithfulness", "answer_relevancy", "context_relevancy"]

    # Collect scores per framework and metric
    ragas_scores = {m: [] for m in metrics}
    deepeval_scores = {m: [] for m in metrics}

    for result in results:
        if result.ragas_result:
            for metric in metrics:
                score = getattr(result.ragas_result, metric).score
                ragas_scores[metric].append(score)

        if result.deepeval_result:
            for metric in metrics:
                score = getattr(result.deepeval_result, metric).score
                deepeval_scores[metric].append(score)

    # Calculate statistics
    def calc_stats(scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {"avg": 0, "min": 0, "max": 0, "std": 0}
        return {
            "avg": statistics.mean(scores),
            "min": min(scores),
            "max": max(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0,
        }

    ragas_stats = {m: calc_stats(ragas_scores[m]) for m in metrics}
    deepeval_stats = {m: calc_stats(deepeval_scores[m]) for m in metrics}

    # Calculate pass rates
    pass_rates = {}
    for metric in metrics:
        threshold = thresholds.get(metric, 0.7)
        all_scores = ragas_scores[metric] + deepeval_scores[metric]
        if all_scores:
            passing = sum(1 for s in all_scores if s >= threshold)
            pass_rates[metric] = passing / len(all_scores)
        else:
            pass_rates[metric] = 0

    return ReportStatistics(
        total_evaluations=len(results),
        ragas_averages={m: ragas_stats[m]["avg"] for m in metrics},
        ragas_min={m: ragas_stats[m]["min"] for m in metrics},
        ragas_max={m: ragas_stats[m]["max"] for m in metrics},
        ragas_std={m: ragas_stats[m]["std"] for m in metrics},
        deepeval_averages={m: deepeval_stats[m]["avg"] for m in metrics},
        deepeval_min={m: deepeval_stats[m]["min"] for m in metrics},
        deepeval_max={m: deepeval_stats[m]["max"] for m in metrics},
        deepeval_std={m: deepeval_stats[m]["std"] for m in metrics},
        pass_rates=pass_rates,
    )


def calculate_comparison_summary(
    results: List[EvaluationOutput],
    discrepancy_threshold: float = 0.2
) -> ComparisonSummary:
    """
    Calculate comparison summary between frameworks.

    Args:
        results: List of evaluation outputs
        discrepancy_threshold: Threshold for flagging discrepancies

    Returns:
        ComparisonSummary with differences and correlations
    """
    metrics = ["faithfulness", "answer_relevancy", "context_relevancy"]

    # Collect paired scores
    paired_scores = {m: {"ragas": [], "deepeval": []} for m in metrics}

    for result in results:
        if result.ragas_result and result.deepeval_result:
            for metric in metrics:
                ragas_score = getattr(result.ragas_result, metric).score
                deepeval_score = getattr(result.deepeval_result, metric).score
                paired_scores[metric]["ragas"].append(ragas_score)
                paired_scores[metric]["deepeval"].append(deepeval_score)

    # Calculate average differences
    avg_diffs = {}
    for metric in metrics:
        if paired_scores[metric]["ragas"] and paired_scores[metric]["deepeval"]:
            diffs = [
                abs(r - d) for r, d in zip(
                    paired_scores[metric]["ragas"],
                    paired_scores[metric]["deepeval"]
                )
            ]
            avg_diffs[metric] = statistics.mean(diffs)
        else:
            avg_diffs[metric] = 0

    # Calculate correlations
    correlations = {}
    for metric in metrics:
        ragas = paired_scores[metric]["ragas"]
        deepeval = paired_scores[metric]["deepeval"]
        if len(ragas) > 1 and len(deepeval) > 1:
            correlations[metric] = _calculate_correlation(ragas, deepeval)
        else:
            correlations[metric] = 0

    # Find significant discrepancies
    discrepancies = []
    for result in results:
        if result.ragas_result and result.deepeval_result:
            for metric in metrics:
                ragas_score = getattr(result.ragas_result, metric).score
                deepeval_score = getattr(result.deepeval_result, metric).score
                diff = abs(ragas_score - deepeval_score)

                if diff > discrepancy_threshold:
                    discrepancies.append(DiscrepancyRecord(
                        evaluation_id=result.evaluation_id,
                        question=result.input.question[:100],
                        metric=metric,
                        ragas_score=ragas_score,
                        deepeval_score=deepeval_score,
                        difference=diff,
                    ))

    return ComparisonSummary(
        avg_score_difference=avg_diffs,
        correlation=correlations,
        significant_discrepancies=discrepancies,
    )


def _calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denominator_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        return 0

    return numerator / (denominator_x * denominator_y)
