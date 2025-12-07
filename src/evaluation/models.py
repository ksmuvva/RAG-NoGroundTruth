"""Evaluation data models."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EvaluationInput(BaseModel):
    """Input for evaluation."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class MetricScore(BaseModel):
    """Score for a single metric."""
    score: float
    explanation: Optional[str] = None
    raw_output: Optional[Dict[str, Any]] = None


class FrameworkResult(BaseModel):
    """Results from a single evaluation framework."""
    framework: str
    faithfulness: MetricScore
    answer_relevancy: MetricScore
    context_relevancy: MetricScore
    evaluation_time_ms: int = 0
    error: Optional[str] = None


class EvaluationOutput(BaseModel):
    """Complete output for a single evaluation."""
    evaluation_id: str = Field(default_factory=lambda: f"eval_{uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.now)
    input: EvaluationInput
    ragas_result: Optional[FrameworkResult] = None
    deepeval_result: Optional[FrameworkResult] = None
    comparison: Optional[Dict[str, float]] = None

    def get_average_scores(self) -> Dict[str, float]:
        """Get average scores across frameworks."""
        metrics = ["faithfulness", "answer_relevancy", "context_relevancy"]
        averages = {}

        for metric in metrics:
            scores = []
            if self.ragas_result:
                scores.append(getattr(self.ragas_result, metric).score)
            if self.deepeval_result:
                scores.append(getattr(self.deepeval_result, metric).score)

            if scores:
                averages[metric] = sum(scores) / len(scores)
            else:
                averages[metric] = 0.0

        return averages


class BatchEvaluationOutput(BaseModel):
    """Output for batch evaluation."""
    batch_id: str = Field(default_factory=lambda: f"batch_{uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=datetime.now)
    total_evaluations: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    results: List[EvaluationOutput] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.results:
            return {}

        metrics = ["faithfulness", "answer_relevancy", "context_relevancy"]
        stats = {
            "ragas": {},
            "deepeval": {},
        }

        for framework in ["ragas", "deepeval"]:
            for metric in metrics:
                scores = []
                for result in self.results:
                    fw_result = getattr(result, f"{framework}_result")
                    if fw_result:
                        metric_score = getattr(fw_result, metric)
                        scores.append(metric_score.score)

                if scores:
                    import statistics
                    stats[framework][metric] = {
                        "avg": statistics.mean(scores),
                        "min": min(scores),
                        "max": max(scores),
                        "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                    }

        return stats
