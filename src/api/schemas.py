"""Pydantic models for API requests and responses."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Document Upload Schemas
# ============================================================================

class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    filename: str
    content_type: str
    metadata: Optional[Dict[str, Any]] = None
    collection_name: Optional[str] = "default"
    chunking_strategy: Optional[str] = "auto"
    chunk_size: Optional[int] = 512
    chunk_overlap: Optional[int] = 50


class UploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    status: str  # "success", "processing", "failed"
    chunks_created: Optional[int] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchUploadResponse(BaseModel):
    """Response model for batch document upload."""
    total: int
    successful: int
    failed: int
    results: List[UploadResponse]
    errors: List[Dict[str, str]] = Field(default_factory=list)


# ============================================================================
# RAG Query Schemas
# ============================================================================

class SourceInfo(BaseModel):
    """Source document information."""
    document_id: str
    document_name: str
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    relevance_score: Optional[float] = None


class RAGQueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str
    collection: Optional[str] = "default"
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None


class RAGResponse(BaseModel):
    """Response model for RAG query."""
    question: str
    answer: str
    contexts: List[str] = Field(default_factory=list)
    sources: Optional[List[SourceInfo]] = None
    latency_ms: Optional[int] = None


# ============================================================================
# Evaluation Schemas
# ============================================================================

class MetricResult(BaseModel):
    """Single metric evaluation result."""
    score: float
    explanation: Optional[str] = None


class RAGASResult(BaseModel):
    """RAGAS evaluation results."""
    faithfulness: MetricResult
    answer_relevancy: MetricResult
    context_relevancy: MetricResult
    evaluation_time_ms: Optional[int] = None


class DeepEvalResult(BaseModel):
    """DeepEval evaluation results."""
    faithfulness: MetricResult
    answer_relevancy: MetricResult
    context_relevancy: MetricResult
    evaluation_time_ms: Optional[int] = None


class ComparisonResult(BaseModel):
    """Framework comparison result."""
    faithfulness_diff: float
    answer_relevancy_diff: float
    context_relevancy_diff: float
    has_significant_discrepancy: bool = False


class SingleEvaluationResult(BaseModel):
    """Complete evaluation result for single question."""
    evaluation_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    question: str
    rag_response: RAGResponse
    ragas_evaluation: Optional[RAGASResult] = None
    deepeval_evaluation: Optional[DeepEvalResult] = None
    comparison: Optional[ComparisonResult] = None


# ============================================================================
# Report Schemas
# ============================================================================

class MetricStatistics(BaseModel):
    """Statistics for a single metric."""
    average: float
    min: float
    max: float
    std: float


class ReportStatistics(BaseModel):
    """Aggregate statistics for report."""
    total_evaluations: int
    ragas_averages: Dict[str, float] = Field(default_factory=dict)
    ragas_min: Dict[str, float] = Field(default_factory=dict)
    ragas_max: Dict[str, float] = Field(default_factory=dict)
    ragas_std: Dict[str, float] = Field(default_factory=dict)
    deepeval_averages: Dict[str, float] = Field(default_factory=dict)
    deepeval_min: Dict[str, float] = Field(default_factory=dict)
    deepeval_max: Dict[str, float] = Field(default_factory=dict)
    deepeval_std: Dict[str, float] = Field(default_factory=dict)
    pass_rates: Dict[str, float] = Field(default_factory=dict)


class DiscrepancyRecord(BaseModel):
    """Record of significant discrepancy between frameworks."""
    evaluation_id: str
    question: str
    metric: str
    ragas_score: float
    deepeval_score: float
    difference: float


class ComparisonSummary(BaseModel):
    """Summary of framework comparison."""
    avg_score_difference: Dict[str, float] = Field(default_factory=dict)
    correlation: Dict[str, float] = Field(default_factory=dict)
    significant_discrepancies: List[DiscrepancyRecord] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Complete evaluation report."""
    report_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    framework_versions: Dict[str, str] = Field(default_factory=dict)
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    evaluations: List[SingleEvaluationResult] = Field(default_factory=list)
    statistics: Optional[ReportStatistics] = None
    comparison_summary: Optional[ComparisonSummary] = None
