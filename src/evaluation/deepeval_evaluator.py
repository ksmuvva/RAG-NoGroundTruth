"""DeepEval framework wrapper for RAG evaluation."""
import os
import time
from typing import List

from src.config.constants import ErrorCode
from src.config.settings import RAGEvalError, Settings, get_azure_openai_config
from src.evaluation.models import FrameworkResult, MetricScore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeepEvalEvaluator:
    """
    DeepEval framework wrapper for RAG evaluation.

    Metrics:
    - FaithfulnessMetric: Factual consistency
    - AnswerRelevancyMetric: Response relevance
    - ContextualRelevancyMetric: Context relevance
    """

    def __init__(self, settings: Settings):
        """
        Initialize DeepEval evaluator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._initialized = False
        self._model = None

    def _initialize(self):
        """Initialize DeepEval with Azure OpenAI."""
        if self._initialized:
            return

        try:
            azure_config = get_azure_openai_config()

            # Set environment variables for DeepEval
            os.environ["AZURE_OPENAI_API_KEY"] = azure_config["api_key"]
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_config["endpoint"]
            os.environ["OPENAI_API_VERSION"] = azure_config["api_version"]

            # Import and configure DeepEval model
            from deepeval.models import AzureOpenAI

            self._model = AzureOpenAI(
                model=azure_config["deployment_name"],
                azure_endpoint=azure_config["endpoint"],
                azure_api_key=azure_config["api_key"],
            )

            self._initialized = True
            logger.info("DeepEval evaluator initialized with Azure OpenAI")

        except Exception as e:
            raise RAGEvalError(
                ErrorCode.E302_DEEPEVAL_INIT_FAILED,
                f"Failed to initialize DeepEval: {str(e)}"
            )

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> FrameworkResult:
        """
        Run DeepEval evaluation on single Q&A pair.

        Args:
            question: User question
            answer: RAG answer
            contexts: Retrieved context chunks

        Returns:
            FrameworkResult with scores and reasons
        """
        self._initialize()

        start_time = time.time()

        try:
            from deepeval import evaluate
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualRelevancyMetric,
                FaithfulnessMetric,
            )
            from deepeval.test_case import LLMTestCase

            # Create test case
            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=contexts,
            )

            # Initialize metrics
            faithfulness_metric = FaithfulnessMetric(
                model=self._model,
                threshold=self.settings.evaluation.thresholds.faithfulness,
            )
            answer_rel_metric = AnswerRelevancyMetric(
                model=self._model,
                threshold=self.settings.evaluation.thresholds.answer_relevancy,
            )
            context_rel_metric = ContextualRelevancyMetric(
                model=self._model,
                threshold=self.settings.evaluation.thresholds.context_relevancy,
            )

            # Measure each metric
            faithfulness_metric.measure(test_case)
            answer_rel_metric.measure(test_case)
            context_rel_metric.measure(test_case)

            elapsed_ms = int((time.time() - start_time) * 1000)

            return FrameworkResult(
                framework="deepeval",
                faithfulness=MetricScore(
                    score=faithfulness_metric.score or 0.0,
                    explanation=faithfulness_metric.reason,
                ),
                answer_relevancy=MetricScore(
                    score=answer_rel_metric.score or 0.0,
                    explanation=answer_rel_metric.reason,
                ),
                context_relevancy=MetricScore(
                    score=context_rel_metric.score or 0.0,
                    explanation=context_rel_metric.reason,
                ),
                evaluation_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("DeepEval evaluation failed", error=str(e))
            raise RAGEvalError(
                ErrorCode.E303_DEEPEVAL_EVAL_FAILED,
                f"DeepEval evaluation failed: {str(e)}"
            )

    def evaluate_sync(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> FrameworkResult:
        """Synchronous evaluation wrapper."""
        import asyncio
        return asyncio.run(self.evaluate(question, answer, contexts))
