"""Main evaluation orchestrator."""
import asyncio
from datetime import datetime
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from src.api.rag_client import RAGClient
from src.api.schemas import RAGResponse
from src.config.settings import Settings
from src.evaluation.deepeval_evaluator import DeepEvalEvaluator
from src.evaluation.models import (
    BatchEvaluationOutput,
    EvaluationInput,
    EvaluationOutput,
    FrameworkResult,
)
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationOrchestrator:
    """
    Central coordinator for RAG evaluation pipeline.

    Workflow:
    1. Load questions (CSV or console)
    2. Query RAG API for each question
    3. Run RAGAS evaluation
    4. Run DeepEval evaluation
    5. Compare results
    6. Generate reports
    """

    def __init__(self, settings: Settings):
        """
        Initialize orchestrator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.rag_client = RAGClient(settings)

        # Initialize evaluators based on config
        self.ragas = None
        self.deepeval = None

        if settings.evaluation.frameworks.ragas.enabled:
            self.ragas = RAGASEvaluator(settings)

        if settings.evaluation.frameworks.deepeval.enabled:
            self.deepeval = DeepEvalEvaluator(settings)

    async def evaluate_single(
        self,
        question: str,
        collection: Optional[str] = None,
        skip_rag_query: bool = False,
        rag_response: Optional[RAGResponse] = None
    ) -> EvaluationOutput:
        """
        Evaluate single question end-to-end.

        Args:
            question: User question
            collection: RAG collection name
            skip_rag_query: Skip RAG query (use provided response)
            rag_response: Pre-fetched RAG response

        Returns:
            EvaluationOutput with all results
        """
        # Get RAG response
        if skip_rag_query and rag_response:
            response = rag_response
        else:
            logger.info("Querying RAG API", question=question[:50])
            response = await self.rag_client.query(question, collection)

        # Create evaluation input
        eval_input = EvaluationInput(
            question=question,
            answer=response.answer,
            contexts=response.contexts,
        )

        # Run evaluations
        ragas_result = None
        deepeval_result = None

        if self.ragas:
            logger.info("Running RAGAS evaluation")
            try:
                ragas_result = await self.ragas.evaluate(
                    question, response.answer, response.contexts
                )
            except Exception as e:
                logger.error("RAGAS evaluation failed", error=str(e))
                ragas_result = FrameworkResult(
                    framework="ragas",
                    faithfulness=MetricScore(score=0, explanation=f"Error: {str(e)}"),
                    answer_relevancy=MetricScore(score=0, explanation=f"Error: {str(e)}"),
                    context_relevancy=MetricScore(score=0, explanation=f"Error: {str(e)}"),
                    error=str(e),
                )

        if self.deepeval:
            logger.info("Running DeepEval evaluation")
            try:
                deepeval_result = await self.deepeval.evaluate(
                    question, response.answer, response.contexts
                )
            except Exception as e:
                logger.error("DeepEval evaluation failed", error=str(e))
                deepeval_result = FrameworkResult(
                    framework="deepeval",
                    faithfulness=MetricScore(score=0, explanation=f"Error: {str(e)}"),
                    answer_relevancy=MetricScore(score=0, explanation=f"Error: {str(e)}"),
                    context_relevancy=MetricScore(score=0, explanation=f"Error: {str(e)}"),
                    error=str(e),
                )

        # Compare frameworks
        comparison = None
        if ragas_result and deepeval_result:
            comparison = self._compare_frameworks(ragas_result, deepeval_result)

        return EvaluationOutput(
            input=eval_input,
            ragas_result=ragas_result,
            deepeval_result=deepeval_result,
            comparison=comparison,
        )

    async def evaluate_batch(
        self,
        questions: List[str],
        collection: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: bool = True
    ) -> BatchEvaluationOutput:
        """
        Evaluate batch of questions with progress tracking.

        Args:
            questions: List of questions
            collection: RAG collection name
            progress_callback: Optional callback(current, total)
            show_progress: Show progress bar

        Returns:
            BatchEvaluationOutput with all results
        """
        batch_output = BatchEvaluationOutput(
            total_evaluations=len(questions),
        )

        # Use tqdm for progress
        iterator = tqdm(questions, desc="Evaluating") if show_progress else questions

        for i, question in enumerate(iterator):
            try:
                result = await self.evaluate_single(question, collection)
                batch_output.results.append(result)
                batch_output.successful_evaluations += 1
            except Exception as e:
                logger.error("Evaluation failed", question=question[:50], error=str(e))
                batch_output.errors.append({
                    "question": question,
                    "error": str(e),
                })
                batch_output.failed_evaluations += 1

            if progress_callback:
                progress_callback(i + 1, len(questions))

        return batch_output

    async def evaluate_from_responses(
        self,
        responses: List[Dict],
        show_progress: bool = True
    ) -> BatchEvaluationOutput:
        """
        Evaluate from pre-fetched RAG responses.

        Args:
            responses: List of dicts with question, response, context
            show_progress: Show progress bar

        Returns:
            BatchEvaluationOutput
        """
        batch_output = BatchEvaluationOutput(
            total_evaluations=len(responses),
        )

        iterator = tqdm(responses, desc="Evaluating") if show_progress else responses

        for item in iterator:
            question = item.get("question", "")
            answer = item.get("response", "")
            context = item.get("context", "")

            # Parse context (might be string or list)
            contexts = [context] if isinstance(context, str) else context

            try:
                # Create RAG response object
                rag_response = RAGResponse(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                )

                result = await self.evaluate_single(
                    question,
                    skip_rag_query=True,
                    rag_response=rag_response,
                )
                batch_output.results.append(result)
                batch_output.successful_evaluations += 1
            except Exception as e:
                logger.error("Evaluation failed", question=question[:50], error=str(e))
                batch_output.errors.append({
                    "question": question,
                    "error": str(e),
                })
                batch_output.failed_evaluations += 1

        return batch_output

    def _compare_frameworks(
        self,
        ragas_result: FrameworkResult,
        deepeval_result: FrameworkResult
    ) -> Dict[str, float]:
        """Compare scores between frameworks."""
        metrics = ["faithfulness", "answer_relevancy", "context_relevancy"]
        comparison = {}

        threshold = self.settings.evaluation.comparison.discrepancy_threshold

        for metric in metrics:
            ragas_score = getattr(ragas_result, metric).score
            deepeval_score = getattr(deepeval_result, metric).score
            diff = abs(ragas_score - deepeval_score)

            comparison[f"{metric}_diff"] = diff
            comparison[f"{metric}_ragas"] = ragas_score
            comparison[f"{metric}_deepeval"] = deepeval_score

            if diff > threshold:
                comparison[f"{metric}_discrepancy"] = True

        return comparison

    def evaluate_single_sync(
        self,
        question: str,
        collection: Optional[str] = None
    ) -> EvaluationOutput:
        """Synchronous wrapper for evaluate_single."""
        return asyncio.run(self.evaluate_single(question, collection))

    def evaluate_batch_sync(
        self,
        questions: List[str],
        collection: Optional[str] = None
    ) -> BatchEvaluationOutput:
        """Synchronous wrapper for evaluate_batch."""
        return asyncio.run(self.evaluate_batch(questions, collection))


# Import MetricScore for error handling
from src.evaluation.models import MetricScore
