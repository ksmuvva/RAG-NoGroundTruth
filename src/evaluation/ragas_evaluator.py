"""RAGAS framework wrapper for RAG evaluation."""
import os
import time
from typing import List, Optional

from src.config.constants import ErrorCode
from src.config.settings import RAGEvalError, Settings, get_azure_openai_config
from src.evaluation.models import EvaluationInput, FrameworkResult, MetricScore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """
    RAGAS framework wrapper for RAG evaluation.

    Metrics:
    - faithfulness: Factual consistency with context
    - answer_relevancy: Response relevance to question
    - context_relevancy: Retrieved context relevance
    """

    def __init__(self, settings: Settings):
        """
        Initialize RAGAS evaluator.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._initialized = False
        self._llm = None
        self._embeddings = None

    def _initialize(self):
        """Initialize RAGAS with Azure OpenAI."""
        if self._initialized:
            return

        try:
            azure_config = get_azure_openai_config()

            # Set environment variables for RAGAS
            os.environ["AZURE_OPENAI_API_KEY"] = azure_config["api_key"]
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_config["endpoint"]

            from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

            # Initialize LLM
            self._llm = AzureChatOpenAI(
                azure_deployment=azure_config["deployment_name"],
                api_version=azure_config["api_version"],
                temperature=0,
            )

            # Initialize embeddings
            self._embeddings = AzureOpenAIEmbeddings(
                azure_deployment=azure_config.get("embedding_deployment", "text-embedding-ada-002"),
                api_version=azure_config["api_version"],
            )

            self._initialized = True
            logger.info("RAGAS evaluator initialized with Azure OpenAI")

        except Exception as e:
            raise RAGEvalError(
                ErrorCode.E300_RAGAS_INIT_FAILED,
                f"Failed to initialize RAGAS: {str(e)}"
            )

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> FrameworkResult:
        """
        Run RAGAS evaluation on single Q&A pair.

        Args:
            question: User question
            answer: RAG answer
            contexts: Retrieved context chunks

        Returns:
            FrameworkResult with scores and explanations
        """
        self._initialize()

        start_time = time.time()

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )

            # Build dataset
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            dataset = Dataset.from_dict(data)

            # Define metrics
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,  # Using context_precision as context_relevancy
            ]

            # Run evaluation
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=self._llm,
                embeddings=self._embeddings,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Extract scores
            faithfulness_score = float(result.get("faithfulness", 0))
            answer_rel_score = float(result.get("answer_relevancy", 0))
            context_rel_score = float(result.get("context_precision", 0))

            return FrameworkResult(
                framework="ragas",
                faithfulness=MetricScore(
                    score=faithfulness_score,
                    explanation=self._generate_explanation("faithfulness", faithfulness_score),
                ),
                answer_relevancy=MetricScore(
                    score=answer_rel_score,
                    explanation=self._generate_explanation("answer_relevancy", answer_rel_score),
                ),
                context_relevancy=MetricScore(
                    score=context_rel_score,
                    explanation=self._generate_explanation("context_relevancy", context_rel_score),
                ),
                evaluation_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("RAGAS evaluation failed", error=str(e))
            raise RAGEvalError(
                ErrorCode.E301_RAGAS_EVAL_FAILED,
                f"RAGAS evaluation failed: {str(e)}"
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

    def _generate_explanation(self, metric: str, score: float) -> str:
        """Generate explanation based on metric and score."""
        explanations = {
            "faithfulness": {
                "high": "The answer is well-grounded in the provided context with strong factual support.",
                "medium": "The answer is partially supported by context, some claims may need verification.",
                "low": "The answer contains information not well-supported by the provided context.",
            },
            "answer_relevancy": {
                "high": "The answer directly and comprehensively addresses the question.",
                "medium": "The answer partially addresses the question with some relevant information.",
                "low": "The answer does not adequately address the question.",
            },
            "context_relevancy": {
                "high": "The retrieved context is highly relevant to the question.",
                "medium": "The retrieved context is partially relevant with some tangential information.",
                "low": "The retrieved context has limited relevance to the question.",
            },
        }

        level = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
        return explanations.get(metric, {}).get(level, f"Score: {score:.2f}")
