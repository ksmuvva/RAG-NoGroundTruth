"""RAG API client for querying."""
import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx

from src.api.schemas import RAGResponse, SourceInfo
from src.config.constants import ErrorCode
from src.config.settings import RAGEvalError, Settings, get_rag_api_key
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGClient:
    """Client for interacting with RAG API query endpoint."""

    def __init__(self, settings: Settings):
        """
        Initialize RAG client.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.base_url = settings.rag_api.base_url.rstrip('/')
        self.query_config = settings.rag_api.query
        self.auth_config = settings.rag_api.auth

        # Get API key
        self.api_key = get_rag_api_key()

        # Build headers
        self.headers = self._build_headers()

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {"Content-Type": "application/json"}

        if self.api_key and self.auth_config.type == "api_key":
            headers[self.auth_config.header_name] = self.api_key
        elif self.api_key and self.auth_config.type == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _build_query_request(
        self,
        question: str,
        collection: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build query request payload."""
        mapping = self.query_config.request_mapping

        payload = {
            mapping.get("question_field", "question"): question,
            mapping.get("top_k_field", "top_k"): top_k,
        }

        if collection:
            payload[mapping.get("collection_field", "collection")] = collection

        if filters:
            payload["filters"] = filters

        return payload

    def _parse_response(self, question: str, data: Dict[str, Any], latency_ms: int) -> RAGResponse:
        """Parse RAG API response into RAGResponse model."""
        mapping = self.query_config.response_mapping

        # Extract fields with mapping
        answer = data.get(mapping.get("answer_field", "answer"), "")
        contexts = data.get(mapping.get("context_field", "contexts"), [])
        sources_data = data.get(mapping.get("sources_field", "sources"), [])

        # Ensure contexts is a list
        if isinstance(contexts, str):
            contexts = [contexts]

        # Parse sources
        sources = None
        if sources_data:
            sources = []
            for src in sources_data:
                if isinstance(src, dict):
                    sources.append(SourceInfo(
                        document_id=src.get("document_id", ""),
                        document_name=src.get("document_name", ""),
                        chunk_id=src.get("chunk_id"),
                        page_number=src.get("page_number"),
                        relevance_score=src.get("relevance_score"),
                    ))

        return RAGResponse(
            question=question,
            answer=answer,
            contexts=contexts,
            sources=sources,
            latency_ms=latency_ms,
        )

    async def query(
        self,
        question: str,
        collection: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> RAGResponse:
        """
        Send question to RAG API and get response.

        Args:
            question: User question
            collection: Document collection to search
            top_k: Number of context chunks to retrieve
            filters: Optional metadata filters

        Returns:
            RAGResponse with answer, contexts, and sources

        Raises:
            RAGEvalError: If query fails
        """
        url = f"{self.base_url}{self.query_config.endpoint}"
        payload = self._build_query_request(question, collection, top_k, filters)

        logger.debug("Querying RAG API", url=url, question=question[:50])

        # Retry logic
        last_error = None
        for attempt in range(self.query_config.retry_count):
            try:
                start_time = time.time()

                async with httpx.AsyncClient(timeout=self.query_config.timeout) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers=self.headers,
                    )

                latency_ms = int((time.time() - start_time) * 1000)

                # Handle response status
                if response.status_code == 401:
                    raise RAGEvalError(
                        ErrorCode.E201_RAG_AUTH_FAILED,
                        "RAG API authentication failed"
                    )
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    raise RAGEvalError(
                        ErrorCode.E203_RAG_RATE_LIMIT,
                        f"Rate limit exceeded. Retry after {retry_after}s"
                    )
                elif response.status_code >= 400:
                    raise RAGEvalError(
                        ErrorCode.E204_RAG_INVALID_RESPONSE,
                        f"RAG API error: {response.status_code} - {response.text}"
                    )

                # Parse response
                data = response.json()
                return self._parse_response(question, data, latency_ms)

            except httpx.ConnectError as e:
                last_error = RAGEvalError(
                    ErrorCode.E200_RAG_CONNECTION_FAILED,
                    f"Failed to connect to RAG API: {str(e)}"
                )
            except httpx.TimeoutException:
                last_error = RAGEvalError(
                    ErrorCode.E202_RAG_TIMEOUT,
                    f"RAG API request timed out after {self.query_config.timeout}s"
                )
            except RAGEvalError:
                raise
            except Exception as e:
                last_error = RAGEvalError(
                    ErrorCode.E204_RAG_INVALID_RESPONSE,
                    f"RAG API error: {str(e)}"
                )

            # Exponential backoff
            if attempt < self.query_config.retry_count - 1:
                wait_time = self.query_config.retry_backoff ** attempt
                logger.warning(
                    "RAG query failed, retrying",
                    attempt=attempt + 1,
                    wait_time=wait_time
                )
                await asyncio.sleep(wait_time)

        raise last_error

    async def query_batch(
        self,
        questions: List[str],
        collection: Optional[str] = None,
        top_k: int = 5,
        progress_callback: Optional[callable] = None
    ) -> List[RAGResponse]:
        """
        Query RAG API for multiple questions.

        Args:
            questions: List of questions
            collection: Document collection
            top_k: Context chunks per query
            progress_callback: Optional callback for progress updates

        Returns:
            List of RAGResponse objects
        """
        results = []
        for i, question in enumerate(questions):
            try:
                response = await self.query(question, collection, top_k)
                results.append(response)
            except RAGEvalError as e:
                logger.error("Query failed", question=question[:50], error=str(e))
                # Return empty response for failed query
                results.append(RAGResponse(
                    question=question,
                    answer=f"Error: {e.message}",
                    contexts=[],
                ))

            if progress_callback:
                progress_callback(i + 1, len(questions))

        return results

    def query_sync(
        self,
        question: str,
        collection: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> RAGResponse:
        """Synchronous wrapper for query."""
        return asyncio.run(self.query(question, collection, top_k, filters))
