"""Document uploader for RAG API."""
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rich.progress import Progress, TaskID
from tqdm import tqdm

from src.api.schemas import BatchUploadResponse, UploadResponse
from src.config.constants import ErrorCode
from src.config.settings import RAGEvalError, Settings, get_rag_api_key
from src.input.document_handler import DocumentHandler, scan_documents
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentUploader:
    """Uploader for sending documents to RAG API."""

    def __init__(self, settings: Settings):
        """
        Initialize document uploader.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.base_url = settings.rag_api.base_url.rstrip('/')
        self.upload_config = settings.rag_api.document_upload
        self.auth_config = settings.rag_api.auth

        # Get API key
        self.api_key = get_rag_api_key()

        # Document handler for validation
        self.doc_handler = DocumentHandler(
            max_file_size_mb=self.upload_config.max_file_size_mb,
            allowed_extensions=self.upload_config.allowed_extensions,
        )

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {}

        if self.api_key and self.auth_config.type == "api_key":
            headers[self.auth_config.header_name] = self.api_key
        elif self.api_key and self.auth_config.type == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def upload_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> UploadResponse:
        """
        Upload document to RAG API.

        Args:
            file_path: Path to document file
            metadata: Optional metadata (title, author, tags, etc.)
            collection_name: Target collection/index name

        Returns:
            UploadResponse with document_id and status

        Raises:
            RAGEvalError: If upload fails
        """
        # Prepare document
        file_bytes, content_type, file_metadata = self.doc_handler.prepare_for_upload(file_path)

        # Merge metadata
        combined_metadata = {**file_metadata}
        if metadata:
            combined_metadata.update(metadata)

        url = f"{self.base_url}{self.upload_config.endpoint}"
        headers = self._build_headers()

        filename = Path(file_path).name

        logger.debug("Uploading document", url=url, filename=filename)

        # Retry logic
        last_error = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.upload_config.timeout) as client:
                    # Prepare multipart form data
                    files = {
                        "file": (filename, file_bytes, content_type)
                    }
                    data = {
                        "collection": collection_name or "default",
                        "chunking_strategy": self.upload_config.chunking.strategy,
                        "chunk_size": str(self.upload_config.chunking.chunk_size),
                        "chunk_overlap": str(self.upload_config.chunking.chunk_overlap),
                    }

                    if combined_metadata:
                        import json
                        data["metadata"] = json.dumps(combined_metadata)

                    response = await client.post(
                        url,
                        files=files,
                        data=data,
                        headers=headers,
                    )

                # Handle response status
                if response.status_code == 401:
                    raise RAGEvalError(
                        ErrorCode.E201_RAG_AUTH_FAILED,
                        "RAG API authentication failed"
                    )
                elif response.status_code == 429:
                    raise RAGEvalError(
                        ErrorCode.E203_RAG_RATE_LIMIT,
                        "Rate limit exceeded during upload"
                    )
                elif response.status_code >= 400:
                    raise RAGEvalError(
                        ErrorCode.E205_RAG_UPLOAD_FAILED,
                        f"Upload failed: {response.status_code} - {response.text}"
                    )

                # Parse response
                resp_data = response.json()

                return UploadResponse(
                    document_id=resp_data.get("document_id", ""),
                    filename=filename,
                    status=resp_data.get("status", "success"),
                    chunks_created=resp_data.get("chunks_created"),
                    message=resp_data.get("message"),
                )

            except httpx.ConnectError as e:
                last_error = RAGEvalError(
                    ErrorCode.E200_RAG_CONNECTION_FAILED,
                    f"Failed to connect to RAG API: {str(e)}"
                )
            except httpx.TimeoutException:
                last_error = RAGEvalError(
                    ErrorCode.E202_RAG_TIMEOUT,
                    f"Upload timed out after {self.upload_config.timeout}s"
                )
            except RAGEvalError:
                raise
            except Exception as e:
                last_error = RAGEvalError(
                    ErrorCode.E205_RAG_UPLOAD_FAILED,
                    f"Upload error: {str(e)}"
                )

            # Wait before retry
            if attempt < 2:
                await asyncio.sleep(1.5 ** attempt)

        raise last_error

    async def upload_documents_batch(
        self,
        directory_path: str,
        recursive: bool = False,
        file_filter: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        show_progress: bool = True
    ) -> BatchUploadResponse:
        """
        Upload multiple documents from directory.

        Args:
            directory_path: Path to directory containing documents
            recursive: Include subdirectories
            file_filter: List of extensions to include (e.g., ['.pdf', '.txt'])
            collection_name: Target collection
            show_progress: Show progress bar

        Returns:
            BatchUploadResponse with success/failure counts
        """
        # Scan for documents
        extensions = file_filter or self.upload_config.allowed_extensions
        documents = scan_documents(directory_path, recursive, extensions)

        if not documents:
            logger.warning("No documents found in directory", directory=directory_path)
            return BatchUploadResponse(
                total=0,
                successful=0,
                failed=0,
                results=[],
            )

        results = []
        errors = []

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.upload_config.concurrency)

        async def upload_with_semaphore(doc_info: Dict[str, Any]) -> Optional[UploadResponse]:
            async with semaphore:
                try:
                    result = await self.upload_document(
                        doc_info["absolute_path"],
                        collection_name=collection_name,
                    )
                    return result
                except RAGEvalError as e:
                    errors.append({
                        "filename": doc_info["name"],
                        "error": e.message,
                    })
                    return None

        # Process with progress bar
        if show_progress:
            with tqdm(total=len(documents), desc="Uploading documents") as pbar:
                for doc in documents:
                    result = await upload_with_semaphore(doc)
                    if result:
                        results.append(result)
                    pbar.update(1)
        else:
            tasks = [upload_with_semaphore(doc) for doc in documents]
            upload_results = await asyncio.gather(*tasks)
            results = [r for r in upload_results if r is not None]

        return BatchUploadResponse(
            total=len(documents),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
        )

    def upload_document_sync(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> UploadResponse:
        """Synchronous wrapper for upload_document."""
        return asyncio.run(self.upload_document(file_path, metadata, collection_name))

    def upload_batch_sync(
        self,
        directory_path: str,
        recursive: bool = False,
        file_filter: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> BatchUploadResponse:
        """Synchronous wrapper for upload_documents_batch."""
        return asyncio.run(self.upload_documents_batch(
            directory_path, recursive, file_filter, collection_name
        ))
