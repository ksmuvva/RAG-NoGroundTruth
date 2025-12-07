"""Document file processor."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PyPDF2
from docx import Document

from src.config.constants import ErrorCode, MAX_FILE_SIZE_BYTES, SUPPORTED_EXTENSIONS
from src.config.settings import RAGEvalError
from src.utils.file_utils import get_file_info, get_mime_type, read_binary_file, read_text_file
from src.utils.validators import (
    detect_encoding,
    validate_file_exists,
    validate_file_extension,
    validate_file_size,
)


class DocumentHandler:
    """Handler for processing various document types."""

    def __init__(
        self,
        max_file_size_mb: int = 50,
        allowed_extensions: Optional[List[str]] = None
    ):
        """
        Initialize document handler.

        Args:
            max_file_size_mb: Maximum file size in MB
            allowed_extensions: List of allowed file extensions
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.allowed_extensions = allowed_extensions or SUPPORTED_EXTENSIONS

    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a document for upload.

        Args:
            file_path: Path to document file

        Returns:
            Document info dict

        Raises:
            RAGEvalError: If validation fails
        """
        # Check file exists
        validate_file_exists(file_path)

        # Check extension
        validate_file_extension(file_path, self.allowed_extensions)

        # Check size
        validate_file_size(file_path, self.max_file_size_bytes)

        return get_file_info(file_path)

    def prepare_for_upload(self, file_path: str) -> Tuple[bytes, str, Dict[str, Any]]:
        """
        Prepare document for upload to RAG API.

        Args:
            file_path: Path to document file

        Returns:
            Tuple of (file_bytes, content_type, metadata)

        Raises:
            RAGEvalError: If preparation fails
        """
        # Validate first
        info = self.validate_document(file_path)

        # Read file
        file_bytes = read_binary_file(file_path)

        # Get content type
        content_type = get_mime_type(file_path)

        # Prepare metadata
        metadata = {
            "filename": info["name"],
            "extension": info["extension"],
            "size_bytes": info["size_bytes"],
        }

        return file_bytes, content_type, metadata

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from document.

        Args:
            file_path: Path to document file

        Returns:
            Extracted text content
        """
        validate_file_exists(file_path)
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return self._extract_pdf_text(file_path)
        elif ext == '.docx':
            return self._extract_docx_text(file_path)
        elif ext in ['.txt', '.md', '.csv', '.json']:
            return self._extract_plain_text(file_path)
        else:
            raise RAGEvalError(
                ErrorCode.E104_UNSUPPORTED_FILE_TYPE,
                f"Cannot extract text from: {ext}"
            )

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text_parts = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        except Exception as e:
            raise RAGEvalError(
                ErrorCode.E106_ENCODING_ERROR,
                f"Failed to extract PDF text: {str(e)}"
            )

    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text_parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            return "\n".join(text_parts)
        except Exception as e:
            raise RAGEvalError(
                ErrorCode.E106_ENCODING_ERROR,
                f"Failed to extract DOCX text: {str(e)}"
            )

    def _extract_plain_text(self, file_path: str) -> str:
        """Extract text from plain text file."""
        encoding = detect_encoding(file_path)
        return read_text_file(file_path, encoding)


def scan_documents(
    directory: str,
    recursive: bool = False,
    extensions: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Scan directory for documents.

    Args:
        directory: Directory path
        recursive: Include subdirectories
        extensions: File extensions to include

    Returns:
        List of document info dicts
    """
    from src.utils.file_utils import scan_directory

    extensions = extensions or SUPPORTED_EXTENSIONS
    files = scan_directory(directory, extensions, recursive)

    documents = []
    for file_path in files:
        try:
            info = get_file_info(str(file_path))
            documents.append(info)
        except Exception:
            continue

    return documents
