"""Input validation utilities."""
import os
from pathlib import Path
from typing import List, Optional

import chardet

from src.config.constants import (
    ErrorCode,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
)
from src.config.settings import RAGEvalError


def validate_file_exists(file_path: str) -> Path:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file

    Returns:
        Path object if file exists

    Raises:
        RAGEvalError: If file not found
    """
    path = Path(file_path)
    if not path.exists():
        raise RAGEvalError(
            ErrorCode.E103_FILE_NOT_FOUND,
            f"File not found: {file_path}"
        )
    if not path.is_file():
        raise RAGEvalError(
            ErrorCode.E103_FILE_NOT_FOUND,
            f"Path is not a file: {file_path}"
        )
    return path


def validate_file_extension(
    file_path: str,
    allowed_extensions: Optional[List[str]] = None
) -> str:
    """
    Validate file extension.

    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed extensions (with dot)

    Returns:
        File extension (lowercase)

    Raises:
        RAGEvalError: If extension not supported
    """
    extensions = allowed_extensions or SUPPORTED_EXTENSIONS
    ext = Path(file_path).suffix.lower()

    if ext not in extensions:
        raise RAGEvalError(
            ErrorCode.E104_UNSUPPORTED_FILE_TYPE,
            f"Unsupported file type: {ext}. Allowed: {extensions}"
        )
    return ext


def validate_file_size(file_path: str, max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> int:
    """
    Validate file size.

    Args:
        file_path: Path to the file
        max_size_bytes: Maximum allowed size in bytes

    Returns:
        File size in bytes

    Raises:
        RAGEvalError: If file too large
    """
    size = os.path.getsize(file_path)
    if size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = size / (1024 * 1024)
        raise RAGEvalError(
            ErrorCode.E105_FILE_TOO_LARGE,
            f"File too large: {actual_mb:.2f}MB (max: {max_mb:.2f}MB)"
        )
    return size


def detect_encoding(file_path: str) -> str:
    """
    Detect file encoding.

    Args:
        file_path: Path to the file

    Returns:
        Detected encoding (default: utf-8)
    """
    with open(file_path, 'rb') as f:
        raw = f.read(10000)  # Read first 10KB for detection
        result = chardet.detect(raw)
        return result.get('encoding') or 'utf-8'


def validate_directory(dir_path: str, create: bool = False) -> Path:
    """
    Validate directory exists or create it.

    Args:
        dir_path: Path to directory
        create: Create if doesn't exist

    Returns:
        Path object

    Raises:
        RAGEvalError: If directory doesn't exist and create=False
    """
    path = Path(dir_path)
    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise RAGEvalError(
                ErrorCode.E103_FILE_NOT_FOUND,
                f"Directory not found: {dir_path}"
            )
    return path


def validate_csv_columns(columns: List[str], required_columns: List[str]) -> None:
    """
    Validate CSV has required columns.

    Args:
        columns: List of column names from CSV
        required_columns: List of required column names

    Raises:
        RAGEvalError: If required columns missing
    """
    missing = [col for col in required_columns if col not in columns]
    if missing:
        raise RAGEvalError(
            ErrorCode.E101_CSV_MISSING_COLUMN,
            f"Missing required columns: {missing}"
        )


def validate_questions(questions: List[str]) -> List[str]:
    """
    Validate list of questions.

    Args:
        questions: List of question strings

    Returns:
        Cleaned list of questions

    Raises:
        RAGEvalError: If no valid questions
    """
    # Filter empty questions
    valid = [q.strip() for q in questions if q and q.strip()]

    if not valid:
        raise RAGEvalError(
            ErrorCode.E102_EMPTY_INPUT,
            "No valid questions found"
        )

    return valid
