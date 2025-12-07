"""Tests for validators."""
import os
import tempfile

import pytest

from src.config.settings import RAGEvalError
from src.utils.validators import (
    validate_file_exists,
    validate_file_extension,
    validate_file_size,
    validate_questions,
)


class TestValidators:
    """Test suite for validator functions."""

    def test_validate_file_exists_success(self):
        """Test validation passes for existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            path = validate_file_exists(temp_path)
            assert path.exists()
        finally:
            os.unlink(temp_path)

    def test_validate_file_exists_failure(self):
        """Test validation fails for non-existing file."""
        with pytest.raises(RAGEvalError) as exc_info:
            validate_file_exists("nonexistent_file.txt")
        assert "E103" in str(exc_info.value)

    def test_validate_file_extension_success(self):
        """Test extension validation for supported types."""
        ext = validate_file_extension("document.pdf")
        assert ext == ".pdf"

        ext = validate_file_extension("data.csv")
        assert ext == ".csv"

    def test_validate_file_extension_failure(self):
        """Test extension validation fails for unsupported types."""
        with pytest.raises(RAGEvalError) as exc_info:
            validate_file_extension("file.xyz")
        assert "E104" in str(exc_info.value)

    def test_validate_file_size_success(self):
        """Test size validation passes for small files."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"small content")
            temp_path = f.name

        try:
            size = validate_file_size(temp_path, max_size_bytes=1024)
            assert size > 0
        finally:
            os.unlink(temp_path)

    def test_validate_file_size_failure(self):
        """Test size validation fails for large files."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 1000)
            temp_path = f.name

        try:
            with pytest.raises(RAGEvalError) as exc_info:
                validate_file_size(temp_path, max_size_bytes=100)
            assert "E105" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_validate_questions_success(self):
        """Test question validation with valid input."""
        questions = validate_questions(["Q1", "Q2", "  Q3  "])
        assert len(questions) == 3
        assert questions[2] == "Q3"  # Trimmed

    def test_validate_questions_empty(self):
        """Test question validation fails for empty list."""
        with pytest.raises(RAGEvalError) as exc_info:
            validate_questions([])
        assert "E102" in str(exc_info.value)

    def test_validate_questions_filters_empty(self):
        """Test question validation filters empty strings."""
        with pytest.raises(RAGEvalError) as exc_info:
            validate_questions(["", "  ", None])
        assert "E102" in str(exc_info.value)
