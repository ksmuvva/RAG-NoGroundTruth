"""Tests for document handler."""
import os
import tempfile

import pytest

from src.config.settings import RAGEvalError
from src.input.document_handler import DocumentHandler


class TestDocumentHandler:
    """Test suite for DocumentHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = DocumentHandler(max_file_size_mb=1)

    def test_validate_document_txt(self):
        """Test validation of text document."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            info = self.handler.validate_document(temp_path)
            assert info["extension"] == ".txt"
            assert info["size_bytes"] > 0
        finally:
            os.unlink(temp_path)

    def test_validate_document_unsupported(self):
        """Test validation fails for unsupported extension."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"content")
            temp_path = f.name

        try:
            with pytest.raises(RAGEvalError) as exc_info:
                self.handler.validate_document(temp_path)
            assert "E104" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_prepare_for_upload(self):
        """Test document preparation for upload."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for upload")
            temp_path = f.name

        try:
            file_bytes, content_type, metadata = self.handler.prepare_for_upload(temp_path)
            assert file_bytes == b"Test content for upload"
            assert "text" in content_type
            assert metadata["extension"] == ".txt"
        finally:
            os.unlink(temp_path)

    def test_extract_text_txt(self):
        """Test text extraction from txt file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello World")
            temp_path = f.name

        try:
            text = self.handler.extract_text(temp_path)
            assert text == "Hello World"
        finally:
            os.unlink(temp_path)
