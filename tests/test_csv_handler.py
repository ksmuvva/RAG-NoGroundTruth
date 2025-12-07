"""Tests for CSV handler."""
import os
import tempfile

import pytest

from src.config.settings import RAGEvalError
from src.input.csv_handler import load_questions_from_csv, save_responses_to_csv


class TestCSVHandler:
    """Test suite for CSV handler functions."""

    def test_load_questions_from_csv_success(self):
        """Test loading questions from valid CSV."""
        # Create temp CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('question\n"What is AI?"\n"How does ML work?"\n')
            temp_path = f.name

        try:
            questions = load_questions_from_csv(temp_path)
            assert len(questions) == 2
            assert questions[0] == "What is AI?"
            assert questions[1] == "How does ML work?"
        finally:
            os.unlink(temp_path)

    def test_load_questions_missing_column(self):
        """Test error when question column is missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('other_column\n"Some data"\n')
            temp_path = f.name

        try:
            with pytest.raises(RAGEvalError) as exc_info:
                load_questions_from_csv(temp_path)
            assert "E101" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_load_questions_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(RAGEvalError) as exc_info:
            load_questions_from_csv("nonexistent.csv")
        assert "E103" in str(exc_info.value)

    def test_load_questions_empty_csv(self):
        """Test error when CSV is empty."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('question\n')
            temp_path = f.name

        try:
            with pytest.raises(RAGEvalError) as exc_info:
                load_questions_from_csv(temp_path)
            assert "E102" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_save_responses_to_csv(self):
        """Test saving responses to CSV."""
        data = [
            {"question": "Q1", "response": "A1", "context": "C1"},
            {"question": "Q2", "response": "A2", "context": "C2"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "responses.csv")
            result_path = save_responses_to_csv(data, output_path)

            assert os.path.exists(result_path)

            # Verify content
            import pandas as pd
            df = pd.read_csv(result_path)
            assert len(df) == 2
            assert list(df.columns) == ["question", "response", "context"]
