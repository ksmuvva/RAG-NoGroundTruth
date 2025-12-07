"""Pytest configuration and fixtures."""
import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_questions():
    """Sample questions for testing."""
    return [
        "What is machine learning?",
        "How does neural network work?",
        "Explain cloud computing benefits.",
    ]


@pytest.fixture
def sample_rag_response():
    """Sample RAG response for testing."""
    return {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of AI...",
        "contexts": [
            "ML is a branch of AI focusing on data and algorithms.",
            "ML algorithms learn from historical data.",
        ],
        "sources": [
            {"document_id": "doc1", "document_name": "ml_guide.pdf"}
        ]
    }


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from src.config.settings import Settings
    return Settings()
