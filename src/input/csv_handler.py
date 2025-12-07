"""CSV file input handler."""
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config.constants import ErrorCode
from src.config.settings import RAGEvalError
from src.utils.validators import (
    detect_encoding,
    validate_csv_columns,
    validate_file_exists,
    validate_questions,
)


def load_questions_from_csv(
    file_path: str,
    question_column: str = "question",
    encoding: Optional[str] = None
) -> List[str]:
    """
    Load questions from CSV file.

    Args:
        file_path: Path to CSV file
        question_column: Name of the question column
        encoding: File encoding (auto-detect if not provided)

    Returns:
        List of question strings

    Raises:
        RAGEvalError: If validation fails
    """
    # Validate file exists
    validate_file_exists(file_path)

    # Detect encoding if not provided
    if encoding is None:
        encoding = detect_encoding(file_path)

    try:
        # Read CSV
        df = pd.read_csv(file_path, encoding=encoding)

        # Validate columns
        validate_csv_columns(df.columns.tolist(), [question_column])

        # Extract questions
        questions = df[question_column].astype(str).tolist()

        # Validate questions
        return validate_questions(questions)

    except pd.errors.EmptyDataError:
        raise RAGEvalError(
            ErrorCode.E102_EMPTY_INPUT,
            f"CSV file is empty: {file_path}"
        )
    except pd.errors.ParserError as e:
        raise RAGEvalError(
            ErrorCode.E100_INVALID_CSV,
            f"Failed to parse CSV: {str(e)}"
        )
    except UnicodeDecodeError as e:
        raise RAGEvalError(
            ErrorCode.E106_ENCODING_ERROR,
            f"Encoding error reading CSV: {str(e)}"
        )


def save_responses_to_csv(
    data: List[dict],
    output_path: str,
    include_context: bool = True
) -> str:
    """
    Save RAG responses to CSV file.

    Args:
        data: List of dicts with question, response, context
        output_path: Output file path
        include_context: Include context column

    Returns:
        Path to saved file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["question", "response"]
    if include_context:
        columns.append("context")

    df = pd.DataFrame(data)

    # Ensure all columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    df[columns].to_csv(path, index=False, encoding='utf-8')

    return str(path.absolute())


def load_responses_from_csv(file_path: str) -> List[dict]:
    """
    Load RAG responses from CSV for evaluation.

    Args:
        file_path: Path to CSV with question, response, context columns

    Returns:
        List of dicts with question, response, context
    """
    validate_file_exists(file_path)

    encoding = detect_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)

    required = ["question", "response"]
    validate_csv_columns(df.columns.tolist(), required)

    records = []
    for _, row in df.iterrows():
        record = {
            "question": str(row.get("question", "")),
            "response": str(row.get("response", "")),
            "context": str(row.get("context", "")),
        }
        records.append(record)

    return records
