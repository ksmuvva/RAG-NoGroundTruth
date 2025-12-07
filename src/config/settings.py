"""Configuration loader and settings management."""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .constants import ErrorCode


class RAGEvalError(Exception):
    """Base exception for RAG Evaluation errors."""
    def __init__(self, code: ErrorCode, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code.value}] {message}")


class AuthConfig(BaseModel):
    """Authentication configuration."""
    type: str = "api_key"
    header_name: str = "X-API-Key"
    token_env: str = "RAG_API_KEY"


class QueryConfig(BaseModel):
    """Query endpoint configuration."""
    endpoint: str = "/api/v1/query"
    method: str = "POST"
    timeout: int = 30
    retry_count: int = 3
    retry_backoff: float = 1.5
    request_mapping: Dict[str, str] = Field(default_factory=lambda: {
        "question_field": "question",
        "collection_field": "collection",
        "top_k_field": "top_k"
    })
    response_mapping: Dict[str, str] = Field(default_factory=lambda: {
        "answer_field": "answer",
        "context_field": "contexts",
        "sources_field": "sources"
    })


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""
    strategy: str = "auto"
    chunk_size: int = 512
    chunk_overlap: int = 50


class DocumentUploadConfig(BaseModel):
    """Document upload configuration."""
    endpoint: str = "/api/v1/documents/upload"
    batch_endpoint: str = "/api/v1/documents/batch"
    timeout: int = 120
    max_file_size_mb: int = 50
    allowed_extensions: list = Field(default_factory=lambda: [
        ".pdf", ".txt", ".docx", ".md", ".csv", ".json"
    ])
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    concurrency: int = 5


class RAGAPIConfig(BaseModel):
    """RAG API configuration."""
    base_url: str = "http://localhost:8000"
    auth: AuthConfig = Field(default_factory=AuthConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    document_upload: DocumentUploadConfig = Field(default_factory=DocumentUploadConfig)


class LLMConfig(BaseModel):
    """LLM (Azure OpenAI) configuration."""
    provider: str = "azure_openai"
    deployment_name: str = "gpt-4"
    api_version: str = "2024-02-15-preview"
    temperature: float = 0
    max_tokens: int = 2000
    requests_per_minute: int = 60
    retry_on_rate_limit: bool = True


class ThresholdsConfig(BaseModel):
    """Evaluation thresholds."""
    faithfulness: float = 0.7
    answer_relevancy: float = 0.7
    context_relevancy: float = 0.7


class FrameworkConfig(BaseModel):
    """Individual framework configuration."""
    enabled: bool = True


class FrameworksConfig(BaseModel):
    """Evaluation frameworks configuration."""
    ragas: FrameworkConfig = Field(default_factory=FrameworkConfig)
    deepeval: FrameworkConfig = Field(default_factory=FrameworkConfig)


class ComparisonConfig(BaseModel):
    """Framework comparison configuration."""
    discrepancy_threshold: float = 0.2
    calculate_correlation: bool = True


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    metrics: list = Field(default_factory=lambda: [
        "faithfulness", "answer_relevancy", "context_relevancy"
    ])
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    frameworks: FrameworksConfig = Field(default_factory=FrameworksConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)


class InputConfig(BaseModel):
    """Input configuration."""
    directory: str = "./data/input"
    csv_encoding: str = "utf-8"
    csv_question_column: str = "question"


class OutputConfig(BaseModel):
    """Output configuration."""
    directory: str = "./data/output"
    report_formats: list = Field(default_factory=lambda: ["csv", "json", "markdown"])
    filename_template: str = "eval_report_{timestamp}"
    timestamp_format: str = "%Y%m%d_%H%M%S"


class DocumentsConfig(BaseModel):
    """Documents directory configuration."""
    directory: str = "./data/documents"


class IOConfig(BaseModel):
    """Input/Output configuration."""
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)


class FileLoggingConfig(BaseModel):
    """File logging configuration."""
    enabled: bool = True
    path: str = "./logs/rag_eval.log"
    rotation: str = "10 MB"
    retention: int = 5


class LoggingOutputsConfig(BaseModel):
    """Logging outputs configuration."""
    console: bool = True
    file: FileLoggingConfig = Field(default_factory=FileLoggingConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "structured"
    outputs: LoggingOutputsConfig = Field(default_factory=LoggingOutputsConfig)


class DisplayConfig(BaseModel):
    """Display configuration."""
    progress_bar: bool = True
    colors: bool = True
    table_style: str = "rounded"


class Settings(BaseModel):
    """Main application settings."""
    rag_api: RAGAPIConfig = Field(default_factory=RAGAPIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)


def _expand_env_vars(value: Any) -> Any:
    """Expand environment variables in string values."""
    if isinstance(value, str):
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(var_name, default)

        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_config(config_path: str = "config.yaml") -> Settings:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Settings object with loaded configuration

    Raises:
        RAGEvalError: If configuration file not found or invalid
    """
    # Load .env file if exists
    load_dotenv()

    config_file = Path(config_path)
    if not config_file.exists():
        raise RAGEvalError(
            ErrorCode.E400_CONFIG_NOT_FOUND,
            f"Configuration file not found: {config_path}"
        )

    try:
        with open(config_file, 'r') as f:
            raw_config = yaml.safe_load(f) or {}

        # Expand environment variables
        expanded_config = _expand_env_vars(raw_config)

        # Create settings object
        return Settings(**expanded_config)

    except yaml.YAMLError as e:
        raise RAGEvalError(
            ErrorCode.E401_CONFIG_INVALID,
            f"Invalid YAML configuration: {str(e)}"
        )
    except Exception as e:
        raise RAGEvalError(
            ErrorCode.E401_CONFIG_INVALID,
            f"Failed to load configuration: {str(e)}"
        )


def get_azure_openai_config() -> Dict[str, str]:
    """Get Azure OpenAI configuration from environment."""
    load_dotenv()

    config = {
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "deployment_name": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    }

    if not config["api_key"]:
        raise RAGEvalError(
            ErrorCode.E402_MISSING_API_KEY,
            "AZURE_OPENAI_API_KEY environment variable not set"
        )

    if not config["endpoint"]:
        raise RAGEvalError(
            ErrorCode.E402_MISSING_API_KEY,
            "AZURE_OPENAI_ENDPOINT environment variable not set"
        )

    return config


def get_rag_api_key() -> Optional[str]:
    """Get RAG API key from environment."""
    load_dotenv()
    return os.environ.get("RAG_API_KEY")
