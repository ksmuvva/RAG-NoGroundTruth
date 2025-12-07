"""Application constants and defaults."""
from enum import Enum


# Supported document extensions
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc', '.md', '.csv', '.json']

# File size limits
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Default values
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_CONCURRENCY = 5
DEFAULT_TIMEOUT = 30
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_BACKOFF = 1.5

# Evaluation thresholds
DEFAULT_THRESHOLD = 0.7

# Report formats
REPORT_FORMATS = ['csv', 'json', 'markdown', 'html']


class ErrorCode(Enum):
    """Error codes for the application."""
    # Input Errors (1xx)
    E100_INVALID_CSV = "E100"
    E101_CSV_MISSING_COLUMN = "E101"
    E102_EMPTY_INPUT = "E102"
    E103_FILE_NOT_FOUND = "E103"
    E104_UNSUPPORTED_FILE_TYPE = "E104"
    E105_FILE_TOO_LARGE = "E105"
    E106_ENCODING_ERROR = "E106"

    # RAG API Errors (2xx)
    E200_RAG_CONNECTION_FAILED = "E200"
    E201_RAG_AUTH_FAILED = "E201"
    E202_RAG_TIMEOUT = "E202"
    E203_RAG_RATE_LIMIT = "E203"
    E204_RAG_INVALID_RESPONSE = "E204"
    E205_RAG_UPLOAD_FAILED = "E205"
    E206_RAG_COLLECTION_NOT_FOUND = "E206"

    # Evaluation Errors (3xx)
    E300_RAGAS_INIT_FAILED = "E300"
    E301_RAGAS_EVAL_FAILED = "E301"
    E302_DEEPEVAL_INIT_FAILED = "E302"
    E303_DEEPEVAL_EVAL_FAILED = "E303"
    E304_LLM_API_ERROR = "E304"
    E305_LLM_RATE_LIMIT = "E305"

    # Config Errors (4xx)
    E400_CONFIG_NOT_FOUND = "E400"
    E401_CONFIG_INVALID = "E401"
    E402_MISSING_API_KEY = "E402"

    # Output Errors (5xx)
    E500_OUTPUT_DIR_NOT_WRITABLE = "E500"
    E501_REPORT_GENERATION_FAILED = "E501"

    # Unknown
    E999_UNKNOWN = "E999"
