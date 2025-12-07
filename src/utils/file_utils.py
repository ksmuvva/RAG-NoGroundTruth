"""File I/O utilities."""
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.constants import SUPPORTED_EXTENSIONS
from src.utils.validators import detect_encoding, validate_file_exists


def get_mime_type(file_path: str) -> str:
    """
    Get MIME type of a file.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def read_text_file(file_path: str, encoding: Optional[str] = None) -> str:
    """
    Read text file with encoding detection.

    Args:
        file_path: Path to the file
        encoding: Optional encoding (auto-detect if not provided)

    Returns:
        File contents as string
    """
    validate_file_exists(file_path)

    if encoding is None:
        encoding = detect_encoding(file_path)

    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def read_binary_file(file_path: str) -> bytes:
    """
    Read file as binary.

    Args:
        file_path: Path to the file

    Returns:
        File contents as bytes
    """
    validate_file_exists(file_path)

    with open(file_path, 'rb') as f:
        return f.read()


def write_json(data: Any, file_path: str, indent: int = 2) -> str:
    """
    Write data to JSON file.

    Args:
        data: Data to write
        file_path: Output file path
        indent: JSON indentation

    Returns:
        Absolute path to written file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)

    return str(path.absolute())


def write_text(content: str, file_path: str) -> str:
    """
    Write text content to file.

    Args:
        content: Text content
        file_path: Output file path

    Returns:
        Absolute path to written file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

    return str(path.absolute())


def scan_directory(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[Path]:
    """
    Scan directory for files with specified extensions.

    Args:
        directory: Directory path to scan
        extensions: List of extensions to include (with dot)
        recursive: Include subdirectories

    Returns:
        List of file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    extensions = extensions or SUPPORTED_EXTENSIONS
    extensions = [ext.lower() for ext in extensions]

    files = []
    pattern = "**/*" if recursive else "*"

    for path in dir_path.glob(pattern):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)

    return sorted(files)


def generate_output_filename(
    template: str,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    extension: str = ""
) -> str:
    """
    Generate output filename from template.

    Args:
        template: Filename template with {timestamp} placeholder
        timestamp_format: strftime format for timestamp
        extension: File extension to append

    Returns:
        Generated filename
    """
    timestamp = datetime.now().strftime(timestamp_format)
    filename = template.format(timestamp=timestamp)

    if extension and not filename.endswith(extension):
        filename = f"{filename}{extension}"

    return filename


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get file information.

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file info
    """
    path = Path(file_path)

    return {
        "name": path.name,
        "stem": path.stem,
        "extension": path.suffix,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "mime_type": get_mime_type(file_path),
        "absolute_path": str(path.absolute()),
    }
