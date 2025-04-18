import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ErrorSeverity(Enum):
    """Enumeration for error severity levels"""

    CRITICAL = "CRITICAL"  # Processing cannot continue
    ERROR = "ERROR"  # Major issue but some processing possible
    WARNING = "WARNING"  # Minor issue, processing can continue
    INFO = "INFO"  # Informational message


# Custom exceptions
class ValidationError(Exception):
    """Error raised when validation fails."""

    def __init__(
        self,
        message: str,
        validation_type: str = "general",
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a ValidationError.

        Args:
            message: Error message
            validation_type: Type of validation that failed
            details: Detailed information about the validation failure
            context: Additional context information
        """
        self.validation_type = validation_type
        self.details = details or {}
        self.context = context or {}

        full_message = f"{validation_type.capitalize()} validation error: {message}"
        super().__init__(full_message)


@dataclass
class DataConsistencyError(Exception):
    """Exception raised for data structure consistency errors."""

    filepath: str
    expected_columns: list
    actual_columns: list
    message: str = "Data consistency error"

    def __str__(self):
        return (
            f"{self.message} in file '{self.filepath}': "
            f"expected columns {self.expected_columns}, "
            f"but found {self.actual_columns}"
        )


class FileDiscoveryError(Exception):
    """Error raised when file discovery fails."""

    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a FileDiscoveryError.

        Args:
            message: Error message
            path: Path being processed when the error occurred
            context: Additional context information
        """
        self.path = Path(path) if path else None
        self.context = context or {}

        # Include path in message if provided
        full_message = message
        if self.path:
            full_message = f"{message} (Path: {self.path})"

        super().__init__(full_message)


class FileParsingError(Exception):
    """Error raised when file parsing fails."""

    def __init__(
        self,
        filepath: Union[str, Path],
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a FileParsingError.

        Args:
            filepath: Path to file that failed to parse
            reason: Reason for parsing failure
            context: Additional context information
        """
        self.filepath = Path(filepath) if filepath else None
        self.reason = reason
        self.context = context or {}

        message = f"Failed to parse file: {reason}"
        if self.filepath:
            message = f"Failed to parse file {self.filepath}: {reason}"

        super().__init__(message)


class DataLoadingError(Exception):
    """Error raised when data loading fails."""

    def __init__(
        self,
        error_details: str,
        filepath: Optional[Union[str, Path]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a DataLoadingError.

        Args:
            error_details: Details about the error
            filepath: Path to file that failed to load
            context: Additional context information
        """
        self.error_details = error_details
        self.filepath = Path(filepath) if filepath else None
        self.context = context or {}

        message = f"Data loading error: {error_details}"
        if self.filepath:
            message = f"Data loading error for {self.filepath}: {error_details}"

        super().__init__(message)


class TimeValidationError(Exception):
    """Error raised when time validation fails."""

    def __init__(
        self,
        message: str,
        invalid_time: Optional[Any] = None,
        files: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a TimeValidationError.

        Args:
            message: Error message
            invalid_time: The invalid time value (if applicable)
            files: List of files involved in the validation error
            context: Additional context information
        """
        self.invalid_time = invalid_time
        self.files = files or []
        self.context = context or {}

        full_message = message
        if self.invalid_time is not None:
            full_message = f"{message} (Invalid time: {self.invalid_time})"

        super().__init__(full_message)


class ProcessingError:
    """Base class for all processing errors."""

    def __init__(
        self,
        timestamp: datetime = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        error_type: str = "ProcessingError",
        message: str = "An error occurred during processing",
        file_path: Optional[Union[str, Path]] = None,
        details: Optional[Dict[str, Any]] = None,
        stacktrace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a ProcessingError.

        Args:
            timestamp: When the error occurred (defaults to now)
            severity: Error severity level
            error_type: Type of error
            message: Error message
            file_path: Path to file that caused the error (if applicable)
            details: Additional error details
            stacktrace: Exception stacktrace
            context: Contextual information about the processing state
        """
        self.timestamp = timestamp or datetime.now()
        self.severity = severity
        self.error_type = error_type
        self.message = message
        self.file_path = Path(file_path) if file_path else None
        self.details = details or {}
        self.stacktrace = stacktrace
        self.context = context or {}

    def __str__(self) -> str:
        """String representation of the error."""
        result = f"[{self.severity.value.upper()}] {self.error_type}: {self.message}"
        if self.file_path:
            result += f" (File: {self.file_path})"
        return result

    def to_dict(self, include_stacktrace: bool = False) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        error_dict = {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "error_type": self.error_type,
            "message": self.message,
            "file_path": str(self.file_path) if self.file_path else None,
            "details": self.details,
            "context": self.context,
        }

        if include_stacktrace and self.stacktrace:
            error_dict["stacktrace"] = self.stacktrace

        return error_dict


class DataFrameProcessingError(Exception):
    """Base exception class for all processing errors"""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR):
        super().__init__(message)
        self.severity = severity
        self.timestamp = datetime.now()
        self.details: Dict[str, Any] = {}
        self.stacktrace = traceback.format_exc()
