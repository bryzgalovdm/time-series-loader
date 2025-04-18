import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .error_handling import FileParsingError
from .file_metadata_parser import FileMetadata, MetadataExtractor
from .ts_validator import ValidationResult


class DataTransformer(ABC):
    """Interface for custom data transformation during loading."""

    @abstractmethod
    def transform(self, df: pd.DataFrame, metadata: FileMetadata) -> pd.DataFrame:
        """
        Transform a DataFrame during the loading process.

        Args:
            df: DataFrame to transform
            metadata: Metadata for the file being loaded

        Returns:
            Transformed DataFrame
        """
        pass


class DefaultDataTransformer(DataTransformer):
    """Default implementation that adds metadata columns."""

    def transform(self, df: pd.DataFrame, metadata: FileMetadata) -> pd.DataFrame:
        """Add standard metadata columns to the DataFrame."""
        df = df.copy()
        df["source_file"] = str(metadata.filepath)
        df["file_start_time"] = metadata.start_time
        df["file_end_time"] = metadata.end_time
        return df


class PostProcessingHook(ABC):
    """Interface for post-processing hooks."""

    @abstractmethod
    def process(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """
        Process the DataFrame after loading and concatenation.

        Args:
            df: DataFrame to process
            context: Processing context information

        Returns:
            Processed DataFrame
        """
        pass


class NoOpPostProcessingHook(PostProcessingHook):
    """Default implementation that does nothing."""

    def process(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Return the DataFrame unchanged."""
        return df


class FileValidator(ABC):
    """Interface for custom file validation logic."""

    @abstractmethod
    def validate(self, filepath: Path, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate a file.

        Args:
            filepath: Path to the file to validate
            context: Validation context information

        Returns:
            ValidationResult object
        """
        pass


class CompositeFileValidator(FileValidator):
    """Validator that combines multiple validators."""

    def __init__(self, validators: List[FileValidator]):
        """
        Initialize with a list of validators.

        Args:
            validators: List of FileValidator objects
        """
        self.validators = validators

    def validate(self, filepath: Path, context: Dict[str, Any]) -> ValidationResult:
        """
        Run all validators and return the first failure or success.

        Args:
            filepath: Path to the file to validate
            context: Validation context information

        Returns:
            ValidationResult object
        """
        for validator in self.validators:
            result = validator.validate(filepath, context)
            if not result.is_valid:
                return result

        return ValidationResult(is_valid=True)


# Example custom data transformer
class TimestampNormalizer(DataTransformer):
    """Data transformer that normalizes timestamp columns."""

    def __init__(
        self, timestamp_columns: List[str], target_format: str = "%Y-%m-%d %H:%M:%S"
    ):
        """
        Initialize with column names and target format.

        Args:
            timestamp_columns: List of column names containing timestamps
            target_format: Target datetime format
        """
        self.timestamp_columns = timestamp_columns
        self.target_format = target_format

    def transform(self, df: pd.DataFrame, metadata: FileMetadata) -> pd.DataFrame:
        """Normalize timestamp columns to a consistent format."""
        df = df.copy()

        # Add standard metadata columns
        df["source_file"] = str(metadata.filepath)
        df["file_start_time"] = metadata.start_time
        df["file_end_time"] = metadata.end_time

        # Normalize timestamp columns
        for col in self.timestamp_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass  # Skip if conversion fails

        return df


# Example custom post-processing hook
class OutlierRemovalHook(PostProcessingHook):
    """Post-processing hook that removes outliers."""

    def __init__(self, numeric_columns: List[str], threshold: float = 3.0):
        """
        Initialize with column names and threshold.

        Args:
            numeric_columns: List of numeric column names to check for outliers
            threshold: Z-score threshold for outlier detection
        """
        self.numeric_columns = numeric_columns
        self.threshold = threshold

    def process(self, df: pd.DataFrame, context: Dict[str, Any]) -> pd.DataFrame:
        """Remove outliers from numeric columns."""
        if df.empty:
            return df

        df_clean = df.copy()
        outliers_removed = 0

        for col in self.numeric_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Calculate z-scores
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:  # Skip if standard deviation is zero
                    continue

                z_scores = (df[col] - mean) / std

                # Identify outliers
                outliers = abs(z_scores) > self.threshold
                outliers_count = outliers.sum()

                if outliers_count > 0:
                    df_clean = df_clean[~outliers]
                    outliers_removed += outliers_count

        # Add outlier removal information to context
        if "processing_stats" not in context:
            context["processing_stats"] = {}
        context["processing_stats"]["outliers_removed"] = outliers_removed

        return df_clean


# Example custom file validator
class FileContentValidator(FileValidator):
    """Validator that checks file content."""

    def __init__(self, required_headers: List[str], min_rows: int = 1):
        """
        Initialize with required headers and minimum row count.

        Args:
            required_headers: List of column headers that must be present
            min_rows: Minimum number of data rows required
        """
        self.required_headers = required_headers
        self.min_rows = min_rows

    def validate(self, filepath: Path, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate file content.

        Args:
            filepath: Path to the file to validate
            context: Validation context information

        Returns:
            ValidationResult object
        """
        try:
            # Read just the header and first few rows
            df = pd.read_csv(filepath, nrows=self.min_rows + 1)

            # Check for required headers
            missing_headers = [
                header for header in self.required_headers if header not in df.columns
            ]
            if missing_headers:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required headers: {', '.join(missing_headers)}",
                    error_type="missing_headers",
                )

            # Check row count
            if len(df) < self.min_rows:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File has {len(df)} rows, but {self.min_rows} are required",
                    error_type="insufficient_rows",
                )

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Error validating file content: {str(e)}",
                error_type="validation_error",
            )


# Example custom metadata extractor
class RegexMetadataExtractor(MetadataExtractor):
    """Metadata extractor that uses custom regex patterns."""

    def __init__(
        self,
        filename_pattern: str,
        datetime_format: str,
        start_time_group: int,
        end_time_group: int,
        additional_groups: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize with custom regex pattern and group mappings.

        Args:
            filename_pattern: Regex pattern for filename validation and extraction
            datetime_format: Format string for parsing datetime from filename
            start_time_group: Group number in regex for start time
            end_time_group: Group number in regex for end time
            additional_groups: Optional mapping of metadata field names to group numbers
        """
        self.filename_pattern = re.compile(filename_pattern)
        self.datetime_format = datetime_format
        self.start_time_group = start_time_group
        self.end_time_group = end_time_group
        self.additional_groups = additional_groups or {}

    def is_valid_filename(self, filename: str) -> bool:
        """Check if filename matches the pattern."""
        return bool(self.filename_pattern.match(filename))

    def extract_metadata(self, filepath: Path) -> FileMetadata:
        """Extract metadata from filepath using the configured pattern."""
        filename = filepath.name
        match = self.filename_pattern.match(filename)

        if not match:
            raise FileParsingError(
                filepath=str(filepath),
                reason=f"Filename {filename} doesn't match expected pattern",
            )

        try:
            # Parse start and end times
            start_time = datetime.strptime(
                match.group(self.start_time_group), self.datetime_format
            )
            end_time = datetime.strptime(
                match.group(self.end_time_group), self.datetime_format
            )

            # Validate start time is before end time
            if start_time >= end_time:
                raise FileParsingError(
                    filepath=str(filepath),
                    reason=f"Start time {start_time} is not before end time {end_time} in {filename}",
                )

            # Extract additional metadata
            additional_metadata = {}
            for field_name, group_num in self.additional_groups.items():
                if group_num <= len(match.groups()):
                    additional_metadata[field_name] = match.group(group_num)

            return FileMetadata(
                filepath=filepath,
                start_time=start_time,
                end_time=end_time,
                additional_metadata=additional_metadata,
            )

        except Exception as e:
            if isinstance(e, FileParsingError):
                raise
            raise FileParsingError(
                filepath=str(filepath), reason=f"Error extracting metadata: {str(e)}"
            )
