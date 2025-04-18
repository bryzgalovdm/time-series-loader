import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .error_handling import FileParsingError


# Metadata dataclass
@dataclass
class FileMetadata:
    filepath: Path
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)


class MetadataExtractor(ABC):
    """Abstract base class for extracting metadata from filenames."""

    @abstractmethod
    def extract_metadata(self, filepath: Path) -> FileMetadata:
        """
        Extract metadata from a file path.

        Args:
            filepath: Path object representing the file

        Returns:
            FileMetadata object containing extracted metadata

        Raises:
            FileParsingError: If metadata extraction fails
        """
        pass

    @abstractmethod
    def is_valid_filename(self, filename: str) -> bool:
        """
        Check if a filename is valid according to this extractor.

        Args:
            filename: Name of the file to check

        Returns:
            Boolean indicating if the filename is valid
        """
        pass


class DefaultMetadataExtractor(MetadataExtractor):
    """Default implementation of MetadataExtractor using configurable parameters."""

    def __init__(
        self,
        filename_pattern: str = None,
        datetime_format: str = "%m-%d-%Y %H_%M_%S",
        start_time_group: int = 1,
        end_time_group: int = 2,
    ):
        """
        Initialize with configurable parameters.

        Args:
            filename_pattern: Regex pattern for filename validation and extraction
            datetime_format: Format string for parsing datetime from filename
            start_time_group: Group number in regex for start time
            end_time_group: Group number in regex for end time
        """
        self.datetime_format = datetime_format
        self.start_time_group = start_time_group
        self.end_time_group = end_time_group

        # Use no default pattern if none provided
        if filename_pattern is None:
            # Expects time pattern
            filename_pattern = r"^.+$"

        self.filename_pattern = re.compile(filename_pattern)

    def is_valid_filename(self, filename: str) -> bool:
        """Check if filename matches the pattern."""
        return bool(self.filename_pattern.match(filename))

    def extract_metadata(self, filepath: Path) -> FileMetadata:
        """
        Extract metadata from filepath using the configured pattern.
        If start/end times aren't found in the filename, logs a warning and uses None.
        """
        filename = filepath.name
        match = self.filename_pattern.match(filename)

        if not match:
            raise FileParsingError(
                filepath=str(filepath),
                reason=f"Filename {filename} doesn't match expected pattern",
            )

        try:
            # Create metadata with potentially None values
            return FileMetadata(
                filepath=filepath,
                additional_metadata={"has_timestamp_metadata": False},
            )

        except Exception as e:
            if isinstance(e, FileParsingError):
                raise
            raise FileParsingError(
                filepath=str(filepath), reason=f"Error extracting metadata: {str(e)}"
            )


class TimeMetadataExtractor(MetadataExtractor):
    """Default implementation of MetadataExtractor using configurable parameters."""

    def __init__(
        self,
        filename_pattern: str = None,
        datetime_format: str = "%m-%d-%Y %H_%M_%S",
        start_time_group: int = 1,
        end_time_group: int = 2,
    ):
        """
        Initialize with configurable parameters.

        Args:
            filename_pattern: Regex pattern for filename validation and extraction
            datetime_format: Format string for parsing datetime from filename
            start_time_group: Group number in regex for start time
            end_time_group: Group number in regex for end time
        """
        self.datetime_format = datetime_format
        self.start_time_group = start_time_group
        self.end_time_group = end_time_group

        # Use no default pattern if none provided
        if filename_pattern is None:
            # Expects time pattern DD-MM-YYYY HH_MM_SS or MM-DD-YYYY HH_MM_SS
            filename_pattern = r".*?(\d{2}-\d{2}-\d{4}\s+\d{2}_\d{2}_\d{2})\s+-\s+(\d{2}-\d{2}-\d{4}\s+\d{2}_\d{2}_\d{2})\.csv"

        self.filename_pattern = re.compile(filename_pattern)

    def _parse_datetime(self, datetime_str: str) -> datetime:
        """Parse datetime string to datetime object."""
        try:
            return datetime.strptime(datetime_str, self.datetime_format)
        except ValueError as e:
            raise FileParsingError(
                filepath="datetime_string",
                reason=f"Invalid datetime format: {datetime_str}",
            ) from e

    def is_valid_filename(self, filename: str) -> bool:
        """Check if filename matches the pattern."""
        return bool(self.filename_pattern.match(filename))

    def extract_metadata(self, filepath: Path) -> FileMetadata:
        """
        Extract metadata from filepath using the configured pattern.
        If start/end times aren't found in the filename, logs a warning and uses None.
        """
        filename = filepath.name
        match = self.filename_pattern.match(filename)

        if not match:
            raise FileParsingError(
                filepath=str(filepath),
                reason=f"Filename {filename} doesn't match expected pattern",
            )

        try:
            # Initialize start and end times as None
            start_time = None
            end_time = None

            # Try to extract start and end times
            # Check if we have enough groups in the match
            if len(match.groups()) >= self.start_time_group:
                try:
                    start_time_str = match.group(self.start_time_group)
                    if start_time_str:
                        start_time = self._parse_datetime(start_time_str)
                except (IndexError, ValueError):
                    pass

            if len(match.groups()) >= self.end_time_group:
                try:
                    end_time_str = match.group(self.end_time_group)
                    if end_time_str:
                        end_time = self._parse_datetime(end_time_str)
                except (IndexError, ValueError):
                    pass

            # Create metadata with potentially None values
            return FileMetadata(
                filepath=filepath,
                start_time=start_time,
                end_time=end_time,
                additional_metadata={
                    "has_timestamp_metadata": bool(
                        start_time is not None and end_time is not None
                    )
                },
            )

        except Exception as e:
            if isinstance(e, FileParsingError):
                raise
            raise FileParsingError(
                filepath=str(filepath), reason=f"Error extracting metadata: {str(e)}"
            )


class FileFilter(ABC):
    """Interface for custom file filtering logic."""

    @abstractmethod
    def is_valid(self, filepath: Path) -> bool:
        """
        Check if a file is valid according to this filter.

        Args:
            filepath: Path object representing the file

        Returns:
            Boolean indicating if the file is valid
        """
        pass


class DefaultFileFilter(FileFilter):
    """Default implementation of FileFilter that checks file extension and basic properties."""

    def __init__(self, valid_extensions: List[str] = None):
        """
        Initialize with configurable parameters.

        Args:
            valid_extensions: List of valid file extensions (default: ['.csv'])
        """
        self.valid_extensions = valid_extensions or [".csv"]

    def is_valid(self, filepath: Path) -> bool:
        """Check if file is valid based on extension and basic properties."""
        try:
            # Check if it's a file
            if not filepath.is_file():
                return False

            # Check extension
            if filepath.suffix.lower() not in self.valid_extensions:
                return False

            # Check if we have read permission
            if not os.access(filepath, os.R_OK):
                return False

            # Check if file is not empty
            if filepath.stat().st_size == 0:
                return False

            return True

        except Exception:
            # Any error means the file is not valid
            return False


class MetadataFileFilter(FileFilter):
    """File filter that combines basic file checks with metadata extraction validation."""

    def __init__(
        self, metadata_extractor: MetadataExtractor, valid_extensions: List[str] = None
    ):
        """
        Initialize with metadata extractor and optional parameters.

        Args:
            metadata_extractor: MetadataExtractor to use for filename validation
            valid_extensions: List of valid file extensions (default: ['.csv'])
        """
        self.metadata_extractor = metadata_extractor
        self.base_filter = DefaultFileFilter(valid_extensions)

    def is_valid(self, filepath: Path) -> bool:
        """Check if file is valid based on basic checks and metadata extraction."""
        # First apply basic file checks
        if not self.base_filter.is_valid(filepath):
            return False

        # Then check if filename matches the pattern
        return self.metadata_extractor.is_valid_filename(filepath.name)
