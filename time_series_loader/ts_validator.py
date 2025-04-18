from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from .file_metadata_parser import FileMetadata


class ValidationStrategy(Enum):
    """Enum for different validation strategies."""

    NONE = "none"  # No validation
    LENIENT = "lenient"  # Allow gaps but no overlaps
    STRICT = "strict"  # No gaps or overlaps allowed
    CUSTOM = "custom"  # Custom validation rules


# Validation result dataclass
@dataclass
class ValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class TimeValidationIssue:
    """Class for storing time validation issues"""

    issue_type: str  # 'gap' or 'overlap'
    start_time: datetime
    end_time: datetime
    file1: Path
    file2: Path
    duration: timedelta


@dataclass
class TimeSeriesGap:
    """Class for storing time series gaps information"""

    start_time: datetime
    end_time: datetime
    duration: timedelta
    expected_points: int


class TimeSeriesValidator(ABC):
    """Interface for time series validation."""

    @abstractmethod
    def validate_files(
        self, metadata_list: List[FileMetadata]
    ) -> List[TimeValidationIssue]:
        """
        Validate a list of files for time series continuity.

        Args:
            metadata_list: List of FileMetadata objects

        Returns:
            List of TimeValidationIssue objects
        """
        pass

    @abstractmethod
    def is_valid_sequence(self, metadata_list: List[FileMetadata]) -> ValidationResult:
        """
        Check if a sequence of files forms a valid time series.

        Args:
            metadata_list: List of FileMetadata objects

        Returns:
            ValidationResult object
        """
        pass


class DefaultTimeSeriesValidator(TimeSeriesValidator):
    """Default implementation of TimeSeriesValidator."""

    def __init__(
        self,
        strategy: Union[ValidationStrategy, str] = ValidationStrategy.LENIENT,
        max_allowed_gap: timedelta = timedelta(minutes=15),
        allow_overlap: bool = False,
        max_allowed_overlap: timedelta = timedelta(seconds=0),
    ):
        """
        Initialize with validation parameters.

        Args:
            strategy: Validation strategy to use
            max_allowed_gap: Maximum allowed gap between files
            allow_overlap: Whether to allow overlaps between files
            max_allowed_overlap: Maximum allowed overlap between files
        """
        if isinstance(strategy, str):
            try:
                self.strategy = ValidationStrategy(strategy)
            except ValueError:
                self.strategy = ValidationStrategy.LENIENT
        else:
            self.strategy = strategy

        self.max_allowed_gap = max_allowed_gap
        self.allow_overlap = allow_overlap
        self.max_allowed_overlap = max_allowed_overlap

    def validate_files(
        self, metadata_list: List[FileMetadata]
    ) -> List[TimeValidationIssue]:
        """
        Validate files for time series continuity.

        Args:
            metadata_list: List of FileMetadata objects

        Returns:
            List of TimeValidationIssue objects
        """
        if not metadata_list or len(metadata_list) < 2:
            return []  # Nothing to validate with 0 or 1 files

        # Sort by start time
        try:
            sorted_metadata = sorted(metadata_list, key=lambda x: x.start_time)
        except TypeError:
            sorted_metadata = metadata_list

        issues = []
        # Check for gaps and overlaps
        for i in range(len(sorted_metadata) - 1):
            current = sorted_metadata[i]
            next_file = sorted_metadata[i + 1]

            # Check for gap
            try:
                if current.end_time < next_file.start_time:
                    gap = next_file.start_time - current.end_time
                    if (
                        self.strategy != ValidationStrategy.NONE
                        and gap > self.max_allowed_gap
                    ):
                        issues.append(
                            TimeValidationIssue(
                                issue_type="gap",
                                start_time=current.end_time,
                                end_time=next_file.start_time,
                                file1=current.filepath,
                                file2=next_file.filepath,
                                duration=gap,
                            )
                        )

                # Check for overlap
                elif current.end_time > next_file.start_time:
                    overlap = current.end_time - next_file.start_time
                    if (
                        self.strategy == ValidationStrategy.STRICT
                        or not self.allow_overlap
                        or overlap > self.max_allowed_overlap
                    ):
                        issues.append(
                            TimeValidationIssue(
                                issue_type="overlap",
                                start_time=next_file.start_time,
                                end_time=current.end_time,
                                file1=current.filepath,
                                file2=next_file.filepath,
                                duration=overlap,
                            )
                        )
            except TypeError:  # in case no timestamps are found in filenames
                issues.append(
                    TimeValidationIssue(
                        issue_type="no_time_info",
                        start_time=None,
                        end_time=None,
                        file1=current.filepath,
                        file2=next_file.filepath,
                        duration=timedelta(0),
                    )
                )

        return issues

    def is_valid_sequence(self, metadata_list: List[FileMetadata]) -> ValidationResult:
        """
        Check if a sequence of files forms a valid time series.

        Args:
            metadata_list: List of FileMetadata objects

        Returns:
            ValidationResult object
        """
        if self.strategy == ValidationStrategy.NONE:
            return ValidationResult(is_valid=True)

        issues = self.validate_files(metadata_list)

        if not issues:
            return ValidationResult(is_valid=True)

        # For STRICT strategy, any issue makes the sequence invalid
        if self.strategy == ValidationStrategy.STRICT:
            first_issue = issues[0]
            return ValidationResult(
                is_valid=False,
                error_message=f"{first_issue.issue_type.capitalize()} detected between files: "
                f"{first_issue.file1} and {first_issue.file2} "
                f"({first_issue.duration})",
                error_type=first_issue.issue_type,
            )

        # For LENIENT strategy, only overlaps make the sequence invalid
        if self.strategy == ValidationStrategy.LENIENT:
            overlaps = [issue for issue in issues if issue.issue_type == "overlap"]
            if overlaps:
                first_overlap = overlaps[0]
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Overlap detected between files: "
                    f"{first_overlap.file1} and {first_overlap.file2} "
                    f"({first_overlap.duration})",
                    error_type="overlap",
                )
            return ValidationResult(is_valid=True)

        # For CUSTOM strategy, use the configured parameters
        if not self.allow_overlap:
            overlaps = [issue for issue in issues if issue.issue_type == "overlap"]
            if overlaps:
                first_overlap = overlaps[0]
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Overlap detected between files: "
                    f"{first_overlap.file1} and {first_overlap.file2} "
                    f"({first_overlap.duration})",
                    error_type="overlap",
                )

        # If we got here, the sequence is valid according to the custom rules
        return ValidationResult(is_valid=True)
