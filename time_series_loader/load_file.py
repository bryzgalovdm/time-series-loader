import json
import logging
import os
import traceback
from dataclasses import fields
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

from .error_handling import (
    DataLoadingError,
    ErrorSeverity,
    FileDiscoveryError,
    FileParsingError,
    ProcessingError,
    TimeValidationError,
    ValidationError,
)
from .file_metadata_parser import (
    DefaultMetadataExtractor,
    FileMetadata,
    MetadataExtractor,
    MetadataFileFilter,
)
from .ts_config import (
    ColumnNamingConfig,
    FileDiscoveryConfig,
    LoadingConfig,
    TimeSeriesConfig,
)
from .ts_extensions import (
    DataTransformer,
    DefaultDataTransformer,
    FileValidator,
    PostProcessingHook,
)
from .ts_validator import (
    DefaultTimeSeriesValidator,
    TimeSeriesGap,
    TimeSeriesValidator,
    TimeValidationIssue,
    ValidationResult,
    ValidationStrategy,
)


class FileDataFrame:
    def __init__(
        self,
        base_path: Optional[str] = None,
        files: Optional[List[Union[str, Path]]] = None,
        streamlit_files: Optional[Union[UploadedFile, List[UploadedFile]]] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
        discovery_config: Optional[FileDiscoveryConfig] = None,
        column_config: Optional[ColumnNamingConfig] = None,
        time_series_config: Optional[TimeSeriesConfig] = None,
        loading_config: Optional[LoadingConfig] = None,
        log_file: Optional[str] = None,
        data_transformer: Optional[DataTransformer] = None,
        post_processing_hooks: Optional[List[PostProcessingHook]] = None,
        file_validator: Optional[FileValidator] = None,
        extension_points: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize FileDataFrame with either base_path, files, or streamlit_files.

        Args:
            base_path: Optional base directory path to search for files
            files: Optional list of file paths to process directly
            streamlit_files: Optional list of Streamlit uploaded files
            metadata_extractor: Optional custom metadata extractor for parsing filenames
            discovery_config: Configuration for file discovery (glob pattern, recursion, filtering)
            column_config: Configuration for column naming (cleaning, renaming)
            time_series_config: Configuration for time series validation
            loading_config: Configuration for data loading (delimiter, encoding, etc.)
            log_file: Optional path to log file for detailed logging
            data_transformer: Optional custom data transformer
            post_processing_hooks: Optional list of post-processing hooks
            file_validator: Optional custom file validator
            extension_points: Optional dictionary of additional extension points

        Raises:
            ValueError: If input parameters are invalid or conflicting
        """
        # Validate input parameters
        if base_path is None and files is None and streamlit_files is None:
            raise ValueError(
                "Either base_path, files or streamlit_files must be provided"
            )

        if sum(x is not None for x in [base_path, files, streamlit_files]) > 1:
            raise ValueError(
                "Cannot specify more than one of: base_path, files, streamlit_files"
            )

        # Set up file sources
        self.base_path = Path(base_path) if base_path else None
        self.files = [Path(f) for f in files] if files else None
        self.streamlit_files = streamlit_files if streamlit_files else None

        # Initialize empty metadata and dataframe
        self.metadata: List[FileMetadata] = []
        self.dataframe: Optional[pd.DataFrame] = None

        # Set up metadata extractor
        self.metadata_extractor = metadata_extractor or DefaultMetadataExtractor()

        # Set up configurations with defaults if not provided
        self.discovery_config = discovery_config or FileDiscoveryConfig()
        self.column_config = column_config or ColumnNamingConfig()
        self.time_series_config = time_series_config or TimeSeriesConfig()
        self.loading_config = loading_config or LoadingConfig()

        # Set up file filter if not provided in discovery config
        if self.discovery_config.file_filter is None:
            self.discovery_config.file_filter = MetadataFileFilter(
                self.metadata_extractor
            )

        # Set up time series validator if not provided in time series config
        if self.time_series_config.time_series_validator is None:
            self.time_series_config.time_series_validator = DefaultTimeSeriesValidator(
                strategy=self.time_series_config.validation_strategy,
                max_allowed_gap=self.time_series_config.max_allowed_gap,
                allow_overlap=self.time_series_config.allow_overlap,
                max_allowed_overlap=self.time_series_config.max_allowed_overlap,
            )

        # Initialize error tracking
        self.errors: List[ProcessingError] = []

        # Set up extension points
        self.data_transformer = data_transformer or DefaultDataTransformer()
        self.post_processing_hooks = post_processing_hooks or []
        self.file_validator = file_validator
        self.extension_points = extension_points or {}

        # Set up logging
        self._setup_logging(log_file)

    def _setup_logging(self, log_file: Optional[str]) -> None:
        """
        Setup logging configuration.

        Args:
            log_file: Optional path to log file
        """
        self.logger = logging.getLogger(f"FileDataFrame_{id(self)}")
        self.logger.setLevel(logging.DEBUG)

        # Create formatters and handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if log_file is provided
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _add_error(self, error: ProcessingError) -> None:
        """
        Add error to the error list and log it with enhanced information.

        Args:
            error: ProcessingError object to add
        """
        self.errors.append(error)

        # Build log message with more context
        log_message = f"{error.error_type}: {error.message}"
        if error.file_path:
            log_message += f" (File: {error.file_path})"

        # Add context information to log message if available
        if error.context:
            context_str = ", ".join(
                f"{k}={v}"
                for k, v in error.context.items()
                if k not in ["stacktrace"] and v is not None
            )
            if context_str:
                log_message += f" [Context: {context_str}]"

        # Log with appropriate severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _handle_error(
        self,
        e: Exception,
        context: str,
        file_path: Optional[Path] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Handle an error during processing with enhanced context.

        Args:
            e: Exception that occurred
            context: Context where the error occurred
            file_path: Optional path to file being processed
            additional_context: Additional contextual information
        """
        # Collect context information
        ctx = {"operation": context, "timestamp": datetime.now().isoformat()}

        # Add configuration information to context
        if hasattr(self, "discovery_config"):
            ctx["discovery_config"] = {
                "glob_pattern": self.discovery_config.glob_pattern,
                "recursive": self.discovery_config.recursive,
            }

        # Add file information to context
        if file_path:
            ctx["file_info"] = {
                "path": str(file_path),
                "exists": file_path.exists() if isinstance(file_path, Path) else None,
                "size": (
                    file_path.stat().st_size
                    if isinstance(file_path, Path) and file_path.exists()
                    else None
                ),
            }

        # Add any additional context
        if additional_context:
            ctx.update(additional_context)

        # Determine severity based on exception type
        if isinstance(e, (FileDiscoveryError, TimeValidationError)):
            severity = ErrorSeverity.CRITICAL
        elif isinstance(e, (FileParsingError, DataLoadingError)):
            severity = ErrorSeverity.ERROR
        else:
            severity = ErrorSeverity.WARNING

        # Extract context from exception if available
        exception_context = {}
        if hasattr(e, "context") and isinstance(e.context, dict):
            exception_context = e.context

        # Merge contexts
        merged_context = {**ctx, **exception_context}

        # Create error object with enhanced context
        error = ProcessingError(
            timestamp=datetime.now(),
            severity=severity,
            error_type=type(e).__name__,
            message=str(e),
            file_path=file_path,
            details={"context": context},
            stacktrace=traceback.format_exc(),
            context=merged_context,
        )

        # Add error to list and log it
        self._add_error(error)

        # Raise error if critical
        if severity == ErrorSeverity.CRITICAL:
            raise e

    def get_error_report(
        self, include_stacktrace: bool = False, include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive error report with enhanced information.

        Args:
            include_stacktrace: Whether to include stack traces in the report
            include_context: Whether to include context information in the report

        Returns:
            Dictionary containing error report
        """
        error_counts = {
            severity.value: len([e for e in self.errors if e.severity == severity])
            for severity in ErrorSeverity
        }

        error_details = []
        for error in self.errors:
            error_dict = error.to_dict(include_stacktrace=include_stacktrace)
            if not include_context:
                error_dict.pop("context", None)
            error_details.append(error_dict)

        # Group errors by type
        error_types = {}
        for error in self.errors:
            if error.error_type not in error_types:
                error_types[error.error_type] = 0
            error_types[error.error_type] += 1

        # Group errors by file path
        file_errors = {}
        for error in self.errors:
            if error.file_path:
                file_path = str(error.file_path)
                if file_path not in file_errors:
                    file_errors[file_path] = 0
                file_errors[file_path] += 1

        return {
            "summary": {
                "total_errors": len(self.errors),
                "error_counts": error_counts,
                "error_types": error_types,
                "file_errors": file_errors,
                "has_critical_errors": error_counts[ErrorSeverity.CRITICAL.value] > 0,
            },
            "errors": error_details,
        }

    def export_error_report(
        self, output_path: Path, include_stacktrace: bool = False
    ) -> None:
        """
        Export error report to a JSON file.

        Args:
            output_path: Path where to save the report
            include_stacktrace: Whether to include stack traces in the report
        """
        report = self.get_error_report(include_stacktrace)
        try:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Error report exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export error report: {str(e)}")

    def get_errors_by_severity(
        self, severity: Union[ErrorSeverity, str]
    ) -> List[ProcessingError]:
        """
        Get errors filtered by severity level.

        Args:
            severity: ErrorSeverity enum or string value

        Returns:
            List of ProcessingError objects with the specified severity
        """
        if isinstance(severity, str):
            try:
                severity = ErrorSeverity(severity)
            except ValueError:
                raise ValueError(f"Invalid severity: {severity}")

        return [error for error in self.errors if error.severity == severity]

    def get_errors_by_type(self, error_type: str) -> List[ProcessingError]:
        """
        Get errors filtered by error type.

        Args:
            error_type: Type of error to filter by

        Returns:
            List of ProcessingError objects with the specified type
        """
        return [error for error in self.errors if error.error_type == error_type]

    def get_errors_by_file(self, file_path: Union[str, Path]) -> List[ProcessingError]:
        """
        Get errors related to a specific file.

        Args:
            file_path: Path to file

        Returns:
            List of ProcessingError objects related to the specified file
        """
        file_path_str = str(file_path)
        return [
            error
            for error in self.errors
            if error.file_path and str(error.file_path) == file_path_str
        ]

    def has_critical_errors(self) -> bool:
        """
        Check if there are any critical errors.

        Returns:
            Boolean indicating presence of critical errors
        """
        return any(error.severity == ErrorSeverity.CRITICAL for error in self.errors)

    def export_detailed_error_report(
        self,
        output_path: Union[str, Path],
        include_stacktrace: bool = True,
        include_context: bool = True,
        min_severity: Union[ErrorSeverity, str] = ErrorSeverity.INFO,
        error_types: Optional[List[str]] = None,
        format: str = "json",
    ) -> None:
        """
        Export a detailed error report with filtering options.

        Args:
            output_path: Path where to save the report
            include_stacktrace: Whether to include stack traces
            include_context: Whether to include context information
            min_severity: Minimum severity level to include
            error_types: Optional list of error types to include
            format: Output format ('json' or 'csv')

        Raises:
            ValueError: If format is invalid
        """
        # Convert string severity to enum if needed
        if isinstance(min_severity, str):
            try:
                min_severity = ErrorSeverity(min_severity)
            except ValueError:
                raise ValueError(f"Invalid severity: {min_severity}")

        # Get severity values in order
        severity_values = [s.value for s in ErrorSeverity]
        min_severity_index = severity_values.index(min_severity.value)
        included_severities = severity_values[min_severity_index:]

        # Filter errors
        filtered_errors = [
            error
            for error in self.errors
            if error.severity.value in included_severities
            and (error_types is None or error.error_type in error_types)
        ]

        # Create report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_errors": len(self.errors),
                "filtered_errors": len(filtered_errors),
                "filter_criteria": {
                    "min_severity": min_severity.value,
                    "error_types": error_types,
                },
            },
            "errors": [
                error.to_dict(include_stacktrace=include_stacktrace)
                for error in filtered_errors
            ],
        }

        # Export in requested format
        output_path = Path(output_path)

        if format.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
        elif format.lower() == "csv":
            try:
                import pandas as pd

                # Flatten error dictionaries for CSV
                rows = []
                for error in filtered_errors:
                    error_dict = error.to_dict(include_stacktrace=include_stacktrace)

                    # Handle nested dictionaries
                    if "details" in error_dict and isinstance(
                        error_dict["details"], dict
                    ):
                        for k, v in error_dict["details"].items():
                            error_dict[f"details_{k}"] = str(v)
                        del error_dict["details"]

                    if (
                        "context" in error_dict
                        and isinstance(error_dict["context"], dict)
                        and include_context
                    ):
                        for k, v in error_dict["context"].items():
                            error_dict[f"context_{k}"] = str(v)
                        del error_dict["context"]
                    elif not include_context and "context" in error_dict:
                        del error_dict["context"]

                    rows.append(error_dict)

                # Create DataFrame and export to CSV
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
            except ImportError:
                raise ImportError(
                    "CSV export requires pandas. Install it with 'pip install pandas'"
                )
        else:
            raise ValueError(f"Invalid format: {format}. Supported formats: json, csv")

        self.logger.info(f"Error report exported to {output_path}")

    @staticmethod
    def create_config_from_dict(config_class, config_dict):
        """
        Create a configuration object from a dictionary.

        Args:
            config_class: Configuration class to instantiate
            config_dict: Dictionary of configuration parameters

        Returns:
            Configuration object
        """
        # Filter out None values and keys not in the config class fields
        valid_fields = {f.name for f in fields(config_class)}
        filtered_dict = {
            k: v for k, v in config_dict.items() if k in valid_fields and v is not None
        }

        return config_class(**filtered_dict)

    def update_config(
        self,
        discovery_config: Optional[Dict[str, Any]] = None,
        column_config: Optional[Dict[str, Any]] = None,
        time_series_config: Optional[Dict[str, Any]] = None,
        loading_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Update configuration settings after initialization.

        Args:
            discovery_config: Dictionary of discovery configuration parameters
            column_config: Dictionary of column naming configuration parameters
            time_series_config: Dictionary of time series configuration parameters
            loading_config: Dictionary of loading configuration parameters

        Returns:
            Self for method chaining
        """
        if discovery_config:
            # Create a new config object with updated values
            current_dict = {
                f.name: getattr(self.discovery_config, f.name)
                for f in fields(FileDiscoveryConfig)
            }
            current_dict.update(discovery_config)
            self.discovery_config = self.create_config_from_dict(
                FileDiscoveryConfig, current_dict
            )

        if column_config:
            current_dict = {
                f.name: getattr(self.column_config, f.name)
                for f in fields(ColumnNamingConfig)
            }
            current_dict.update(column_config)
            self.column_config = self.create_config_from_dict(
                ColumnNamingConfig, current_dict
            )

        if time_series_config:
            current_dict = {
                f.name: getattr(self.time_series_config, f.name)
                for f in fields(TimeSeriesConfig)
            }
            current_dict.update(time_series_config)
            self.time_series_config = self.create_config_from_dict(
                TimeSeriesConfig, current_dict
            )

            # Update time series validator if strategy changed
            if "validation_strategy" in time_series_config:
                self.time_series_config.time_series_validator = (
                    DefaultTimeSeriesValidator(
                        strategy=self.time_series_config.validation_strategy,
                        max_allowed_gap=self.time_series_config.max_allowed_gap,
                        allow_overlap=self.time_series_config.allow_overlap,
                        max_allowed_overlap=self.time_series_config.max_allowed_overlap,
                    )
                )

        if loading_config:
            current_dict = {
                f.name: getattr(self.loading_config, f.name)
                for f in fields(LoadingConfig)
            }
            current_dict.update(loading_config)
            self.loading_config = self.create_config_from_dict(
                LoadingConfig, current_dict
            )

        return self

    @classmethod
    def from_directory(
        cls,
        directory_path: Union[str, Path],
        glob_pattern: str = "*.csv",
        recursive: bool = False,
        metadata_extractor: Optional[MetadataExtractor] = None,
        **kwargs,
    ) -> "FileDataFrame":
        """
        Create a FileDataFrame from a directory of files.

        Args:
            directory_path: Path to directory containing files
            glob_pattern: Pattern to use for file discovery
            recursive: Whether to search recursively
            metadata_extractor: Optional custom metadata extractor
            **kwargs: Additional arguments to pass to FileDataFrame constructor

        Returns:
            FileDataFrame instance

        Raises:
            ValueError: If directory_path is invalid
        """
        discovery_config = FileDiscoveryConfig(
            glob_pattern=glob_pattern, recursive=recursive
        )

        return cls(
            base_path=directory_path,
            metadata_extractor=metadata_extractor,
            discovery_config=discovery_config,
            **kwargs,
        )

    @classmethod
    def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        metadata_extractor: Optional[MetadataExtractor] = None,
        **kwargs,
    ) -> "FileDataFrame":
        """
        Create a FileDataFrame from a list of file paths.

        Args:
            file_paths: List of paths to files
            metadata_extractor: Optional custom metadata extractor
            **kwargs: Additional arguments to pass to FileDataFrame constructor

        Returns:
            FileDataFrame instance

        Raises:
            ValueError: If file_paths is empty
        """
        if not file_paths:
            raise ValueError("file_paths cannot be empty")

        return cls(files=file_paths, metadata_extractor=metadata_extractor, **kwargs)

    @classmethod
    def from_streamlit(
        cls,
        uploaded_files: Union[UploadedFile, List[UploadedFile]],
        metadata_extractor: Optional[MetadataExtractor] = None,
        **kwargs,
    ) -> "FileDataFrame":
        """
        Create a FileDataFrame from Streamlit uploaded files.

        Args:
            uploaded_files: Streamlit uploaded file(s)
            metadata_extractor: Optional custom metadata extractor
            **kwargs: Additional arguments to pass to FileDataFrame constructor

        Returns:
            FileDataFrame instance

        Raises:
            ValueError: If uploaded_files is empty
        """
        if isinstance(uploaded_files, list) and not uploaded_files:
            raise ValueError("uploaded_files cannot be empty")

        return cls(
            streamlit_files=uploaded_files,
            metadata_extractor=metadata_extractor,
            **kwargs,
        )

    @classmethod
    def with_custom_validation(
        cls,
        file_source: Union[
            str, Path, List[Union[str, Path]], UploadedFile, List[UploadedFile]
        ],
        validation_strategy: Union[ValidationStrategy, str] = ValidationStrategy.STRICT,
        max_allowed_gap: Union[timedelta, str] = timedelta(minutes=5),
        allow_overlap: bool = False,
        **kwargs,
    ) -> "FileDataFrame":
        """
        Create a FileDataFrame with custom validation settings.

        Args:
            file_source: Directory path, list of files, or Streamlit uploaded files
            validation_strategy: Validation strategy to use
            max_allowed_gap: Maximum allowed gap between files
            allow_overlap: Whether to allow overlaps between files
            **kwargs: Additional arguments to pass to FileDataFrame constructor

        Returns:
            FileDataFrame instance

        Raises:
            ValueError: If file_source is invalid
        """
        # Convert string timedelta to timedelta object if needed
        if isinstance(max_allowed_gap, str):
            if max_allowed_gap.isdigit():
                max_allowed_gap = timedelta(minutes=int(max_allowed_gap))
            else:
                max_allowed_gap = pd.Timedelta(max_allowed_gap).to_pytimedelta()

        time_series_config = TimeSeriesConfig(
            validation_strategy=validation_strategy,
            max_allowed_gap=max_allowed_gap,
            allow_overlap=allow_overlap,
        )

        # Determine the type of file source
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if path.is_dir():
                return cls.from_directory(
                    directory_path=path, time_series_config=time_series_config, **kwargs
                )
            else:
                return cls.from_files(
                    file_paths=[path], time_series_config=time_series_config, **kwargs
                )
        elif isinstance(file_source, list):
            if all(isinstance(f, (str, Path)) for f in file_source):
                return cls.from_files(
                    file_paths=file_source,
                    time_series_config=time_series_config,
                    **kwargs,
                )
            else:
                return cls.from_streamlit(
                    uploaded_files=file_source,
                    time_series_config=time_series_config,
                    **kwargs,
                )
        else:
            return cls.from_streamlit(
                uploaded_files=file_source,
                time_series_config=time_series_config,
                **kwargs,
            )

    def _validate_configs(self) -> None:
        """
        Validate configuration objects for consistency and correctness.

        Raises:
            ValidationError: If any configuration is invalid
        """
        errors = []

        # Validate discovery config
        if hasattr(self, "discovery_config"):
            if not isinstance(self.discovery_config.glob_pattern, str):
                errors.append("discovery_config.glob_pattern must be a string")
            if not isinstance(self.discovery_config.recursive, bool):
                errors.append("discovery_config.recursive must be a boolean")

        # Validate loading config
        if hasattr(self, "loading_config"):
            if not isinstance(self.loading_config.delimiter, str):
                errors.append("loading_config.delimiter must be a string")
            if not isinstance(self.loading_config.time_format, str):
                errors.append("loading_config.time_format must be a string")

        # Validate time series config
        if hasattr(self, "time_series_config"):
            if not isinstance(self.time_series_config.max_allowed_gap, timedelta):
                errors.append("time_series_config.max_allowed_gap must be a timedelta")
            if not isinstance(self.time_series_config.allow_overlap, bool):
                errors.append("time_series_config.allow_overlap must be a boolean")

        # Raise validation error if any issues found
        if errors:
            raise ValidationError(
                message="Invalid configuration settings",
                validation_type="configuration",
                details={"errors": errors},
                context={
                    "discovery_config": (
                        self.discovery_config.__dict__
                        if hasattr(self, "discovery_config")
                        else None
                    ),
                    "loading_config": (
                        self.loading_config.__dict__
                        if hasattr(self, "loading_config")
                        else None
                    ),
                    "time_series_config": (
                        {
                            k: str(v) if isinstance(v, timedelta) else v
                            for k, v in self.time_series_config.__dict__.items()
                        }
                        if hasattr(self, "time_series_config")
                        else None
                    ),
                },
            )

    def _validate_direct_files(self) -> List[Path]:
        """
        Validate directly provided files using the configured file filter.

        Returns:
            List of valid Path objects

        Raises:
            FileDiscoveryError: If no valid files are found
        """
        if not self.files:
            raise FileDiscoveryError("No files provided for direct file processing")

        valid_files = []
        invalid_files = []

        for file_path in self.files:
            try:
                if self._is_valid_file(file_path):
                    valid_files.append(file_path)
                else:
                    invalid_files.append((file_path, "Failed validation checks"))
            except Exception as e:
                self.logger.error(
                    f"File {file_path} is not included in the "
                    f"final data due to {str(e)}"
                )
                invalid_files.append((file_path, str(e)))

        if not valid_files:
            error_msg = "No valid files found"
            if invalid_files:
                error_msg += "\nInvalid files:"
                for file_path, reason in invalid_files:
                    error_msg += f"\n- {file_path}: {reason}"
            raise FileDiscoveryError(error_msg)

        # Store file discovery statistics
        self.discovery_stats = {
            "total_files_found": len(valid_files) + len(invalid_files),
            "valid_files": len(valid_files),
            "invalid_files": len(invalid_files),
            "invalid_files_details": invalid_files,
        }

        return sorted(valid_files)

    def _validate_streamlit_files(
        self, is_critical: bool = True
    ) -> List[Tuple[str, BytesIO]]:
        """
        Validate Streamlit uploaded files using the configured metadata extractor.

        Args:
            is_critical: Whether the errors become critical if encountered

        Returns:
            List of tuples containing (filename, file content as BytesIO)

        Raises:
            FileDiscoveryError: If files are invalid
        """
        if not self.streamlit_files:
            raise FileDiscoveryError("No Streamlit files provided")

        # Convert single file to list if necessary
        files_list = (
            [self.streamlit_files]
            if isinstance(self.streamlit_files, UploadedFile)
            else self.streamlit_files
        )

        valid_files = []
        invalid_files = []

        for uploaded_file in files_list:
            try:
                # Check if file content is not empty
                if not uploaded_file.getvalue():
                    invalid_files.append((uploaded_file.name, "File content is empty"))
                    continue

                # Check if filename is valid using metadata extractor
                if self.metadata_extractor.is_valid_filename(uploaded_file.name):
                    content = BytesIO(uploaded_file.getvalue())
                    valid_files.append((uploaded_file.name, content))
                else:
                    invalid_files.append(
                        (uploaded_file.name, "Failed filename validation")
                    )
            except Exception as e:
                invalid_files.append((uploaded_file.name, str(e)))

        # Store file discovery statistics
        self.discovery_stats = {
            "total_files_found": len(valid_files) + len(invalid_files),
            "valid_files": len(valid_files),
            "invalid_files": len(invalid_files),
            "invalid_files_details": invalid_files,
        }

        if not valid_files:
            error_msg = "No valid files found"
            if invalid_files:
                error_msg += "\nInvalid files:"
                for file_name, reason in invalid_files:
                    error_msg += f"\n- {file_name}: {reason}"
            raise FileDiscoveryError(error_msg)
        elif is_critical and invalid_files:
            self._handle_error(invalid_files[0][1], "validate_streamlit_files")
            raise FileDiscoveryError(invalid_files[0][1])

        return valid_files

    def _is_valid_file(self, file_path: Path) -> bool:
        """
        Check if a file is valid using the configured file filter and validator.

        Args:
            file_path: Path object representing the file

        Returns:
            Boolean indicating if the file is valid
        """
        try:
            # First check with the file filter
            if not self.discovery_config.file_filter.is_valid(file_path):
                return False

            # Then use custom validator if provided
            if self.file_validator:
                validation_context = {
                    "operation": "_is_valid_file",
                    "timestamp": datetime.now().isoformat(),
                }
                result = self.file_validator.validate(file_path, validation_context)
                return result.is_valid

            return True
        except Exception as e:
            self.logger.debug(f"Error validating file {file_path}: {str(e)}")
            return False

    def validate_time_series(self) -> ValidationResult:
        """
        Validate that the files form a continuous time series.

        Returns:
            ValidationResult object

        Raises:
            TimeValidationError: If validation fails and fail_on_validation_error is True
        """

        result = self.time_series_config.time_series_validator.is_valid_sequence(
            self.metadata
        )

        if not result.is_valid and self.time_series_config.fail_on_validation_error:
            raise TimeValidationError(
                invalid_time=None,
                message=result.error_message or "Time series validation failed",
            )

        return result

    def get_time_series_issues(self) -> List[TimeValidationIssue]:
        """
        Get a list of time series continuity issues.

        Returns:
            List of TimeValidationIssue objects

        Raises:
            ValueError: If metadata is not available
        """
        if not self.metadata:
            raise ValueError("No metadata available. Run file processing first.")

        return self.time_series_validator.validate_files(self.metadata)

    def generate_time_series_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on time series continuity.

        Returns:
            Dictionary containing time series report

        Raises:
            ValueError: If metadata is not available
        """
        if not self.metadata:
            raise ValueError("No metadata available. Run file processing first.")

        issues = self.time_series_validator.validate_files(self.metadata)

        # Sort metadata by start_time
        sorted_metadata = sorted(self.metadata, key=lambda x: x.start_time)

        # Calculate total time span
        total_span = sorted_metadata[-1].end_time - sorted_metadata[0].start_time

        # Calculate coverage
        covered_time = timedelta(0)
        for meta in sorted_metadata:
            file_duration = meta.end_time - meta.start_time
            covered_time += file_duration

        # Calculate gaps
        gaps = [issue for issue in issues if issue.issue_type == "gap"]
        total_gap_time = sum((gap.duration for gap in gaps), timedelta(0))

        # Calculate overlaps
        overlaps = [issue for issue in issues if issue.issue_type == "overlap"]
        total_overlap_time = sum(
            (overlap.duration for overlap in overlaps), timedelta(0)
        )

        # Adjust covered time for overlaps
        adjusted_coverage = covered_time - total_overlap_time

        # Calculate coverage percentage
        coverage_percentage = (
            (adjusted_coverage / total_span) * 100
            if total_span.total_seconds() > 0
            else 0
        )

        return {
            "total_files": len(self.metadata),
            "start_time": sorted_metadata[0].start_time,
            "end_time": sorted_metadata[-1].end_time,
            "total_time_span": total_span,
            "covered_time": covered_time,
            "adjusted_coverage": adjusted_coverage,
            "coverage_percentage": coverage_percentage,
            "total_gaps": len(gaps),
            "total_gap_time": total_gap_time,
            "total_overlaps": len(overlaps),
            "total_overlap_time": total_overlap_time,
            "issues": [
                {
                    "issue_type": issue.issue_type,
                    "start_time": issue.start_time,
                    "end_time": issue.end_time,
                    "duration": issue.duration,
                    "file1": str(issue.file1),
                    "file2": str(issue.file2),
                }
                for issue in issues
            ],
            "files": [
                {
                    "filepath": str(meta.filepath),
                    "start_time": meta.start_time,
                    "end_time": meta.end_time,
                    "duration": meta.end_time - meta.start_time,
                }
                for meta in sorted_metadata
            ],
        }

    def discover_files(self) -> List[Path]:
        """
        Discover and validate files based on the configured source and discovery settings.

        Returns:
            List of Path objects for valid files

        Raises:
            FileDiscoveryError: If file discovery fails
        """
        try:
            if self.base_path is not None:
                # Simple directory check
                if not self.base_path.exists():
                    raise FileDiscoveryError(
                        f"Base path does not exist: {self.base_path}"
                    )

                if not self.base_path.is_dir():
                    raise FileDiscoveryError(
                        f"Base path is not a directory: {self.base_path}"
                    )

                try:
                    next(os.scandir(self.base_path), None)
                except Exception:
                    raise FileDiscoveryError(
                        f"No read permission for directory: {self.base_path}"
                    )

                # Use glob pattern for file discovery
                if self.discovery_config.recursive:
                    all_files = list(
                        self.base_path.glob(f"**/{self.discovery_config.glob_pattern}")
                    )
                else:
                    all_files = list(
                        self.base_path.glob(self.discovery_config.glob_pattern)
                    )

                if not all_files:
                    raise FileDiscoveryError(
                        f"No files matching pattern '{self.discovery_config.glob_pattern}' found in {self.base_path}"
                    )

                # Filter files
                valid_files = []
                invalid_files = []

                for file_path in all_files:
                    try:
                        if self._is_valid_file(file_path):
                            valid_files.append(file_path)
                        else:
                            invalid_files.append(
                                (file_path, "Failed validation checks")
                            )
                    except Exception as e:
                        invalid_files.append((file_path, str(e)))

                if not valid_files:
                    error_msg = f"No valid files found matching pattern '{self.discovery_config.glob_pattern}'"
                    if invalid_files:
                        error_msg += "\nInvalid files:"
                        for file_path, reason in invalid_files:
                            error_msg += f"\n- {file_path}: {reason}"
                    raise FileDiscoveryError(error_msg)

                # Store file discovery statistics
                self.discovery_stats = {
                    "total_files_found": len(valid_files) + len(invalid_files),
                    "valid_files": len(valid_files),
                    "invalid_files": len(invalid_files),
                    "invalid_files_details": invalid_files,
                    "glob_pattern": self.discovery_config.glob_pattern,
                    "recursive": self.discovery_config.recursive,
                }

                # Sort files by name for consistent processing
                valid_files.sort()
                return valid_files

            elif self.files is not None:
                # Use direct file validation
                return self._validate_direct_files()

            elif self.streamlit_files is not None:
                # Use streamlit file validation
                return self._validate_streamlit_files()

        except Exception as e:
            if isinstance(e, FileDiscoveryError):
                raise
            raise FileDiscoveryError(f"Error during file discovery: {str(e)}")

    def get_discovery_stats(self) -> dict:
        """
        Get statistics about the file discovery process.

        Returns:
            Dictionary containing discovery statistics

        Raises:
            FileDiscoveryError: If discovery hasn't been run
        """
        if not hasattr(self, "discovery_stats"):
            raise FileDiscoveryError(
                "Discovery statistics not available. Run discover_files first."
            )
        return self.discovery_stats

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the processing status.

        Returns:
            Dictionary containing processing summary
        """
        summary = {
            "status": "not_started",
            "errors": {
                "total": len(self.errors),
                "by_severity": {severity.value: 0 for severity in ErrorSeverity},
            },
            "files": {
                "discovered": getattr(self, "discovery_stats", {}).get(
                    "total_files_found", 0
                ),
                "valid": getattr(self, "discovery_stats", {}).get("valid_files", 0),
                "invalid": getattr(self, "discovery_stats", {}).get("invalid_files", 0),
                "processed": len(self.metadata) if hasattr(self, "metadata") else 0,
            },
            "data": {
                "loaded": self.dataframe is not None,
                "rows": len(self.dataframe) if self.dataframe is not None else 0,
                "columns": (
                    len(self.dataframe.columns) if self.dataframe is not None else 0
                ),
            },
        }

        # Count errors by severity
        for error in self.errors:
            summary["errors"]["by_severity"][error.severity.value] += 1

        # Determine overall status
        if self.dataframe is not None:
            summary["status"] = "completed"
        elif self.metadata:
            summary["status"] = "metadata_extracted"
        elif hasattr(self, "discovery_stats"):
            summary["status"] = "files_discovered"

        # Check for critical errors
        if summary["errors"]["by_severity"][ErrorSeverity.CRITICAL.value] > 0:
            summary["status"] = "failed"

        return summary

    def initialize_processing(self) -> None:
        """
        Initialize the complete file processing pipeline.

        This method runs the complete pipeline:
        1. Validate configurations
        2. Discover files
        3. Process files for metadata
        4. Validate time series continuity
        5. Load and concatenate data

        Raises:
            Various exceptions from individual processing steps
        """
        processing_context = {
            "start_time": datetime.now().isoformat(),
            "operation": "initialize_processing",
        }

        try:
            # Validate configurations
            self._validate_configs()
            processing_context["step"] = "config_validation"

            # Discover files
            processing_context["step"] = "file_discovery"
            files = self.discover_files()
            processing_context["files_found"] = len(files)

            # Process files for metadata based on type
            processing_context["step"] = "metadata_extraction"
            if self.streamlit_files is not None:
                self.metadata = self.process_streamlit_files(files)
            else:
                self.metadata = self.process_files(files)
            processing_context["metadata_count"] = len(self.metadata)

            # Time series validation is done in process_files/process_streamlit_files
            processing_context["step"] = "data_loading"

            # Load and concatenate data
            self.load_and_concatenate()

            # Update processing context with success information
            processing_context["end_time"] = datetime.now().isoformat()
            processing_context["status"] = "success"
            processing_context["rows_loaded"] = (
                len(self.dataframe) if self.dataframe is not None else 0
            )

        except Exception as e:
            # Update processing context with failure information
            processing_context["end_time"] = datetime.now().isoformat()
            processing_context["status"] = "failed"
            processing_context["error_type"] = type(e).__name__
            processing_context["error_message"] = str(e)

            self._handle_error(
                e, "initialize_processing", additional_context=processing_context
            )
            raise

    def extract_metadata_from_stream(
        self, filename: str, content: BytesIO
    ) -> FileMetadata:
        """
        Extract metadata from a Streamlit uploaded file.

        Args:
            filename: Name of the uploaded file
            content: BytesIO object containing file content

        Returns:
            FileMetadata object containing validated metadata

        Raises:
            FileParsingError: If metadata extraction or validation fails
        """
        try:
            # Validate content exists
            if content is None or content.getvalue() == b"":
                raise FileParsingError(
                    filepath=filename,
                    reason="File content is empty",
                )

            # Create a temporary Path object for the filename to use with the extractor
            return self.metadata_extractor.extract_metadata(Path(filename))

        except Exception as e:
            # Wrap any unexpected errors
            if isinstance(e, FileParsingError):
                raise
            raise FileParsingError(
                filepath=filename, reason=f"Unexpected error: {str(e)}"
            )

    def process_streamlit_files(
        self, streamlit_files: List[Tuple[str, BytesIO]]
    ) -> List[FileMetadata]:
        """
        Process a list of Streamlit uploaded files and extract metadata from each.

        Args:
            streamlit_files: List of tuples containing (filename, BytesIO content)

        Returns:
            List of FileMetadata objects

        Raises:
            FileParsingError: If any file fails processing
            TimeValidationError: If time series validation fails
        """
        metadata_list = []
        errors = []

        for filename, content in streamlit_files:
            try:
                metadata = self.extract_metadata_from_stream(filename, content)
                metadata_list.append(metadata)
            except Exception as e:
                errors.append(f"Error processing {filename}: {str(e)}")
                self._handle_error(e, "process_streamlit_files", Path(filename))

        if errors:
            raise FileParsingError(
                filepath=(str(streamlit_files[0][0]) if streamlit_files else "unknown"),
                reason="Multiple errors processing files:\n" + "\n".join(errors),
            )

        # Sort metadata by start_time
        try:
            metadata_list.sort(key=lambda x: x.start_time)
            self.metadata = metadata_list
        except TypeError:  # in case no timestamps are found in filenames
            self.logger.info(
                "No consistent start time timestamps is found in filenames"
            )
            self.metadata = metadata_list

        # Validate time series continuity
        self.validate_time_series()

        return metadata_list

    def extract_metadata(self, filepath: Path) -> FileMetadata:
        """
        Extract metadata from a file and validate it.

        Args:
            filepath: Path object representing the file

        Returns:
            FileMetadata object containing validated metadata

        Raises:
            FileParsingError: If metadata extraction or validation fails
        """
        try:
            # Validate file exists and is a file
            if not filepath.is_file():
                raise FileParsingError(
                    filepath=str(filepath),
                    reason="Path is not a file or doesn't exist",
                )

            # Use metadata extractor to extract metadata
            return self.metadata_extractor.extract_metadata(filepath)

        except Exception as e:
            # Wrap any unexpected errors
            if isinstance(e, FileParsingError):
                raise
            raise FileParsingError(
                filepath=str(filepath), reason=f"Unexpected error: {str(e)}"
            )

    def process_files(self, filepaths: List[Path]) -> List[FileMetadata]:
        """
        Process a list of files and extract metadata from each.

        Args:
            filepaths: List of Path objects representing files to process

        Returns:
            List of FileMetadata objects

        Raises:
            FileParsingError: If any file fails processing
            TimeValidationError: If time series validation fails
        """
        metadata_list = []
        errors = []

        for filepath in filepaths:
            try:
                metadata = self.extract_metadata(filepath)
                metadata_list.append(metadata)
            except Exception as e:
                errors.append(f"Error processing {filepath}: {str(e)}")
                self._handle_error(e, "process_files", filepath)

        if errors:
            raise FileParsingError(
                filepath=(str(filepaths[0]) if filepaths else "unknown"),
                reason="Multiple errors processing files:\n" + "\n".join(errors),
            )

        # Sort metadata by start_time
        try:
            metadata_list.sort(key=lambda x: x.start_time)
            self.metadata = metadata_list
        except TypeError:  # in case no timestamps are found in filenames
            self.logger.info(
                "No consistent start time timestamps is found in filenames"
            )
            self.metadata = metadata_list

        # Validate time series continuity
        if self.metadata:
            self.validate_time_series()
        else:
            self.logger.info("No metadata have been found")

        return metadata_list

    def _validate_dataframe_structure(self, df: pd.DataFrame, filepath: Path) -> None:
        """
        Validate the structure of a loaded DataFrame.

        Args:
            df: DataFrame to validate
            filepath: Source file path for error messages

        Raises:
            DataLoadingError: If validation fails
        """
        # Check if DataFrame is empty
        if df.empty:
            raise DataLoadingError(
                filepath=filepath, error_details=f"File {filepath} contains no data"
            )

        # If this is the first DataFrame, store its structure for comparison
        if self.dataframe is None:
            self._columns = set(df.columns)
            self._dtypes = df.dtypes
            return

        # Validate columns match
        current_columns = set(df.columns)
        if current_columns != self._columns:
            extra_cols = current_columns - self._columns
            missing_cols = self._columns - current_columns
            error_msg = []
            if extra_cols:
                error_msg.append(f"Extra columns found: {extra_cols}")
            if missing_cols:
                error_msg.append(f"Missing columns: {missing_cols}")
            raise ValueError(f"Column mismatch in {filepath}: {' | '.join(error_msg)}")

        # Validate data types match
        for col in df.columns:
            if not np.issubdtype(df[col].dtype, np.dtype(self._dtypes[col])):
                raise DataLoadingError(
                    filepath=filepath,
                    error_details=f"Data type mismatch in {filepath} for column {col}: "
                    + f"expected {self._dtypes[col]}, got {df[col].dtype}",
                )

    def _load_single_file(self, metadata: FileMetadata) -> pd.DataFrame:
        """
        Load and validate a single CSV file.

        Args:
            metadata: FileMetadata object containing file information

        Returns:
            Loaded and validated DataFrame

        Raises:
            DataLoadingError: If file cannot be loaded or validates
        """
        try:
            # Check if we're using Streamlit files
            if self.streamlit_files is not None:
                # Handle single file case
                if not isinstance(self.streamlit_files, (list, tuple)):
                    # For single file, just use it directly regardless of name
                    # This is because in the test, the metadata.filepath might be set to the filename
                    # but we only have one file anyway
                    df = pd.read_csv(
                        BytesIO(self.streamlit_files.getvalue()),
                        delimiter=self.loading_config.delimiter,
                        encoding=self.loading_config.encoding,
                    )
                # Handle multiple files case
                else:
                    # For debugging, let's print the names
                    file_names = [f.name for f in self.streamlit_files]

                    # Try to find a matching file
                    for uploaded_file in self.streamlit_files:
                        # Use exact match or check if the filename is in the path
                        if (
                            uploaded_file.name == metadata.filepath
                            or uploaded_file.name == Path(metadata.filepath).name
                        ):
                            df = pd.read_csv(
                                BytesIO(uploaded_file.getvalue()),
                                delimiter=self.loading_config.delimiter,
                                encoding=self.loading_config.encoding,
                            )
                            break
                    else:
                        raise DataLoadingError(
                            filepath=metadata.filepath,
                            error_details=f"Could not find Streamlit file content for {metadata.filepath}. Available files: {file_names}",
                        )
            # Regular file path case
            elif isinstance(metadata.filepath, (str, Path)):
                df = pd.read_csv(
                    metadata.filepath,
                    delimiter=self.loading_config.delimiter,
                    encoding=self.loading_config.encoding,
                )
            else:
                raise DataLoadingError(
                    filepath=str(metadata.filepath),
                    error_details=f"Unsupported file type: {type(metadata.filepath)}",
                )

            # Basic validation
            self._validate_dataframe_structure(df, metadata.filepath)

            # Apply custom data transformation
            df = self.data_transformer.transform(df, metadata)

            # Add metadata columns
            df["source_file"] = str(metadata.filepath)
            df["file_start_time"] = metadata.start_time
            df["file_end_time"] = metadata.end_time

            return df

        except pd.errors.EmptyDataError:
            raise DataLoadingError(
                filepath=str(metadata.filepath),
                error_details=f"File {metadata.filepath} is empty",
            )
        except pd.errors.ParserError as e:
            raise DataLoadingError(
                filepath=str(metadata.filepath),
                error_details=f"Error parsing CSV file {metadata.filepath}: {str(e)}",
            )
        except Exception as e:
            raise DataLoadingError(
                filepath=str(metadata.filepath),
                error_details=f"Unexpected error loading {metadata.filepath}: {str(e)}",
            )

    def _apply_column_naming_configuration(self) -> pd.DataFrame:
        """
        Apply the configured column naming rules to the dataframe.
        This includes whitespace stripping, custom renaming, and prefix removal.

        Returns:
            DataFrame with renamed columns

        Raises:
            ValueError: If DataFrame is not loaded
        """
        if self.dataframe is None:
            raise ValueError("DataFrame not loaded. Run load_and_concatenate first.")

        df = self.dataframe

        # Apply column renaming in this order:

        # 1. Strip whitespace if configured
        if self.column_config.strip_column_whitespace:
            df.columns = [col.strip() for col in df.columns]

        # 2. Apply custom rename map if provided
        if self.column_config.column_rename_map:
            # Only rename columns that exist in the DataFrame
            rename_dict = {
                k: v
                for k, v in self.column_config.column_rename_map.items()
                if k in df.columns
            }
            if rename_dict:
                df = df.rename(columns=rename_dict)

        # 3. Clean column names if configured (remove prefixes)
        if self.column_config.clean_column_names:
            # Create a function to clean column names based on a common pattern
            def clean_name(col):
                # Split on the last occurrence of " - " and take the last part
                if " - " in col:
                    return col.rsplit(" - ", maxsplit=1)[-1].strip()
                return col

            df.columns = [clean_name(col) for col in df.columns]

        self.dataframe = df
        return df

    def rename_columns(
        self, rename_map: Dict[str, str], inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Rename columns using a mapping dictionary.

        Args:
            rename_map: Dictionary mapping original column names to new names
            inplace: Whether to modify the dataframe in place or return a copy

        Returns:
            Modified DataFrame if inplace=False, otherwise None

        Raises:
            ValueError: If DataFrame is not loaded
        """
        if self.dataframe is None:
            raise ValueError("DataFrame not loaded. Run load_and_concatenate first.")

        if inplace:
            self.dataframe.rename(columns=rename_map, inplace=True)
            return None
        else:
            return self.dataframe.rename(columns=rename_map)

    def get_original_column_names(self) -> List[str]:
        """
        Get the original column names from the first file.

        Returns:
            List of original column names

        Raises:
            DataLoadingError: If no files have been processed
        """
        if not self.metadata:
            raise DataLoadingError(
                filepath=None,
                error_details="No metadata available. Run file processing first.",
            )

        try:
            # Load the first file to get column names
            first_metadata = self.metadata[0]

            # Handle both Path and Streamlit files
            if isinstance(first_metadata.filepath, (str, Path)):
                if (
                    isinstance(first_metadata.filepath, str)
                    and self.streamlit_files is not None
                ):
                    # Find the matching Streamlit file
                    if isinstance(self.streamlit_files, UploadedFile):
                        if self.streamlit_files.name == first_metadata.filepath:
                            df = pd.read_csv(
                                BytesIO(self.streamlit_files.getvalue()),
                                delimiter=";",
                                nrows=0,
                            )
                        else:
                            raise DataLoadingError(
                                filepath=first_metadata.filepath,
                                error_details=f"Could not find Streamlit file content for {first_metadata.filepath}",
                            )
                    else:
                        for uploaded_file in self.streamlit_files:
                            if uploaded_file.name == first_metadata.filepath:
                                df = pd.read_csv(
                                    BytesIO(uploaded_file.getvalue()),
                                    delimiter=";",
                                    nrows=0,
                                )
                                break
                        else:
                            raise DataLoadingError(
                                filepath=first_metadata.filepath,
                                error_details=f"Could not find Streamlit file content for {first_metadata.filepath}",
                            )
                else:
                    # Regular file path
                    df = pd.read_csv(first_metadata.filepath, delimiter=";", nrows=0)
            else:
                raise DataLoadingError(
                    filepath=str(first_metadata.filepath),
                    error_details=f"Unsupported file type: {type(first_metadata.filepath)}",
                )

            return list(df.columns)

        except Exception as e:
            if isinstance(e, DataLoadingError):
                raise
            raise DataLoadingError(
                filepath=str(self.metadata[0].filepath),
                error_details=f"Error getting original column names: {str(e)}",
            )

    def load_and_concatenate(self) -> None:
        """
        Load all files, validate them, and concatenate into a single DataFrame.
        The function sort timestamps if present.

        Raises:
            DataLoadingError: If any file fails to load or validate
        """
        if not self.metadata:
            raise DataLoadingError(
                filepath=None,
                error_details="No metadata available. Run file processing first.",
            )

        dfs = []
        errors = []

        # Load each file
        for metadata in self.metadata:
            try:
                df = self._load_single_file(metadata)

                # Apply basic whitespace stripping to column names if configured
                if self.column_config.strip_column_whitespace:
                    df.rename(columns=lambda x: x.strip(), inplace=True)

                dfs.append(df)
            except DataLoadingError as e:
                errors.append(str(e))

        if errors:
            raise DataLoadingError(
                filepath=metadata.filepath,
                error_details="Errors loading files:\n" + "\n".join(errors),
            )

        try:
            # Concatenate all DataFrames
            self.dataframe = pd.concat(dfs, axis=0, ignore_index=True)

            # Sort by timestamp if present
            timestamp_cols = [
                col
                for col in self.dataframe.columns
                if "time" in col.lower()
                and self.dataframe[col].dtype in ["datetime64[ns]", "object"]
            ]

            if timestamp_cols:
                # Try to convert to datetime if not already
                for col in timestamp_cols:
                    if self.dataframe[col].dtype == "object":
                        try:
                            self.dataframe[col] = pd.to_datetime(
                                self.dataframe[col],
                                format=self.loading_config.time_format,
                            )
                        except Exception:
                            continue

                # Sort by the first valid timestamp column
                self.dataframe.sort_values(timestamp_cols[0], inplace=True)
                self.dataframe.reset_index(drop=True, inplace=True)

            # Apply column renaming based on configuration
            if (
                self.column_config.clean_column_names
                or self.column_config.column_rename_map
            ):
                self._apply_column_naming_configuration()

            # Apply post-processing hooks
            processing_context = {
                "metadata": self.metadata,
                "files": [str(meta.filepath) for meta in self.metadata],
                "config": {
                    "column_config": self.column_config.__dict__,
                    "loading_config": self.loading_config.__dict__,
                    "time_series_config": {
                        k: str(v) if isinstance(v, timedelta) else v
                        for k, v in self.time_series_config.__dict__.items()
                    },
                },
            }

            for hook in self.post_processing_hooks:
                try:
                    self.dataframe = hook.process(self.dataframe, processing_context)
                except Exception as e:
                    self._handle_error(
                        e,
                        "post_processing_hook",
                        additional_context={"hook_type": type(hook).__name__},
                    )

            # Add concatenation metadata
            self.concat_metadata = {
                "total_rows": len(self.dataframe),
                "total_files": len(dfs),
                "memory_usage": self.dataframe.memory_usage(deep=True).sum(),
            }
            try:
                self.concat_metadata["start_time"] = min(
                    meta.start_time for meta in self.metadata
                )
                self.concat_metadata["end_time"] = min(
                    meta.end_time for meta in self.metadata
                )
            except Exception:
                self.concat_metadata["start_time"] = None
                self.concat_metadata["end_time"] = None

        except Exception as e:
            raise DataLoadingError(
                filepath=metadata.filepath,
                error_details=f"Error concatenating DataFrames: {str(e)}",
            )

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the concatenated DataFrame.

        Returns:
            The concatenated DataFrame

        Raises:
            DataLoadingError: If DataFrame hasn't been loaded
        """
        if self.dataframe is None:
            if not self.metadata:
                raise DataLoadingError(
                    filepath=None,
                    error_details="DataFrame not loaded. Run process_files and load_and_concatenate first.",
                )
            else:
                raise DataLoadingError(
                    filepath=self.metadata[0].filepath,
                    error_details="DataFrame not loaded. Run load_and_concatenate first.",
                )
        return self.dataframe

    def get_concat_metadata(self) -> dict:
        """
        Get metadata about the concatenated DataFrame.

        Returns:
            Dictionary containing concatenation metadata

        Raises:
            DataLoadingError: If DataFrame hasn't been loaded
        """
        if not hasattr(self, "concat_metadata"):
            if not self.metadata:
                raise DataLoadingError(
                    filepath=None,
                    error_details="Concatenation metadata not available. Run process_files and load_and_concatenate first.",
                )
            else:
                raise DataLoadingError(
                    filepath=self.metadata[0].filepath,
                    error_details="Concatenation metadata not available. Run load_and_concatenate first.",
                )
        return self.concat_metadata

    def analyze_time_series_continuity(
        self,
        time_column: str = None,
        expected_frequency: str = None,
        min_gap_size: str = "1min",
    ) -> Dict:
        """
        Analyze continuity of time series data in the DataFrame.

        Args:
            time_column: Name of the datetime column to analyze. If None, tries to auto-detect.
            expected_frequency: Expected frequency of data ('1s', '1min', '1H', etc.).
                              If None, tries to infer from data.
            min_gap_size: Minimum gap size to report (default '1min' = 1 minute)

        Returns:
            Dictionary containing gap analysis results

        Raises:
            DataLoadingError: If DataFrame hasn't been loaded
            TimeValidationError: If time column cannot be identified or frequency cannot be inferred
        """
        if self.dataframe is None:
            raise DataLoadingError(
                filepath=None,
                error_details="DataFrame not loaded. Run load_and_concatenate first.",
            )

        # Find datetime column if not specified
        if time_column is None:
            datetime_cols = [
                col
                for col in self.dataframe.columns
                if pd.api.types.is_datetime64_any_dtype(self.dataframe[col])
            ]
            if not datetime_cols:
                raise TimeValidationError(
                    invalid_time=None, message="No datetime column found in DataFrame"
                )
            time_column = datetime_cols[0]

        # Ensure column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.dataframe[time_column]):
            try:
                self.dataframe[time_column] = pd.to_datetime(
                    self.dataframe[time_column], format="%d/%m/%Y %H:%M"
                )
            except Exception as e:
                raise TimeValidationError(
                    invalid_time=time_column,
                    message=f"Could not convert {time_column} to datetime: {str(e)}",
                )

        # Sort by time column
        df = self.dataframe.sort_values(time_column).copy()

        # Infer frequency if not provided
        if expected_frequency is None:
            try:
                expected_frequency = pd.infer_freq(df[time_column])
                if expected_frequency is None:
                    # Calculate median time difference
                    median_diff = (
                        np.median(np.diff(df[time_column].astype(np.int64))) / 1e9
                    )
                    expected_frequency = f"{int(median_diff)}s"
            except Exception as e:
                raise TimeValidationError(f"Could not infer time frequency: {str(e)}")

        # Convert min_gap_size to timedelta
        min_gap = pd.Timedelta(min_gap_size)

        # Find gaps
        time_diffs = df[time_column].diff()
        expected_diff = pd.Timedelta(expected_frequency)
        gaps = []

        for idx, diff in enumerate(time_diffs[1:], 1):
            if diff > expected_diff + min_gap:
                gap = TimeSeriesGap(
                    start_time=df[time_column].iloc[idx - 1],
                    end_time=df[time_column].iloc[idx],
                    duration=diff,
                    expected_points=int(diff / expected_diff) - 1,
                )
                gaps.append(gap)

        # Calculate statistics
        total_duration = df[time_column].max() - df[time_column].min()
        total_gap_duration = sum((gap.duration for gap in gaps), pd.Timedelta(0))

        # Prepare report
        report = {
            "time_column": time_column,
            "min_gap_size": min_gap_size,
            "inferred_frequency": expected_frequency,
            "total_points": len(df),
            "start_time": df[time_column].min(),
            "end_time": df[time_column].max(),
            "total_duration": total_duration,
            "total_gaps": len(gaps),
            "total_gap_duration": total_gap_duration,
            "coverage_percentage": (
                (total_duration - total_gap_duration) / total_duration * 100
            ),
            "gaps": [
                {
                    "start_time": gap.start_time,
                    "end_time": gap.end_time,
                    "duration": gap.duration,
                    "expected_points": gap.expected_points,
                }
                for gap in gaps
            ],
        }

        self.time_series_analysis = report
        return report

    @staticmethod
    def upsample_df(df, target_freq, method="mean"):
        """
        Upsample dataframe to lower frequency (e.g., 15min to 1H)

        Parameters:
        df: DataFrame with datetime index
        target_freq: string representing target frequency (e.g., '1H' for 1 hour)
        method: aggregation method ('mean', 'sum', 'last')

        Returns:
        Upsampled DataFrame with aggregated values
        """
        if method == "mean":
            return df.resample(target_freq).mean()
        elif method == "sum":
            return df.resample(target_freq).sum()
        elif method == "last":
            return df.resample(target_freq).last()
        elif method == "first":
            return df.resample(target_freq).first()
        else:
            raise ValueError("Method must be one of: 'mean', 'sum', 'last'")

    @staticmethod
    def resample_with_dates(df, date_points, method="mean", skipna=False):
        """
        Resample dataframe using custom datetime breakpoints, handling both numerical and non-numerical columns

        Parameters:
        df: DataFrame with datetime index
        date_points: DatetimeIndex or list of datetime values defining the bins
        method: aggregation method ('mean', 'sum', 'last', 'first') for numerical columns
        skipna: whether to skip NaN values while calculating mean or sum

        Returns:
        Resampled DataFrame with custom periods
        """
        # Convert to list if it's DatetimeIndex
        date_points = (
            date_points.tolist()
            if isinstance(date_points, pd.DatetimeIndex)
            else date_points
        )
        date_points = sorted(date_points)

        # Separate numerical and non-numerical columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        non_numeric_cols = df.select_dtypes(exclude=np.number).columns

        result_dfs = []

        # Handle numeric columns
        if len(numeric_cols) > 0:
            numeric_df = df[numeric_cols]
            labels = date_points[:-1]
            bins = pd.cut(
                numeric_df.index, bins=date_points, labels=labels, include_lowest=True
            )

            if method == "mean":
                numeric_result = numeric_df.groupby(bins).agg(
                    lambda x: x.mean(skipna=skipna)
                )
            elif method == "sum":
                numeric_result = numeric_df.groupby(bins).agg(
                    lambda x: x.sum(skipna=skipna)
                )
            elif method == "last":
                numeric_result = numeric_df.groupby(bins).last()
            elif method == "first":
                numeric_result = numeric_df.groupby(bins).first()
            else:
                raise ValueError(
                    "Method must be one of: 'mean', 'sum', 'last', 'first'"
                )

            result_dfs.append(numeric_result)

        # Handle non-numeric columns
        if len(non_numeric_cols) > 0:
            non_numeric_df = df[non_numeric_cols]
            # Create a new index for resampling
            new_index = pd.DatetimeIndex(date_points[:-1])
            # For each non-numeric column, find the nearest value
            non_numeric_result = pd.DataFrame(index=new_index)

            for col in non_numeric_cols:
                # Create a Series with the original data
                series = non_numeric_df[col]
                # Find nearest values for each target timestamp
                nearest_values = []

                for target_time in new_index:
                    # Calculate time differences
                    time_diffs = abs(series.index - target_time)
                    # Find the index of the minimum time difference
                    nearest_idx = time_diffs.argmin()
                    # Get the corresponding value
                    nearest_values.append(series.iloc[nearest_idx])

                non_numeric_result[col] = nearest_values

            result_dfs.append(non_numeric_result)

        # Combine results if we have both types
        if result_dfs:
            final_result = pd.concat(result_dfs, axis=1)
            # Ensure original column order
            final_result = final_result[df.columns]
            return final_result
        else:
            return pd.DataFrame()

    def resample_time_series(
        self,
        time_column: str = None,
        frequency: str = None,
        method_resample: str = None,
        method_fill: str = None,
        limit: int = None,
        include_all_gaps: bool = True,
        max_gap_size: str = None,
    ) -> pd.DataFrame:
        """
        Resample time series data to a regular frequency and handle gaps.

        Args:
            time_column: Name of the datetime column. If None, uses the one from analysis.
            frequency: Target frequency for resampling. If None, uses inferred frequency.
            method_resample: Method for resampling ('mean', 'sum', 'last')
            method_fill: Method for filling gaps ('ffill', 'bfill', 'interpolate', None)
            limit: Maximum number of consecutive NaN values to fill
            include_all_gaps: Whether to include large gaps in resampling
            max_gap_size: Maximum gap size to include in resampling (e.g., '1H', '30min').
                     Gaps larger than this will be excluded if include_gaps=False.
                     If None, uses min_gap_size from the continuity analysis.

        Returns:
            Resampled DataFrame

        Raises:
            TimeValidationError: If parameters are invalid or required data is missing
        """
        # Ensure we have time series analysis
        if not hasattr(self, "time_series_analysis"):
            self.analyze_time_series_continuity(time_column)

        time_column = time_column or self.time_series_analysis["time_column"]
        frequency = frequency or self.time_series_analysis["inferred_frequency"]

        # Set up gap threshold
        if not include_all_gaps:
            if max_gap_size is None:
                # Use the min_gap_size from the previous analysis
                max_gap_size = self.time_series_analysis.get("min_gap_size", "5min")
            max_gap_duration = pd.Timedelta(max_gap_size)

            # Filter gaps based on max_gap_size
            small_gaps = [
                gap
                for gap in self.time_series_analysis["gaps"]
                if pd.Timedelta(gap["duration"]) < max_gap_duration
            ]
        else:
            small_gaps = []

        # Create copy of DataFrame
        df = self.dataframe.copy()

        # Set time column as index
        df.set_index(time_column, inplace=True)

        # Create regular time index
        if include_all_gaps or not small_gaps:
            # Simple case: include all periods
            full_range = pd.date_range(
                start=df.index.min(), end=df.index.max(), freq=frequency
            )
        else:
            # Build ranges excluding large gaps
            full_range = pd.Index([])
            current_time = df.index.min()

            # Sort gaps by start time
            small_gaps.sort(key=lambda x: x["start_time"])

            for gap in small_gaps:
                # Add range up to gap
                range_to_gap = pd.date_range(
                    start=current_time, end=gap["start_time"], freq=frequency
                )
                full_range = full_range.union(range_to_gap)
                current_time = gap["end_time"]

            # Add final range after last gap
            final_range = pd.date_range(
                start=current_time, end=df.index.max(), freq=frequency
            )
            full_range = full_range.union(final_range)

        # Reindex DataFrame
        if method_resample is None or method_resample == "first":
            resampled = df.reindex(full_range)
        else:
            resampled = self.resample_with_dates(df, full_range, method=method_resample)

        # Fill gaps according to specified method
        if method_fill == "ffill":
            resampled.fillna(method="ffill", limit=limit, inplace=True)
        elif method_fill == "bfill":
            resampled.fillna(method="bfill", limit=limit, inplace=True)
        elif method_fill == "interpolate":
            resampled.interpolate(method="time", limit=limit, inplace=True)

        # Reset index to make time column available again
        resampled.reset_index(inplace=True)
        resampled.rename(columns={"index": time_column}, inplace=True)

        return resampled

    def register_extension(
        self, extension_type: str, extension: Any
    ) -> "FileDataFrame":
        """
        Register a custom extension.

        Args:
            extension_type: Type of extension to register
            extension: Extension implementation

        Returns:
            Self for method chaining

        Raises:
            ValueError: If extension_type is invalid
        """
        if extension_type == "data_transformer" and isinstance(
            extension, DataTransformer
        ):
            self.data_transformer = extension
        elif extension_type == "post_processing_hook" and isinstance(
            extension, PostProcessingHook
        ):
            self.post_processing_hooks.append(extension)
        elif extension_type == "file_validator" and isinstance(
            extension, FileValidator
        ):
            self.file_validator = extension
        elif extension_type == "metadata_extractor" and isinstance(
            extension, MetadataExtractor
        ):
            self.metadata_extractor = extension
        elif extension_type == "time_series_validator" and isinstance(
            extension, TimeSeriesValidator
        ):
            self.time_series_config.time_series_validator = extension
        else:
            # Store in extension_points dictionary for custom extensions
            self.extension_points[extension_type] = extension

        return self

    def get_available_extension_points(self) -> Dict[str, str]:
        """
        Get a dictionary of available extension points and their descriptions.

        Returns:
            Dictionary mapping extension point names to descriptions
        """
        return {
            "data_transformer": "Transform DataFrames during loading",
            "post_processing_hook": "Process the DataFrame after loading and concatenation",
            "file_validator": "Validate files during discovery",
            "metadata_extractor": "Extract metadata from filenames",
            "time_series_validator": "Validate time series continuity",
            "custom": "Custom extension points defined in extension_points dictionary",
        }

    @classmethod
    def with_extensions(
        cls,
        file_source: Union[
            str, Path, List[Union[str, Path]], UploadedFile, List[UploadedFile]
        ],
        extensions: Dict[str, Any],
        **kwargs,
    ) -> "FileDataFrame":
        """
        Create a FileDataFrame with custom extensions.

        Args:
            file_source: Directory path, list of files, or Streamlit uploaded files
            extensions: Dictionary mapping extension types to implementations
            **kwargs: Additional arguments to pass to FileDataFrame constructor

        Returns:
            FileDataFrame instance

        Raises:
            ValueError: If file_source is invalid
        """
        # Extract extensions
        data_transformer = extensions.get("data_transformer")
        post_processing_hooks = extensions.get("post_processing_hooks", [])
        file_validator = extensions.get("file_validator")
        metadata_extractor = extensions.get("metadata_extractor")
        time_series_validator = extensions.get("time_series_validator")

        # Create time_series_config if time_series_validator is provided
        time_series_config = kwargs.get("time_series_config")
        if time_series_validator and not time_series_config:
            time_series_config = TimeSeriesConfig(
                time_series_validator=time_series_validator
            )

        # Determine the type of file source
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if path.is_dir():
                return cls.from_directory(
                    directory_path=path,
                    metadata_extractor=metadata_extractor,
                    data_transformer=data_transformer,
                    post_processing_hooks=post_processing_hooks,
                    file_validator=file_validator,
                    time_series_config=time_series_config,
                    **kwargs,
                )
            else:
                return cls.from_files(
                    file_paths=[path],
                    metadata_extractor=metadata_extractor,
                    data_transformer=data_transformer,
                    post_processing_hooks=post_processing_hooks,
                    file_validator=file_validator,
                    time_series_config=time_series_config,
                    **kwargs,
                )
        elif isinstance(file_source, list):
            if all(isinstance(f, (str, Path)) for f in file_source):
                return cls.from_files(
                    file_paths=file_source,
                    metadata_extractor=metadata_extractor,
                    data_transformer=data_transformer,
                    post_processing_hooks=post_processing_hooks,
                    file_validator=file_validator,
                    time_series_config=time_series_config,
                    **kwargs,
                )
            else:
                return cls.from_streamlit(
                    uploaded_files=file_source,
                    metadata_extractor=metadata_extractor,
                    data_transformer=data_transformer,
                    post_processing_hooks=post_processing_hooks,
                    file_validator=file_validator,
                    time_series_config=time_series_config,
                    **kwargs,
                )
        else:
            return cls.from_streamlit(
                uploaded_files=file_source,
                metadata_extractor=metadata_extractor,
                data_transformer=data_transformer,
                post_processing_hooks=post_processing_hooks,
                file_validator=file_validator,
                time_series_config=time_series_config,
                **kwargs,
            )

    @classmethod
    def create_pipeline(
        cls,
        file_source: Union[
            str, Path, List[Union[str, Path]], UploadedFile, List[UploadedFile]
        ],
        pipeline_steps: List[Dict[str, Any]],
        **kwargs,
    ) -> "FileDataFrame":
        """
        Create a FileDataFrame with a pipeline of custom extensions.

        Args:
            file_source: Directory path, list of files, or Streamlit uploaded files
            pipeline_steps: List of dictionaries defining pipeline steps
            **kwargs: Additional arguments to pass to FileDataFrame constructor

        Returns:
            FileDataFrame instance

        Raises:
            ValueError: If pipeline_steps is invalid

        Example:
            FileDataFrame.create_pipeline(
                "my_data_dir",
                pipeline_steps=[
                    {"type": "data_transformer", "implementation": MyTransformer()},
                    {"type": "post_processing_hook", "implementation": MyHook()}
                ]
            )
        """
        # Initialize extensions dictionary
        extensions = {"post_processing_hooks": []}

        # Process pipeline steps
        for step in pipeline_steps:
            step_type = step.get("type")
            implementation = step.get("implementation")

            if not step_type or not implementation:
                raise ValueError(f"Invalid pipeline step: {step}")

            if step_type == "post_processing_hook":
                extensions["post_processing_hooks"].append(implementation)
            else:
                extensions[step_type] = implementation

        # Create FileDataFrame with extensions
        return cls.with_extensions(file_source, extensions, **kwargs)
