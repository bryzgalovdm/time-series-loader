from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Optional, Union

from .file_metadata_parser import FileFilter
from .ts_validator import TimeSeriesValidator, ValidationStrategy


@dataclass
class FileDiscoveryConfig:
    """Configuration for file discovery."""

    glob_pattern: str = "*"
    recursive: bool = False
    file_filter: Optional[FileFilter] = None


@dataclass
class LoadingConfig:
    """Configuration for data loading."""

    delimiter: str = ";"
    decimal: str = "."
    timestamp_column: str = None
    time_format: str = "%d/%m/%Y %H:%M"
    encoding: str = "utf-8"
    parse_dates: bool = True


@dataclass
class ColumnNamingConfig:
    """Configuration for column naming."""

    clean_column_names: bool = True
    strip_column_whitespace: bool = True
    column_rename_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimeSeriesConfig:
    """Configuration for time series validation."""

    validation_strategy: Union[ValidationStrategy, str] = ValidationStrategy.LENIENT
    max_allowed_gap: timedelta = timedelta(minutes=15)
    allow_overlap: bool = False
    max_allowed_overlap: timedelta = timedelta(seconds=0)
    fail_on_validation_error: bool = True
    time_series_validator: Optional[TimeSeriesValidator] = None
