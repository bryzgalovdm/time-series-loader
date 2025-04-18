# Time Series Loader - Time Series Data Processing Library

## Overview

FileDataFrame is a powerful Python library for processing time series data from multiple files. It is conceived as a comprehensive framework for file discovery, metadata extraction, time series validation, data loading, and transformation. The library is designed to handle complex time series data processing tasks with robust error handling and extensive configuration options.

### Disclaimer
The work is still in progress.

## Features

- **File Discovery**: Discover files from directories, direct file paths, or Streamlit uploaded files
- **Metadata Extraction**: Extract metadata from filenames using customizable extractors
- **Time Series Validation**: Validate time series continuity, detect gaps and overlaps
- **Data Loading and Transformation**: Load, transform, and concatenate data from multiple files
- **Robust Error Handling**: Comprehensive error tracking and reporting
- **Extensible Architecture**: Custom extension points for data transformation, validation, and processing
- **Time Series Analysis**: Analyze and resample time series data with gap handling

## Installation

It is not installable yet

## Quick Start

### Basic Usage

```python
from time_series_loader.load_file import FileDataFrame

# Create from directory
fdf = FileDataFrame.from_directory("path/to/data", glob_pattern="*.csv")

# Process files
fdf.initialize_processing()

# Get the processed DataFrame
df = fdf.get_dataframe()
```

### Using with Streamlit

```python
import streamlit as st
from filedataframe import FileDataFrame

uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    fdf = FileDataFrame.from_streamlit(uploaded_files)
    fdf.initialize_processing()
    st.dataframe(fdf.get_dataframe())
```

## Configuration Options

### File Discovery Configuration

```python
from filedataframe import FileDataFrame, FileDiscoveryConfig

config = FileDiscoveryConfig(
    glob_pattern="*.csv",
    recursive=True
)

fdf = FileDataFrame(
    base_path="path/to/data",
    discovery_config=config
)
```

### Time Series Configuration

```python
from filedataframe import FileDataFrame, TimeSeriesConfig
from datetime import timedelta

config = TimeSeriesConfig(
    validation_strategy="STRICT",
    max_allowed_gap=timedelta(minutes=5),
    allow_overlap=False
)

fdf = FileDataFrame(
    base_path="path/to/data",
    time_series_config=config
)
```

### Loading Configuration

```python
from filedataframe import FileDataFrame, LoadingConfig

config = LoadingConfig(
    delimiter=";",
    encoding="utf-8",
    time_format="%Y-%m-%d %H:%M:%S"
)

fdf = FileDataFrame(
    base_path="path/to/data",
    loading_config=config
)
```

## Advanced Usage

### Custom Extensions

```python
from filedataframe import FileDataFrame, DataTransformer

class MyTransformer(DataTransformer):
    def transform(self, df, metadata):
        # Custom transformation logic
        return df

fdf = FileDataFrame.with_extensions(
    "path/to/data",
    extensions={"data_transformer": MyTransformer()}
)
```

### Time Series Analysis and Resampling

```python
# Analyze time series continuity
analysis = fdf.analyze_time_series_continuity(
    time_column="timestamp",
    expected_frequency="1min"
)

# Resample time series data
resampled_df = fdf.resample_time_series(
    time_column="timestamp",
    frequency="5min",
    method_resample="mean",
    method_fill="interpolate"
)
```

### Error Handling

```python
# Get error report
error_report = fdf.get_error_report()

# Export detailed error report
fdf.export_detailed_error_report(
    "errors.json",
    include_stacktrace=True,
    min_severity="WARNING"
)
```

## API Documentation

Coming soon

## License

TBD

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.