import logging
import os
import platform
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import pytest
from streamlit.runtime.uploaded_file_manager import UploadedFile

from time_series_loader.error_handling import (
    DataLoadingError,
    ErrorSeverity,
    FileDiscoveryError,
    FileParsingError,
    ProcessingError,
    TimeValidationError,
)
from time_series_loader.file_metadata_parser import TimeMetadataExtractor
from time_series_loader.load_file import FileDataFrame, FileMetadata
from time_series_loader.ts_config import LoadingConfig, TimeSeriesConfig


def get_formatted_path(path):
    if os.name == "nt":  # Windows
        windows_path = PureWindowsPath(path)
        return "\\" + "\\".join(windows_path.parts[1:])
    else:  # Unix-like (Linux, macOS)
        return str(path.resolve())


@pytest.fixture
def file_dataframe(tmp_path):
    log_file = tmp_path / "test_log.log"
    fd = FileDataFrame(
        base_path="/path/to/data",
        time_series_config=TimeSeriesConfig(max_allowed_overlap=timedelta(minutes=15)),
        metadata_extractor=TimeMetadataExtractor(),
        log_file=str(log_file),
    )

    yield fd
    # Teardown: remove the log file
    if log_file.exists():
        shutil.rmtree(log_file.parent, ignore_errors=True)


@pytest.fixture
def mock_folder_structure(tmp_path):
    """Create a mock folder structure for testing"""
    # Create base structure
    type_folder = tmp_path / "Main_SD_1A_Data"
    type_folder.mkdir()

    # Create valid files
    valid_files = [
        "E1 1A - Data - 01-01-2023 00_00_00 - 01-01-2023 23_59_59.csv",
        "E1 1A - Data - 01-02-2023 00_00_00 - 01-02-2023 23_59_59.csv",
        "E1 2A - Data - 01-01-2023 00_00_00 - 01-01-2023 23_59_59.csv",
        "TYPE2 1A - Data - 01-01-2023 00_00_00 - 01-01-2023 23_59_59.csv",
        "invalid_file.csv",  # No times in the good format
    ]

    invalid_files = [
        "E1 2A - Data - 01-01-2023 00_00_00 - 01-01-2023 23_59_59",  # No extension
        "E1 1A - Data - 01-01-2023 00_00_00 - 01-01-2023 23_59_59.txt",  # Wrong extension
    ]

    # Create all files
    for filename in valid_files + invalid_files:
        filepath = type_folder / filename
        filepath.write_text("test data")

    # Create an empty file
    empty_file = (
        type_folder / "E1 1A - Data - 01-03-2023 00_00_00 - 01-03-2023 23_59_59.csv"
    )
    empty_file.touch()

    return type_folder


@pytest.fixture
def file_dataframe_from_folder(mock_folder_structure):
    return FileDataFrame(base_path=str(mock_folder_structure))


def test_file_dataframe_initialization(file_dataframe):
    assert file_dataframe.base_path == Path("/path/to/data")
    assert file_dataframe.time_series_config.max_allowed_gap == timedelta(minutes=15)
    assert file_dataframe.errors == []
    assert file_dataframe.dataframe is None


def test_add_error(file_dataframe):
    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError",
        message="Test error message",
        file_path=Path("/path/to/file.csv"),
        details={"context": "test"},
        stacktrace="test stacktrace",
    )
    file_dataframe._add_error(error)
    assert len(file_dataframe.errors) == 1
    assert file_dataframe.errors[0] == error


def test_add_error_basic(file_dataframe):
    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError",
        message="Test error message",
        file_path=Path("/path/to/file.csv"),
        details={"context": "test"},
        stacktrace="test stacktrace",
    )
    file_dataframe._add_error(error)
    assert len(file_dataframe.errors) == 1
    assert file_dataframe.errors[0] == error


def test_add_error_multiple(file_dataframe):
    errors = [
        ProcessingError(
            timestamp=datetime.now(),
            severity=ErrorSeverity.WARNING,
            error_type=f"TestError{i}",
            message=f"Test error message {i}",
            file_path=Path(f"/path/to/file{i}.csv"),
            details={"context": f"test{i}"},
            stacktrace=f"test stacktrace {i}",
        )
        for i in range(3)
    ]
    for error in errors:
        file_dataframe._add_error(error)
    assert len(file_dataframe.errors) == 3
    assert file_dataframe.errors == errors


def test_add_error_different_severities(file_dataframe):
    severities = [
        ErrorSeverity.INFO,
        ErrorSeverity.WARNING,
        ErrorSeverity.ERROR,
        ErrorSeverity.CRITICAL,
    ]
    for severity in severities:
        error = ProcessingError(
            timestamp=datetime.now(),
            severity=severity,
            error_type=f"TestError{severity.name}",
            message=f"Test {severity.name.lower()} message",
            file_path=Path(f"/path/to/file_{severity.name.lower()}.csv"),
            details={"context": f"test_{severity.name.lower()}"},
            stacktrace=f"test stacktrace {severity.name.lower()}",
        )
        file_dataframe._add_error(error)

    assert len(file_dataframe.errors) == 4
    assert [error.severity for error in file_dataframe.errors] == severities


def test_add_error_no_file_path(file_dataframe):
    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError",
        message="Test error message",
        file_path=None,
        details={"context": "test"},
        stacktrace="test stacktrace",
    )
    file_dataframe._add_error(error)
    assert len(file_dataframe.errors) == 1
    assert file_dataframe.errors[0].file_path is None


def test_add_error_empty_message(file_dataframe):
    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError",
        message="",
        file_path=Path("/path/to/file.csv"),
        details={"context": "test"},
        stacktrace="test stacktrace",
    )
    file_dataframe._add_error(error)
    assert len(file_dataframe.errors) == 1
    assert file_dataframe.errors[0].message == ""


def test_add_error_no_details(file_dataframe):
    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError",
        message="Test error message",
        file_path=Path("/path/to/file.csv"),
        details=None,
        stacktrace="test stacktrace",
    )
    file_dataframe._add_error(error)
    assert len(file_dataframe.errors) == 1
    assert not file_dataframe.errors[0].details


def test_add_error_no_stacktrace(file_dataframe):
    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError",
        message="Test error message",
        file_path=Path("/path/to/file.csv"),
        details={"context": "test"},
        stacktrace=None,
    )
    file_dataframe._add_error(error)
    assert len(file_dataframe.errors) == 1
    assert file_dataframe.errors[0].stacktrace is None


def test_add_error_logging(file_dataframe, caplog):
    path = Path("/path/to/file.csv")

    with caplog.at_level(logging.WARNING):
        error = ProcessingError(
            timestamp=datetime.now(),
            severity=ErrorSeverity.WARNING,
            error_type="TestError",
            message="Test warning message",
            file_path=path,
            details={"context": "test"},
            stacktrace="test stacktrace",
        )
        file_dataframe._add_error(error)

    formatted_path = get_formatted_path(path)
    assert f"TestError: Test warning message (File: {formatted_path})" in caplog.text


def test_add_error_critical_logging(file_dataframe, caplog):
    path = Path("/path/to/critical_file.csv")
    with caplog.at_level(logging.CRITICAL):
        error = ProcessingError(
            timestamp=datetime.now(),
            severity=ErrorSeverity.CRITICAL,
            error_type="TestCriticalError",
            message="Test critical error message",
            file_path=path,
            details={"context": "critical test"},
            stacktrace="critical test stacktrace",
        )
        file_dataframe._add_error(error)

    formatted_path = get_formatted_path(path)
    assert (
        f"TestCriticalError: Test critical error message (File: {formatted_path}"
        in caplog.text
    )


def test_add_error_duplicate(file_dataframe):
    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError",
        message="Test error message",
        file_path=Path("/path/to/file.csv"),
        details={"context": "test"},
        stacktrace="test stacktrace",
    )
    file_dataframe._add_error(error)
    file_dataframe._add_error(error)
    assert len(file_dataframe.errors) == 2  # Duplicates are allowed


def test_handle_error(file_dataframe):

    with pytest.raises(FileDiscoveryError):
        file_dataframe._handle_error(
            FileDiscoveryError("Test critical error"),
            "Test context",
            Path("/path/to/file.csv"),
        )
    assert len(file_dataframe.errors) == 1
    assert file_dataframe.errors[0].severity == ErrorSeverity.CRITICAL
    assert file_dataframe.errors[0].error_type == "FileDiscoveryError"


def test_get_error_report(file_dataframe):
    error1 = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError1",
        message="Test error message 1",
        file_path=Path("/path/to/file1.csv"),
        details={"context": "test1"},
        stacktrace="test stacktrace 1",
    )
    error2 = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.ERROR,
        error_type="TestError2",
        message="Test error message 2",
        file_path=Path("/path/to/file2.csv"),
        details={"context": "test2"},
        stacktrace="test stacktrace 2",
    )
    file_dataframe._add_error(error1)
    file_dataframe._add_error(error2)

    report = file_dataframe.get_error_report(include_stacktrace=True)
    assert report["summary"]["total_errors"] == 2
    assert report["summary"]["error_counts"]["WARNING"] == 1
    assert report["summary"]["error_counts"]["ERROR"] == 1
    assert len(report["errors"]) == 2
    assert "stacktrace" in report["errors"][0]


def test_export_error_report(file_dataframe, tmp_path):
    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError",
        message="Test error message",
        file_path=Path("/path/to/file.csv"),
        details={"context": "test"},
        stacktrace="test stacktrace",
    )
    file_dataframe._add_error(error)

    output_path = tmp_path / "error_report.json"
    file_dataframe.export_error_report(output_path)
    assert output_path.exists()


def test_has_critical_errors(file_dataframe):
    assert not file_dataframe.has_critical_errors()

    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.CRITICAL,
        error_type="TestError",
        message="Test critical error message",
        file_path=Path("/path/to/file.csv"),
        details={"context": "test"},
        stacktrace="test stacktrace",
    )
    file_dataframe._add_error(error)

    assert file_dataframe.has_critical_errors()


def test_log_file_creation_and_deletion(file_dataframe, tmp_path):
    log_file = tmp_path / "test_log.log"

    # Check that the log file was created
    assert log_file.exists(), "Log file was not created"

    # Add an error to ensure something is logged
    error = ProcessingError(
        timestamp=datetime.now(),
        severity=ErrorSeverity.WARNING,
        error_type="TestError",
        message="Test error message",
        file_path=Path("/path/to/file.csv"),
        details={"context": "test"},
        stacktrace="test stacktrace",
    )
    file_dataframe._add_error(error)

    # Force the logger to flush its handlers
    for handler in file_dataframe.logger.handlers:
        handler.flush()

    # Check that the log file is not empty
    assert log_file.stat().st_size > 0, "Log file is empty"


# This test will run last to verify log file deletion
@pytest.mark.last
def test_log_file_deletion(tmp_path):
    log_file = tmp_path / "test_log.log"
    assert not log_file.exists(), "Log file was not deleted after tests"


def test_is_valid_file(file_dataframe_from_folder, mock_folder_structure):
    valid_file = (
        mock_folder_structure
        / "E1 1A - Data - 01-01-2023 00_00_00 - 01-01-2023 23_59_59.csv"
    )
    assert file_dataframe_from_folder._is_valid_file(valid_file) is True

    # Test invalid files
    invalid_cases = [
        mock_folder_structure / "some_file.csv",  # Wrong format
        mock_folder_structure
        / "Main_SD_1A_Data"
        / "E1 1A - Data - 01-01-2023 00_00_00 - 01-01-2023 23_59_59.txt",  # Wrong extension
        mock_folder_structure
        / "Main_SD_1A_Data"
        / "E1 1A - Data - 01-03-2023 00_00_00 - 01-03-2023 23_59_59.csv",  # Empty file
    ]

    for invalid_file in invalid_cases:
        assert file_dataframe_from_folder._is_valid_file(invalid_file) is False


def test_discover_files(file_dataframe_from_folder):
    valid_files = file_dataframe_from_folder.discover_files()
    assert len(valid_files) == 5


def test_discover_files_no_valid_files(mock_folder_structure):
    # Remove all valid files
    for file in mock_folder_structure.rglob("*.csv"):
        if "E1" in str(file) or "1A" in str(file):
            file.unlink()

    fd = FileDataFrame(base_path=str(mock_folder_structure))

    with pytest.raises(
        FileDiscoveryError, match="No valid files found matching pattern"
    ):
        fd.discover_files()


def test_get_discovery_stats(file_dataframe_from_folder):
    # Should raise error before discovery
    with pytest.raises(FileDiscoveryError, match="Discovery statistics not available"):
        file_dataframe_from_folder.get_discovery_stats()

    # Run discovery
    file_dataframe_from_folder.discover_files()

    # Check stats
    stats = file_dataframe_from_folder.get_discovery_stats()
    assert stats["valid_files"] == 5
    assert stats["invalid_files"] == 3  # Including empty file
    assert "invalid_files_details" in stats


def test_discover_files_permission_error(mock_folder_structure):
    type_folder = mock_folder_structure

    try:
        # Make sure the folder exists before modifying permissions
        assert type_folder.exists(), f"Test folder {type_folder} does not exist"

        if platform.system() == "Windows":
            # On Windows, use icacls with proper path formatting and quotes
            folder_path = str(type_folder).replace("/", "\\")
            # Deny read permission for the current user
            subprocess.run(
                f'icacls "{folder_path}" /deny "%USERNAME%":(R)', shell=True, check=True
            )
        else:
            # On Unix-like systems, use chmod to remove read permissions
            # 0o000 is too restrictive and might cause issues with cleanup
            # 0o300 allows write and execute but not read
            os.chmod(type_folder, 0o300)

        # Now run the actual test
        fd = FileDataFrame(base_path=str(mock_folder_structure))

        with pytest.raises(FileDiscoveryError, match="No read permission"):
            fd.discover_files()

    finally:
        # Always restore permissions to ensure cleanup can happen
        if platform.system() == "Windows":
            folder_path = str(type_folder).replace("/", "\\")
            # Grant read permission back
            subprocess.run(
                f'icacls "{folder_path}" /grant "%USERNAME%":(R)',
                shell=True,
                # Don't use check=True here to ensure cleanup happens even if command fails
            )
            # Additional reset to ensure permissions are restored
            subprocess.run(f'icacls "{folder_path}" /reset', shell=True)
        else:
            # Restore standard permissions (read, write, execute for owner)
            os.chmod(type_folder, 0o755)


class TestMetadataExtraction:
    @pytest.fixture
    def valid_file(self, tmp_path):
        file_path = (
            tmp_path / "E1 1A - Data - 01-01-2023 00_00_00 - 01-02-2023 23_59_59.csv"
        )
        file_path.write_text("test data")
        return file_path

    def test_extract_metadata_valid(self, file_dataframe, valid_file):
        metadata = file_dataframe.extract_metadata(valid_file)

        assert isinstance(metadata, FileMetadata)
        assert metadata.filepath == valid_file
        assert isinstance(metadata.start_time, datetime)
        assert isinstance(metadata.end_time, datetime)
        assert metadata.start_time < metadata.end_time
        assert metadata.start_time.year == 2023
        assert metadata.start_time.month == 1
        assert metadata.start_time.day == 1
        assert metadata.end_time.day == 2

    def test_extract_metadata_nonexistent_file(self, file_dataframe, tmp_path):
        nonexistent_file = tmp_path / "nonexistent.csv"

        with pytest.raises(FileParsingError) as exc_info:
            file_dataframe.extract_metadata(nonexistent_file)
        assert "not a file or doesn't exist" in str(exc_info.value)

    def test_extract_metadata_invalid_filename(self, file_dataframe, tmp_path):
        invalid_file = tmp_path / "invalid_filename.csv"
        invalid_file.write_text("test data")

        with pytest.raises(FileParsingError) as exc_info:
            file_dataframe.extract_metadata(invalid_file)
        assert "doesn't match expected pattern" in str(exc_info.value)


class TestFileProcessing:
    @pytest.fixture
    def valid_files(self, tmp_path):
        files = [
            "E1 1A - Data - 01-01-2023 00_00_00 - 01-01-2023 23_59_59.csv",
            "E1 1A - Data - 01-02-2023 00_00_00 - 01-02-2023 23_59_59.csv",
            "E1 1A - Data - 01-03-2023 00_00_00 - 01-03-2023 23_59_59.csv",
        ]

        created_files = []
        for filename in files:
            file_path = tmp_path / filename
            file_path.write_text("test data")
            created_files.append(file_path)

        return created_files

    def test_process_files_valid(self, file_dataframe, valid_files):
        metadata_list = file_dataframe.process_files(valid_files)

        assert len(metadata_list) == 3
        assert all(isinstance(m, FileMetadata) for m in metadata_list)

        # Check that metadata was stored in the instance
        assert file_dataframe.metadata == metadata_list

    def test_process_files_empty_list(self, file_dataframe):
        metadata_list = file_dataframe.process_files([])
        assert len(metadata_list) == 0
        assert file_dataframe.metadata == []

    def test_process_files_mixed_valid_invalid(self, file_dataframe, tmp_path):
        # Create mix of valid and invalid files
        valid_file = (
            tmp_path / "E1 1A - Data - 01-01-2023 00_00_00 - 01-01-2023 23_59_59.csv"
        )
        invalid_file = tmp_path / "invalid_file.csv"
        nonexistent_file = tmp_path / "nonexistent.csv"

        valid_file.write_text("test data")
        invalid_file.write_text("test data")

        files = [valid_file, invalid_file, nonexistent_file]

        with pytest.raises(FileParsingError) as exc_info:
            file_dataframe.process_files(files)

        assert "Multiple errors" in str(exc_info.value)
        # Check that the error message contains information about all failed files
        error_message = str(exc_info.value)
        assert "nonexistent.csv" in error_message

    @pytest.mark.parametrize(
        "files",
        [
            None,
            "not_a_list",
            [123, 456],  # non-Path objects
        ],
    )
    def test_process_files_invalid_input(self, file_dataframe, files):
        with pytest.raises(
            Exception
        ):  # Type of exception depends on your error handling
            file_dataframe.process_files(files)


class TestDataFrameLoadingAndValidation:
    @pytest.fixture
    def sample_csv_content(self):
        """Create sample CSV content"""
        return (
            "int_col;float_col;str_col;datetime_col\n"
            "1;1.1;a;2023-01-01\n"
            "2;2.2;b;2023-01-02\n"
            "3;3.3;c;2023-01-03\n"
        )

    @pytest.fixture
    def sample_csv_file(self, tmp_path, sample_csv_content):
        """Create a sample CSV file"""
        file_path = tmp_path / "test.csv"
        file_path.write_text(sample_csv_content)
        return file_path

    @pytest.fixture
    def file_metadata(self, sample_csv_file):
        """Create a FileMetadata instance"""
        return FileMetadata(
            filepath=sample_csv_file,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
        )

    @pytest.fixture
    def file_dataframe(self):
        return FileDataFrame(base_path="/path/to/data")

    @pytest.fixture
    def loaded_file_dataframe(self, file_dataframe, file_metadata):
        """FileDataFrame instance with one file already loaded"""
        _ = file_dataframe._load_single_file(file_metadata)
        return file_dataframe

    def test_load_single_file_success(self, file_dataframe, file_metadata):
        """Test successful loading of a single file"""
        df = file_dataframe._load_single_file(file_metadata)

        # Check basic DataFrame properties
        assert not df.empty
        assert all(
            col in df.columns
            for col in ["int_col", "float_col", "str_col", "datetime_col"]
        )

        # Check metadata columns were added
        assert "source_file" in df.columns
        assert "file_start_time" in df.columns
        assert "file_end_time" in df.columns

        # Check metadata values
        assert str(file_metadata.filepath) in df["source_file"].iloc[0]
        assert df["file_start_time"].iloc[0] == file_metadata.start_time
        assert df["file_end_time"].iloc[0] == file_metadata.end_time

    def test_load_single_file_empty(self, tmp_path, file_dataframe):
        """Test loading an empty file"""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        metadata = FileMetadata(
            filepath=empty_file,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
        )

        with pytest.raises(DataLoadingError) as exc_info:
            file_dataframe._load_single_file(metadata)
        assert "empty" in str(exc_info.value)

    # skip this test when delimiter is present explicitly
    @pytest.mark.skip
    def test_load_single_file_invalid_csv(self, tmp_path, file_dataframe):
        """Test loading an invalid CSV file

        WARNING: ParserError is very difficult to trigger once delimiter
        or separator has been set explicitly. See pandas.read_csv(). For this
        reason, corresponding test is disabled.
        """
        invalid_file = tmp_path / "invalid.csv"
        invalid_file.write_text("invalid/tcsv\tformat\nincomplete")

        metadata = FileMetadata(
            filepath=invalid_file,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
        )

        with pytest.raises(DataLoadingError) as exc_info:
            file_dataframe._load_single_file(metadata)
        assert "Error parsing CSV file" in str(exc_info.value)

    def test_load_multiple_files_matching_structure(
        self, tmp_path, file_dataframe, sample_csv_content
    ):
        """Test loading multiple files with matching structure"""
        # Create two identical structure files
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        file1.write_text(sample_csv_content)
        file2.write_text(sample_csv_content)

        metadata1 = FileMetadata(
            filepath=file1,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
        )
        metadata2 = FileMetadata(
            filepath=file2,
            start_time=datetime(2023, 1, 2),
            end_time=datetime(2023, 1, 3),
        )

        # Should load without errors
        _ = file_dataframe._load_single_file(metadata1)
        _ = file_dataframe._load_single_file(metadata2)

    def test_load_files_structure_mismatch(self, tmp_path, loaded_file_dataframe):
        """Test loading files with mismatched structure"""
        # Create file with different structure
        file_path = tmp_path / "one_file.csv"
        different_file = tmp_path / "different.csv"
        file_path.write_text(
            "int_col;float_col;str_col;datetime_col\n1;1.1;a;2023-01-01\n2;2.2;b;2023-01-02\n"
        )
        different_file.write_text("different_col;another_col\n1;2\n3;4")

        metadata = FileMetadata(
            filepath=file_path,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
        )
        metadata_diff = FileMetadata(
            filepath=different_file,
            start_time=datetime(2023, 1, 2),
            end_time=datetime(2023, 1, 3),
        )

        loaded = loaded_file_dataframe._load_single_file(metadata)
        loaded_file_dataframe.dataframe = loaded  # Imitate a file already loaded
        loaded_file_dataframe._columns = set(loaded.columns)
        loaded_file_dataframe._dtypes = loaded.dtypes
        with pytest.raises(DataLoadingError) as exc_info:
            loaded_file_dataframe._load_single_file(metadata_diff)
        assert "Column mismatch" in str(exc_info.value)

    def test_load_file_dtype_mismatch(self, tmp_path, loaded_file_dataframe):
        """Test loading file with mismatched data types"""
        # Create file with same columns but different types
        # Create file with different structure
        file_path = tmp_path / "one_file.csv"
        different_file = tmp_path / "different.csv"
        file_path.write_text(
            "int_col;float_col;str_col;datetime_col\n1;1.1;a;2023-01-01\n2;2.2;b;2023-01-02\n"
        )
        different_file.write_text(
            "int_col;float_col;str_col;datetime_col\na;1.1;a;2023-01-01\nb;2.2;b;2023-01-02\n"
        )

        metadata = FileMetadata(
            filepath=file_path,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
        )
        metadata_diff = FileMetadata(
            filepath=different_file,
            start_time=datetime(2023, 1, 2),
            end_time=datetime(2023, 1, 3),
        )

        loaded = loaded_file_dataframe._load_single_file(metadata)
        loaded_file_dataframe.dataframe = loaded  # Imitate a file already loaded
        loaded_file_dataframe._columns = set(loaded.columns) - set(
            ["source_file", "file_start_time", "file_end_time"]
        )
        loaded_file_dataframe._dtypes = loaded.dtypes[:-3]
        with pytest.raises(DataLoadingError) as exc_info:
            loaded_file_dataframe._load_single_file(metadata_diff)
        assert "Data type mismatch" in str(exc_info.value)

    @pytest.mark.parametrize("delimiter", [",", "\t", "|"])
    def test_load_file_different_delimiters(self, tmp_path, file_dataframe, delimiter):
        """Test loading files with different delimiters"""
        # Create file with specified delimiter
        content = f"col1{delimiter}col2\n1{delimiter}2\n3{delimiter}4"
        file = tmp_path / "delimiter_test.csv"
        file.write_text(content)

        metadata = FileMetadata(
            filepath=file,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
        )

        # Change the delimiter
        file_dataframe.loading_config.delimiter = delimiter
        df = file_dataframe._load_single_file(metadata)
        assert set(df.columns) == {
            "col1",
            "col2",
            "source_file",
            "file_start_time",
            "file_end_time",
        }


class TestDataFrameLoadingAndConcatenation:
    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample CSV files with different timestamps"""
        files = [
            (
                "file1.csv",
                "Type1 SubType - Column1 - Value;Type1 SubType - Time;Extra - Data\n"
                "1;01/01/2023 10:00;A\n"
                "2;01/01/2023 11:00;B",
            ),
            (
                "file2.csv",
                "Type1 SubType - Column1 - Value;Type1 SubType - Time;Extra - Data\n"
                "3;01/01/2023 12:00;C\n"
                "4;01/01/2023 13:00;D",
            ),
        ]

        file_paths = []
        for filename, content in files:
            file_path = tmp_path / filename
            file_path.write_text(content)
            file_paths.append(file_path)

        return file_paths

    @pytest.fixture
    def file_metadata_list(self, sample_files):
        """Create FileMetadata objects for sample files"""
        return [
            FileMetadata(
                filepath=file,
                start_time=datetime(2023, 1, 1, 10),
                end_time=datetime(2023, 1, 1, 12),
            )
            for file in sample_files
        ]

    @pytest.fixture
    def file_dataframe(self):
        return FileDataFrame(
            base_path="/path/to/data",
            loading_config=LoadingConfig(timestamp_column="Type1 SubType - Time"),
        )

    @pytest.fixture
    def loaded_dataframe(self, file_dataframe, file_metadata_list):
        """FileDataFrame instance with loaded and concatenated data"""
        file_dataframe.metadata = file_metadata_list
        file_dataframe.load_and_concatenate()
        return file_dataframe

    @pytest.mark.skip(reason="TODO: do much more comprehensive testing")
    def test_clean_column_name(self, loaded_dataframe):
        """Test column name cleaning"""
        test_cases = [
            ("Type1 SubType - Column Name", "Column Name"),
            ("Simple Column", "Simple Column"),
            ("A - B - C - D", "D"),
            ("No Dash", "No Dash"),
            ("  Spaces  -  Extra  ", "Extra"),
        ]

        for input_name, expected in test_cases:
            assert loaded_dataframe._apply_column_naming_configuration() == expected

    @pytest.mark.skip(reason="TODO: do much more comprehensive testing")
    def test_rename_columns(self, loaded_dataframe):
        """Test column renaming"""
        df = loaded_dataframe.dataframe
        assert all(not col.startswith("Type1 SubType -") for col in df.columns)
        assert "Value" in df.columns
        assert "Time" in df.columns
        assert "Data" in df.columns

    @pytest.mark.skip(reason="TODO: do much more comprehensive testing")
    def test_rename_columns_no_dataframe(self, file_dataframe):
        """Test renaming columns when no DataFrame is loaded"""
        with pytest.raises(FileNotFoundError):
            file_dataframe._rename_columns()

    def test_load_and_concatenate_success(self, loaded_dataframe):
        """Test successful loading and concatenation"""
        df = loaded_dataframe.dataframe
        assert len(df) == 4  # Total rows from both files
        assert "Value" in df.columns
        assert "Time" in df.columns
        assert df["Time"].dtype == "datetime64[ns]"
        assert df["Time"].is_monotonic_increasing  # Check if sorted

    def test_load_and_concatenate_no_metadata(self, file_dataframe):
        """Test concatenation without metadata"""
        with pytest.raises(DataLoadingError, match="No metadata available"):
            file_dataframe.load_and_concatenate()

    def test_load_and_concatenate_with_errors(self, file_dataframe, file_metadata_list):
        """Test concatenation with file loading errors"""
        # Modify one filepath to be invalid
        file_metadata_list[0] = FileMetadata(
            filepath=Path("nonexistent.csv"),
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
        )

        file_dataframe.metadata = file_metadata_list
        with pytest.raises(DataLoadingError, match="Errors loading files"):
            file_dataframe.load_and_concatenate()

    def test_get_dataframe(self, loaded_dataframe):
        """Test getting the loaded DataFrame"""
        df = loaded_dataframe.get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_get_dataframe_not_loaded(self, file_dataframe):
        """Test getting DataFrame when not loaded"""
        with pytest.raises(DataLoadingError, match="DataFrame not loaded"):
            file_dataframe.get_dataframe()

    def test_get_concat_metadata(self, loaded_dataframe):
        """Test getting concatenation metadata"""
        metadata = loaded_dataframe.get_concat_metadata()
        assert isinstance(metadata, dict)
        assert "total_rows" in metadata
        assert "total_files" in metadata
        assert "memory_usage" in metadata
        assert "start_time" in metadata
        assert "end_time" in metadata
        assert metadata["total_rows"] == 4
        assert metadata["total_files"] == 2

    def test_get_concat_metadata_not_loaded(self, file_dataframe):
        """Test getting metadata when not loaded"""
        with pytest.raises(
            DataLoadingError, match="Concatenation metadata not available"
        ):
            file_dataframe.get_concat_metadata()

    def test_load_and_concatenate_memory_usage(self, loaded_dataframe):
        """Test memory usage calculation"""
        metadata = loaded_dataframe.get_concat_metadata()
        assert metadata["memory_usage"] > 0


class TestAnalyzeTimeSeriesContinuity:
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range(start="2023-01-01", end="2023-01-02", freq="1h")
        df = pd.DataFrame({"timestamp": dates, "value": range(len(dates))})
        # Create a gap
        df = df.drop(df.index[5:7])
        return df

    @pytest.fixture
    def file_dataframe_with_data(self, sample_df):
        fd = FileDataFrame(base_path="/path/to/data")
        fd.dataframe = sample_df
        return fd

    def test_analyze_time_series_continuity_success(self, file_dataframe_with_data):
        report = file_dataframe_with_data.analyze_time_series_continuity(
            time_column="timestamp"
        )
        assert report["time_column"] == "timestamp"
        assert report["inferred_frequency"] == "3600s"
        assert report["total_gaps"] == 1
        assert len(report["gaps"]) == 1
        assert report["coverage_percentage"] < 100

    def test_analyze_time_series_continuity_no_data(self, file_dataframe_with_data):
        file_dataframe_with_data.dataframe = None
        with pytest.raises(DataLoadingError):
            file_dataframe_with_data.analyze_time_series_continuity()

    def test_analyze_time_series_continuity_no_datetime(self, file_dataframe_with_data):
        file_dataframe_with_data.dataframe = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(TimeValidationError):
            file_dataframe_with_data.analyze_time_series_continuity()

    def test_analyze_time_series_continuity_custom_params(
        self, file_dataframe_with_data
    ):
        report = file_dataframe_with_data.analyze_time_series_continuity(
            time_column="timestamp", expected_frequency="30min", min_gap_size="2h"
        )
        assert report["inferred_frequency"] == "30min"
        assert report["min_gap_size"] == "2h"


class TestUpsampleDF:
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range(start="2023-01-01", end="2023-01-02", freq="1h")
        return pd.DataFrame({"timestamp": dates, "value": range(len(dates))}).set_index(
            "timestamp"
        )

    def test_upsample_df_mean(self, sample_df):
        from math import ceil

        upsampled = FileDataFrame.upsample_df(sample_df, "2h", method="mean")
        assert len(upsampled) == ceil(len(sample_df) / 2)
        assert upsampled.index.freq == "2h"

    def test_upsample_df_sum(self, sample_df):
        upsampled = FileDataFrame.upsample_df(sample_df, "2h", method="sum")
        assert (upsampled["value"] == sample_df["value"].resample("2h").sum()).all()

    def test_upsample_df_last(self, sample_df):
        upsampled = FileDataFrame.upsample_df(sample_df, "2h", method="last")
        assert (upsampled["value"] == sample_df["value"].resample("2h").last()).all()

    def test_upsample_df_invalid_method(self, sample_df):
        with pytest.raises(ValueError):
            FileDataFrame.upsample_df(sample_df, "2h", method="invalid")


class TestResampleWithDates:
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range(start="2023-01-01", end="2023-01-02", freq="1h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "numeric": range(n),
                "category": ["A", "B"] * (n // 2) + ["A"] * (n % 2),
            }
        )
        return df.set_index("timestamp")

    def test_resample_with_dates_mean(self, sample_df):
        date_points = pd.date_range(start="2023-01-01", end="2023-01-02", freq="4h")
        resampled = FileDataFrame.resample_with_dates(
            sample_df, date_points, method="mean"
        )
        assert len(resampled) == len(date_points) - 1
        assert "numeric" in resampled.columns
        assert "category" in resampled.columns
        assert resampled["numeric"].iloc[0] == 2.0  # Mean of 0, 1, 2, 3
        assert resampled["category"].iloc[0] == "A"  # First value in the 4-hour period

    def test_resample_with_dates_sum(self, sample_df):
        date_points = pd.date_range(start="2023-01-01", end="2023-01-02", freq="4h")
        resampled = FileDataFrame.resample_with_dates(
            sample_df, date_points, method="sum"
        )
        assert resampled["numeric"].iloc[0] == 10  # Sum of 0, 1, 2, 3
        assert resampled["category"].iloc[0] == "A"  # First value in the 4-hour period

    def test_resample_with_dates_last(self, sample_df):
        date_points = pd.date_range(start="2023-01-01", end="2023-01-02", freq="4h")
        resampled = FileDataFrame.resample_with_dates(
            sample_df, date_points, method="last"
        )
        assert resampled["numeric"].iloc[0] == 4  # Last value in the 4-hour period
        assert resampled["category"].iloc[0] == "A"  # Last value in the 4-hour period

    def test_resample_with_dates_first(self, sample_df):
        date_points = pd.date_range(start="2023-01-01", end="2023-01-02", freq="4h")
        resampled = FileDataFrame.resample_with_dates(
            sample_df, date_points, method="first"
        )
        assert resampled["numeric"].iloc[0] == 0  # First value in the 4-hour period
        assert resampled["category"].iloc[0] == "A"  # First value in the 4-hour period

    def test_resample_with_dates_invalid_method(self, sample_df):
        date_points = pd.date_range(start="2023-01-01", end="2023-01-02", freq="4h")
        with pytest.raises(ValueError):
            FileDataFrame.resample_with_dates(sample_df, date_points, method="invalid")

    def test_resample_with_dates_with_nan(self, sample_df):
        sample_df.loc[sample_df.index[5:10], "numeric"] = np.nan
        date_points = pd.date_range(start="2023-01-01", end="2023-01-02", freq="4h")
        resampled_skipna = FileDataFrame.resample_with_dates(
            sample_df, date_points, method="mean", skipna=True
        )
        resampled_with_na = FileDataFrame.resample_with_dates(
            sample_df, date_points, method="mean", skipna=False
        )
        assert not np.isnan(resampled_skipna["numeric"]).all()
        assert np.isnan(resampled_with_na["numeric"]).any()


class TestResampleTimeSeries:
    @pytest.fixture
    def file_dataframe_with_gaps(self):
        dates = pd.date_range(start="2023-01-01", end="2023-01-02", freq="1h")
        df = pd.DataFrame({"timestamp": dates, "value": range(len(dates))})
        # Create gaps
        df = df.drop(df.index[5:7])  # 2-hour gap
        df = df.drop(df.index[15:20])  # 5-hour gap

        fd = FileDataFrame(base_path="/path/to/data")
        fd.dataframe = df
        fd.analyze_time_series_continuity(time_column="timestamp")
        return fd

    def test_resample_time_series_basic(self, file_dataframe_with_gaps):
        resampled = file_dataframe_with_gaps.resample_time_series(
            time_column="timestamp",
            frequency="30min",
            method_resample="mean",
            method_fill="ffill",
        )
        assert len(resampled) > len(file_dataframe_with_gaps.dataframe)

        # Check if the resampled data has the correct frequency
        resampled["timestamp"] = pd.to_datetime(
            resampled["timestamp"]
        )  # convert to datetime
        time_diff = resampled["timestamp"].diff()
        assert (time_diff[1:] == pd.Timedelta("30min")).all()

    @pytest.mark.skip  # Due to inconsistencies with time formatting this is skipped for now
    def test_resample_time_series_interpolate(self, file_dataframe_with_gaps):
        resampled = file_dataframe_with_gaps.resample_time_series(
            time_column="timestamp",
            frequency="30min",
            method_resample="mean",
            method_fill="interpolate",
        )
        assert not resampled["value"].isna().any()
        assert (
            resampled["value"].diff().abs() <= 0.5
        ).all()  # Check if interpolated values are reasonable

    def test_resample_time_series_no_fill(self, file_dataframe_with_gaps):
        resampled = file_dataframe_with_gaps.resample_time_series(
            time_column="timestamp",
            frequency="30min",
            method_resample="mean",
            method_fill=None,
        )
        assert resampled["value"].isna().any()
        resampled["timestamp"] = pd.to_datetime(
            resampled["timestamp"]
        )  # convert to datetime
        assert (resampled["timestamp"].diff()[1:] == pd.Timedelta("30min")).all()

    def test_resample_time_series_invalid_params(self, file_dataframe_with_gaps):
        with pytest.raises(ValueError):
            file_dataframe_with_gaps.resample_time_series(
                time_column="timestamp", frequency="30min", method_resample="invalid"
            )

    def test_resample_time_series_different_frequencies(self, file_dataframe_with_gaps):
        frequencies = ["15min", "1h", "2h", "1D"]
        for freq in frequencies:
            resampled = file_dataframe_with_gaps.resample_time_series(
                time_column="timestamp",
                frequency=freq,
                method_resample="mean",
                method_fill="ffill",
            )
            resampled["timestamp"] = pd.to_datetime(
                resampled["timestamp"]
            )  # convert to datetime
            assert (resampled["timestamp"].diff()[1:] == pd.Timedelta(freq)).all()

    def test_resample_time_series_with_non_numeric_column(
        self, file_dataframe_with_gaps
    ):
        # Add a non-numeric column
        file_dataframe_with_gaps.dataframe["category"] = ["A", "B"] * (
            len(file_dataframe_with_gaps.dataframe) // 2
        )

        resampled = file_dataframe_with_gaps.resample_time_series(
            time_column="timestamp",
            frequency="30min",
            method_resample="first",
            method_fill="ffill",
        )
        assert "category" in resampled.columns
        assert resampled["category"].dtype == object


class TestDirectFileInput:
    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample files for direct file input testing"""
        files = [
            (
                "D2 1B - Data_Tot - 01-01-2024 00_00_00 - 01-02-2024 23_59_59.csv",
                "col1;col2\n1;2\n3;4",
            ),
            (
                "D2 1B - Data_Tot - 01-03-2024 00_00_00 - 01-04-2024 23_59_59.csv",
                "col1;col2\n5;6\n7;8",
            ),
        ]

        file_paths = []
        for filename, content in files:
            file_path = tmp_path / filename
            file_path.write_text(content)
            file_paths.append(str(file_path))

        return file_paths

    def test_direct_file_input_initialization(self, sample_files):
        """Test initialization with direct file input"""
        fd = FileDataFrame(files=sample_files)
        assert fd.files is not None
        assert len(fd.files) == 2
        assert fd.base_path is None

    def test_direct_file_input_invalid_files(self, tmp_path):
        """Test direct file input with invalid files"""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("invalid content")

        with pytest.raises(FileDiscoveryError, match="No valid files found"):
            FileDataFrame(files=[str(invalid_file)]).discover_files()

    def test_direct_file_input_complete_processing(self, sample_files):
        """Test complete processing pipeline with direct file input"""
        fd = FileDataFrame(files=sample_files)
        fd.initialize_processing()

        assert fd.dataframe is not None
        assert len(fd.metadata) == 2


class TestStreamlitFileInput:
    @pytest.fixture
    def mock_uploaded_file(self, mocker):
        """Create a mock UploadedFile class using mocker"""

        def create_mock(name: str, content: bytes):
            mock_file = mocker.MagicMock()
            mock_file.name = name
            mock_file.getvalue.return_value = content
            # Make isinstance(mock_file, UploadedFile) return True
            mock_file.__class__ = UploadedFile
            return mock_file

        return create_mock

    @pytest.fixture
    def sample_streamlit_files(self, mock_uploaded_file):
        """Create sample Streamlit uploaded files"""
        return [
            mock_uploaded_file(
                "3Q14 - Data_All - 01-01-2022 00_00_00 - 12-31-2022 23_59_59.csv",
                "col1;col2\n1;2\n3;4".encode("utf-8"),
            ),
            mock_uploaded_file(
                "3Q14 - Data_All - 01-01-2023 00_00_00 - 12-31-2023 23_59_59.csv",
                "col1;col2\n5;6\n7;8".encode("utf-8"),
            ),
        ]

    def test_streamlit_file_initialization(self, sample_streamlit_files):
        """Test initialization with Streamlit file input"""
        fd = FileDataFrame(
            streamlit_files=sample_streamlit_files,
            metadata_extractor=TimeMetadataExtractor(),
        )
        assert fd.streamlit_files is not None
        assert fd.base_path is None
        assert fd.files is None

    def test_streamlit_single_file(self, mock_uploaded_file):
        """Test initialization with single Streamlit file"""
        single_file = mock_uploaded_file(
            "3Q14 - Data_All - 01-01-2022 00_00_00 - 12-31-2022 23_59_59.csv",
            "col1;col2\n1;2\n3;4".encode("utf-8"),
        )
        fd = FileDataFrame(
            streamlit_files=single_file, metadata_extractor=TimeMetadataExtractor()
        )
        fd.initialize_processing()

        assert fd.dataframe is not None
        assert len(fd.metadata) == 1

    def test_streamlit_multiple_files(self, sample_streamlit_files):
        """Test processing multiple Streamlit files"""
        fd = FileDataFrame(
            streamlit_files=sample_streamlit_files,
            metadata_extractor=TimeMetadataExtractor(),
        )
        fd.initialize_processing()

        assert fd.dataframe is not None
        assert len(fd.metadata) == 2

    def test_streamlit_invalid_file_content(self, mock_uploaded_file):
        """Test handling invalid file content"""
        invalid_file = mock_uploaded_file(
            "3Q14 - Data_All - 01-01-2022 00_00_00 - 12-31-2022 23_59_59.csv",
            "col1\n1;2;3\n4;5;6\n7;8\nladslf;;sdf;".encode(
                "utf-8"
            ),  # Mismatched columns will cause parsing error
        )
        fd = FileDataFrame(
            streamlit_files=invalid_file, metadata_extractor=TimeMetadataExtractor()
        )

        with pytest.raises(DataLoadingError, match="Error parsing CSV file"):
            fd.initialize_processing()

    def test_streamlit_empty_file(self, mock_uploaded_file):
        """Test handling empty file"""
        empty_file = mock_uploaded_file(
            "3Q14 - Data_All - 01-01-2022 00_00_00 - 12-31-2022 23_59_59.csv",
            "".encode("utf-8"),
        )
        fd = FileDataFrame(streamlit_files=empty_file)

        with pytest.raises(FileDiscoveryError, match="empty"):
            fd.initialize_processing()

    def test_streamlit_invalid_filename(self, mock_uploaded_file):
        """Test handling invalid filename"""
        invalid_file = mock_uploaded_file(
            "invalid_filename.csv", "col1;col2\n1;2".encode("utf-8")
        )
        fd = FileDataFrame(
            streamlit_files=invalid_file, metadata_extractor=TimeMetadataExtractor()
        )

        with pytest.raises(FileDiscoveryError, match="No valid files found"):
            fd.initialize_processing()

    def test_streamlit_data_structure(self, sample_streamlit_files):
        """Test the structure of loaded data from Streamlit files"""
        fd = FileDataFrame(
            streamlit_files=sample_streamlit_files,
            metadata_extractor=TimeMetadataExtractor(),
        )
        fd.initialize_processing()

        # Check DataFrame structure
        assert all(col in fd.dataframe.columns for col in ["col1", "col2"])
        assert "source_file" in fd.dataframe.columns
        assert "file_start_time" in fd.dataframe.columns
        assert "file_end_time" in fd.dataframe.columns

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(fd.dataframe["file_start_time"])
        assert pd.api.types.is_datetime64_any_dtype(fd.dataframe["file_end_time"])

    def test_streamlit_discovery_stats(self, sample_streamlit_files):
        """Test discovery statistics for Streamlit files"""
        fd = FileDataFrame(
            streamlit_files=sample_streamlit_files,
            metadata_extractor=TimeMetadataExtractor(),
        )
        fd.initialize_processing()

        stats = fd.get_discovery_stats()
        assert stats["total_files_found"] == 2
        assert stats["valid_files"] == 2
        assert stats["invalid_files"] == 0
