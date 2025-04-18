def get_column_union(column_lists: list) -> set:
    """
    Get union of all columns across all files.

    Args:
        column_lists: List of lists containing column names

    Returns:
        Set of all unique column names

    Example:
        >>> cols = [['col1', 'col2'], ['col1', 'col2', 'col3'], ['col1']]
        >>> get_column_union(cols)
        {'col1', 'col2', 'col3'}
    """
    return set().union(*column_lists)


def get_common_columns(column_lists: list) -> set:
    """
    Get columns that appear in all files.

    Args:
        column_lists: List of lists containing column names

    Returns:
        Set of column names that appear in all files

    Example:
        >>> cols = [['col1', 'col2'], ['col1', 'col2', 'col3'], ['col1']]
        >>> get_common_columns(cols)
        {'col1'}
    """
    common_columns = set(column_lists[0]).intersection(*column_lists)
    return common_columns


def get_different_columns(column_lists: list) -> set:
    """
    Get columns that appear in only one file.

    Args:
        column_lists: List of lists containing column names

    Returns:
        Set of column names that appear in exactly one file

    Example:
        >>> cols = [['col1', 'col2'], ['col1', 'col2', 'col3'], ['col1']]
        >>> get_different_columns(cols)
        {'col3'}
    """
    # Create a dictionary to count occurrences of each column
    from collections import Counter

    column_counts = Counter()

    # Count occurrences of each column
    for columns in column_lists:
        # Convert to set to count each column only once per file
        column_counts.update(set(columns))

    # Return columns that appear in exactly one file
    return {col for col, count in column_counts.items() if count == 1}
