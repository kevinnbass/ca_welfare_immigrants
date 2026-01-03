"""Shared pytest fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_acs_df():
    """Create a small ACS-like DataFrame for testing."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame(
        {
            "SERIALNO": [f"2023{i:010d}" for i in range(n)],
            "SPORDER": np.random.randint(1, 5, n),
            "PWGTP": np.random.randint(1, 1000, n),
            "NATIVITY": np.random.choice([1, 2], n, p=[0.73, 0.27]),
            "CIT": np.random.choice([1, 2, 3, 4, 5], n),
            "AGEP": np.random.randint(0, 100, n),
            "SEX": np.random.choice([1, 2], n),
            "HINS4": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "FS": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "SSIP": np.random.choice([0, 100, 500], n, p=[0.95, 0.03, 0.02]),
            "PAP": np.random.choice([0, 200], n, p=[0.98, 0.02]),
            # Add replicate weights
            **{f"PWGTP{i}": np.random.randint(1, 1000, n) for i in range(1, 81)},
        }
    )


@pytest.fixture
def sample_weights():
    """Create sample weight arrays for testing."""
    np.random.seed(42)
    return np.random.randint(1, 1000, 100).astype(float)


@pytest.fixture
def sample_indicator():
    """Create sample binary indicator for testing."""
    np.random.seed(42)
    return np.random.choice([0, 1], 100, p=[0.7, 0.3]).astype(float)


@pytest.fixture
def sample_probabilities():
    """Create sample probability values for testing."""
    np.random.seed(42)
    return np.random.uniform(0, 1, 100)


@pytest.fixture
def tmp_zip_file(tmp_path):
    """Create a temporary ZIP file for testing."""
    import zipfile

    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test_file.txt", "test content")

    return zip_path


@pytest.fixture
def malicious_zip_file(tmp_path):
    """Create a ZIP file with path traversal attempt."""
    import zipfile

    zip_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # This simulates a path traversal attempt
        zf.writestr("../../../etc/passwd", "fake passwd content")

    return zip_path
