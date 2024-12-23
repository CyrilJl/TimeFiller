import numpy as np
import pandas as pd
import pytest

from timefiller import PositiveOutput


@pytest.fixture
def random_data():
    """Fixture to generate random data for testing."""
    m, n = 10_000, 50
    X = np.random.exponential(scale=1, size=(m, n))
    return X


@pytest.fixture
def random_dataframe():
    """Fixture to generate random dataframe for testing."""
    m, n = 10_000, 50
    X = np.random.exponential(scale=1, size=(m, n))
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n)])
    return df


def test_positive_output(random_data):
    """Test the PositiveOutput class with numpy array data."""
    X = random_data
    pot = PositiveOutput()
    Xt = pot.fit_transform(X)

    assert pot.thresholds_ is not None, "Thresholds should be initialized after fitting."

    Xtt = pot.inverse_transform(Xt)
    assert np.allclose(X, Xtt), "Original and inverse-transformed data should match."


def test_positive_output_columns(random_dataframe):
    """Test the PositiveOutput class with pandas DataFrame data."""
    df = random_dataframe
    pot = PositiveOutput(columns=df.columns[:10])

    dft = pot.fit_transform(df)
    assert pot.thresholds_ is not None, "Thresholds should be initialized after fitting."
    assert dft.columns.tolist() == df.columns.tolist(), "Column names should remain unchanged."
    assert np.allclose(df[df.columns[10:]], dft[df.columns[10:]]), "Unselected columns should remain unchanged."

    dftt = pot.inverse_transform(dft)
    assert np.allclose(df, dftt), "Original and inverse-transformed DataFrame should match."
