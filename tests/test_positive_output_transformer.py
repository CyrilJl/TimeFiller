import numpy as np
import pandas as pd

from timefiller import PositiveOutput


def test_positive_output():
    m, n = 10_000, 50
    X = np.random.exponential(scale=1, size=(m, n))
    pot = PositiveOutput()
    Xt = pot.fit_transform(X)
    assert pot.thresholds_ is not None
    Xtt = pot.inverse_transform(Xt)
    assert np.allclose(X, Xtt)


def test_positive_output_columns():
    m, n = 10_000, 50
    X = np.random.exponential(scale=1, size=(m, n))
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n)])

    pot = PositiveOutput(columns=df.columns[:10])
    dft = pot.fit_transform(df)
    assert pot.thresholds_ is not None
    assert dft.columns.tolist() == df.columns.tolist()
    assert np.allclose(df[df.columns[10:]], dft[df.columns[10:]])
    dftt = pot.inverse_transform(dft)
    assert np.allclose(df, dftt)
