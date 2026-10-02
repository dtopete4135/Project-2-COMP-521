import numpy as np
import pytest

from integrate import simpson
from planck import dimensionless_planck


# Test 5: Simpson on the dimensionless Planck integral.
def test_dimensionless_planck_integral():
    result = simpson(
        dimensionless_planck,
        0,
        40,
        40000
    )

    exact = np.pi**4 / 15

    # At cutoff b = 40, the n = 20,000 and n = 40,000 results agreed.
    # The tolerance is placed slightly above that resolved discrepancy.

    assert result == pytest.approx(
        exact,
        rel=0,
        abs=5e-13
    )