import numpy as np
import pytest

from transit import flux, limb_darkened_flux


# Test 6: Limb-darkened flux validation against uniform-disk flux.
@pytest.mark.parametrize(
    "z",
    [0.0, 0.5, 1.0, 1.1, 1.2]
)
def test_uniform_disk_validation(z):
    p = 0.1
    u = 0.0
    n = 4000

    reference = flux(p, z)
    result = limb_darkened_flux(p, z, u, n)

    # The n=4000 validation produced a largest discrepancy of
    # 1.82e-7, which decreased to 6.43e-8 at n=8000.
    tolerance = 2e-7

    assert np.isfinite(result)

    assert result == pytest.approx(
        reference,
        rel=0.0,
        abs=tolerance
    )