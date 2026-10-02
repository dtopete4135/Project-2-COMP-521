import numpy as np
import pytest
from integrate import rectangle, trapezoid, simpson, adaptive_simpson


# Test 1: Rectangle and trapezoid on the sine integral, n = 1000.
def test_rectangle_sine():
    result = rectangle(np.sin, 0.0, np.pi / 2, 1000)

    # The first-order rectangle error bound is approximately 0.00124.
    assert result == pytest.approx(1.0, rel=0.0, abs=0.0013)


def test_trapezoid_sine():
    result = trapezoid(np.sin, 0.0, np.pi / 2, 1000)

    # The second-order trapezoid error bound is approximately 3.23e-7.
    assert result == pytest.approx(1.0, rel=0.0, abs=3.3e-7)



# Test 2: Simpson on the same integral and grid.
def test_simpson_sine():
    result = simpson(np.sin, 0.0, np.pi / 2, 1000)

    # The Simpson error bound is approximately 5.313e-14.
    # This tolerance also allows for floating-point rounding.
    assert result == pytest.approx(1.0, rel=0.0, abs=1e-13)



# Test 3: Simpson on the integral of x^3 from 0 to 1. 2 n subintervals.
def test_simpson_cubic():
    result = simpson(lambda x: x**3, 0.0, 1.0, 2)

    # Simpson is exact for cubic polynomials in exact arithmetic.
    # This tolerance allows for floating-point rounding.
    assert result == pytest.approx(0.25, rel=0.0, abs=1e-15)



# Test 4: Fixed rules with improper inputs. Simpson with odd n.
@pytest.mark.parametrize("rule", [rectangle, trapezoid, simpson])
@pytest.mark.parametrize("invalid_n", [0, -2, 2.5, 4.0, True])
def test_fixed_rules_reject_invalid_n(rule, invalid_n):
    with pytest.raises(ValueError):
        rule(np.sin, 0.0, np.pi / 2, invalid_n)


def test_simpson_rejects_odd_n():
    with pytest.raises(ValueError):
        simpson(np.sin, 0.0, np.pi / 2, 3)



# Test 7: Adaptive Simpson on sin, invalid tol and max_depth.
def test_adaptive_simpson_sine():
    result = adaptive_simpson(
        np.sin,
        0.0,
        np.pi / 2,
        tol=1e-10
    )

    # The sine integrand is smooth, and the requested absolute
    # tolerance is 1e-10.
    assert result == pytest.approx(
        1.0,
        rel=0.0,
        abs=1e-10
    )

@pytest.mark.parametrize(
    "invalid_tol",
    [0.0, -1e-6, np.inf, np.nan, True]
)
def test_adaptive_simpson_rejects_invalid_tol(invalid_tol):
    with pytest.raises(ValueError):
        adaptive_simpson(
            np.sin,
            0.0,
            np.pi / 2,
            tol=invalid_tol
        )

@pytest.mark.parametrize(
    "invalid_depth",
    [-1, 2.5, True]
)
def test_adaptive_simpson_rejects_invalid_depth(
    invalid_depth
):
    with pytest.raises(ValueError):
        adaptive_simpson(
            np.sin,
            0.0,
            np.pi / 2,
            tol=1e-8,
            max_depth=invalid_depth
        )



# Test 8: Adaptive Simpson with delibertately insufficient depth.
def test_adaptive_simpson_insufficient_depth():
    with pytest.raises(RuntimeError):
        adaptive_simpson(
            np.sin,
            0.0,
            np.pi / 2,
            tol=1e-15,
            max_depth=0
        )