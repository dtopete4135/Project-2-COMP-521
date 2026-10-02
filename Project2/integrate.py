import numpy as np

# Rectangle Rule
def rectangle(f, a, b, n):
    """
    Approximate an integral using n left rectangles

    f : accepts and returns matching arrays of finite values
    a, b : finite scalar limits with a < b
    n : positive integer 

    Returns a float approximation of the integral
    Raises ValueError if n is not a positive integer
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be a positive integer")

    # A valid integer type can still have an invalid value.
    if n <= 0:
        raise ValueError("n must be a positive integer")

    width = (b - a) / n
    nodes = a + width * np.arange(n)

    return float(width * np.sum(f(nodes)))



# Trapezoid Rule
def trapezoid(f, a, b, n):
    """
    Approximate an integral using n trapezoids
    
    f : accepts and returns matching arrays of finite values
    a, b : finite scalar limits with a < b
    n : positive integer 

    Returns a float approximation of the integral
    Raises ValueError if n is not a positive integer
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be a positive integer")

    # A valid integer type can still have an invalid value.
    if n <= 0:
        raise ValueError("n must be a positive integer")

    width = (b - a) / n
    nodes = np.linspace(a, b, n + 1)
    values = f(nodes)

    return float(width * (
        0.5 * values[0]
        + np.sum(values[1:-1])
        + 0.5 * values[-1]
    ))



# Simpson Rule
def simpson(f, a, b, n):
    """
    Approximate an integral using n Simpson subintervals

    f : accepts and returns matching arrays of finite values
    a, b : finite scalar limits with a < b
    n : positive even integer

    Returns a float approximation of the integral
    Raises ValueError if n is not a positive even integer
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be a positive even integer")

    # A valid integer type can still have an invalid value.
    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be a positive even integer")

    width = (b - a) / n
    nodes = np.linspace(a, b, n + 1)
    values = f(nodes)

    odd_sum = np.sum(values[1:-1:2])
    even_sum = np.sum(values[2:-1:2])

    return float((width / 3) * (
        values[0]
        + 4 * odd_sum
        + 2 * even_sum
        + values[-1]
    ))



# Adaptive Simpson Rule
def adaptive_simpson(f, a, b, tol, max_depth=50):
    """
    Approximate an integral using adaptive Simpson integration.

    f : accepts a scalar and returns a finite real scalar
    a, b : finite scalar limits with a < b
    tol : finite positive error tolerance
    max_depth : nonnegative Python or NumPy integer

    Returns a float approximation of the integral.
    Raises ValueError for invalid tol or max_depth.
    Raises RuntimeError if the tolerance cannot be met within max_depth.
    """
    if (
        isinstance(tol, (bool, np.bool_))
        or not np.isscalar(tol)
    ):
        raise ValueError("tol must be a finite positive number")

    try:
        tol = float(tol)
    except (TypeError, ValueError):
        raise ValueError("tol must be a finite positive number")

    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite positive number")

    if (
        isinstance(max_depth, (bool, np.bool_))
        or not isinstance(max_depth, (int, np.integer))
        or max_depth < 0
    ):
        raise ValueError(
            "max_depth must be a nonnegative integer"
        )

    def simpson_interval(left, right, f_left, f_mid, f_right):
        """
        Calculate Simpson's estimate from three known function values.
        """
        return (
            (right - left)
            * (f_left + 4 * f_mid + f_right)
            / 6
        )

    midpoint = (a + b) / 2

    # Further subdivision is impossible if floating-point rounding
    # makes the midpoint equal to an endpoint.
    if midpoint == a or midpoint == b:
        raise RuntimeError(
            "Midpoint is indistinguishable from an endpoint"
        )

    f_a = float(f(a))
    f_midpoint = float(f(midpoint))
    f_b = float(f(b))

    whole = simpson_interval(
        a,
        b,
        f_a,
        f_midpoint,
        f_b
    )

    def refine(
        left,
        right,
        f_left,
        f_mid,
        f_right,
        whole_estimate,
        local_tol,
        depth
    ):
        """
        Recursively refine one interval.
        """
        mid = (left + right) / 2
        left_mid = (left + mid) / 2
        right_mid = (mid + right) / 2

        # Stop if required new points cannot be represented.
        if (
            left_mid == left
            or left_mid == mid
            or right_mid == mid
            or right_mid == right
        ):
            raise RuntimeError(
                "Required midpoint is indistinguishable "
                "from an endpoint"
            )

        # Only the two quarter-point values are new.
        f_left_mid = float(f(left_mid))
        f_right_mid = float(f(right_mid))

        left_estimate = simpson_interval(
            left,
            mid,
            f_left,
            f_left_mid,
            f_mid
        )

        right_estimate = simpson_interval(
            mid,
            right,
            f_mid,
            f_right_mid,
            f_right
        )

        fine_estimate = (
            left_estimate + right_estimate
        )

        difference = (
            fine_estimate - whole_estimate
        )

        # Acceptance is allowed even at max_depth.
        if abs(difference) <= 15 * local_tol:
            return (
                fine_estimate
                + difference / 15
            )

        # Raise before attempting another subdivision.
        if depth >= max_depth:
            raise RuntimeError(
                "Tolerance not met within max_depth"
            )

        left_result = refine(
            left,
            mid,
            f_left,
            f_left_mid,
            f_mid,
            left_estimate,
            local_tol / 2,
            depth + 1
        )

        right_result = refine(
            mid,
            right,
            f_mid,
            f_right_mid,
            f_right,
            right_estimate,
            local_tol / 2,
            depth + 1
        )

        return left_result + right_result

    return float(
        refine(
            a,
            b,
            f_a,
            f_midpoint,
            f_b,
            whole,
            tol,
            0
        )
    )
