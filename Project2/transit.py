import numpy as np

from integrate import simpson

def kappa0(p, z):
    """
    Calculate the first angle in the partial-overlap formula.

    p : planet-to-star radius ratio
    z : center-to-center separation in stellar radii

    Returns kappa0 in radians.
    """
    argument = (p**2 + z**2 - 1) / (2 * p * z)

    # Protect arccos from small floating-point excursions.
    argument = np.clip(argument, -1.0, 1.0)

    return np.arccos(argument)



def kappa1(p, z):
    """
    Calculate the second angle in the partial-overlap formula.

    p : planet-to-star radius ratio
    z : center-to-center separation in stellar radii

    Returns kappa1 in radians.
    """
    argument = (1 - p**2 + z**2) / (2 * z)

    # Protect arccos from small floating-point excursions.
    argument = np.clip(argument, -1.0, 1.0)

    return np.arccos(argument)



def flux(p, z):
    """
    Calculate the normalized flux for a uniform stellar disk.

    p : positive planet-to-star radius ratio
    z : nonnegative center-to-center separation in stellar radii

    Returns the visible fraction of the star's light.
    """
    if not np.isscalar(p) or not np.isscalar(z):
        raise ValueError("p and z must be scalars.")

    try:
        p = float(p)
        z = float(z)
    except (TypeError, ValueError):
        raise ValueError("p and z must be numeric.")

    if not np.isfinite(p) or not np.isfinite(z):
        raise ValueError("p and z must be finite.")

    if p <= 0:
        raise ValueError("p must be positive.")

    if z < 0:
        raise ValueError("z must be nonnegative.")

    if z >= 1 + p:
        return 1.0

    if z <= p - 1:
        return 0.0

    if z <= 1 - p:
        return 1 - p**2

    overlap = (
        p**2 * kappa0(p, z)
        + kappa1(p, z)
        - np.sqrt(
            (
                4 * z**2
                - (1 + z**2 - p**2)**2
            ) / 4
        )
    )

    return 1 - overlap / np.pi



def separation(times, a_over_rstar, period):
    """
    Calculate projected separation for a central transit.

    times : scalar or array of times in days
    a_over_rstar : orbital distance in stellar-radius units
    period : orbital period in days

    Returns separation in stellar-radius units.
    """
    times = np.asarray(times, dtype=float)

    return a_over_rstar * np.abs(
        np.sin(2 * np.pi * times / period)
    )



def uniform_transit_curve(times, p, a_over_rstar, period):
    """
    Calculate a uniform-disk transit light curve.

    times : scalar or array of times in days
    a_over_rstar : orbital distance in stellar-radius units
    period : orbital period in days

    Returns uniform transit light curve.    
    """
    z_values = separation(
        times,
        a_over_rstar,
        period
    )

    return np.array([
        flux(p, z)
        for z in z_values
    ])



def intensity(r, u):
    """
    Calculate the linear limb-darkened stellar intensity.

    r : scalar or array of radii in stellar-radius units
    u : linear limb-darkening coefficient

    Returns intensity with central intensity I0 = 1.
    """
    r = np.asarray(r, dtype=float)

    mu = np.sqrt(
        np.clip(1 - r**2, 0.0, None)
    )

    result = 1 - u * (1 - mu)

    if result.ndim == 0:
        return float(result)

    return result



def annulus_coverage(r, z, p):
    """
    Calculate the fraction of an annulus covered by the planet.

    r : scalar or array of stellar radii
    z : nonnegative center-to-center separation
    p : planet-to-star radius ratio

    Returns the covered fraction from 0 to 1.
    """
    r = np.asarray(r, dtype=float)
    coverage = np.zeros_like(r)

    # For a central transit, annuli inside the planet are fully covered.
    if z == 0:
        coverage[r <= p] = 1.0

        if coverage.ndim == 0:
            return float(coverage)

        return coverage

    fully_covered = z + r <= p

    not_covered = (
        (z > r + p)
        | (z < r - p)
    )

    partial_overlap = ~(
        fully_covered
        | not_covered
    )

    coverage[fully_covered] = 1.0
    coverage[not_covered] = 0.0

    # Evaluate arccos only where partial overlap occurs.
    r_partial = r[partial_overlap]

    argument = (
        r_partial**2 + z**2 - p**2
    ) / (
        2 * r_partial * z
    )

    argument = np.clip(argument, -1.0, 1.0)

    coverage[partial_overlap] = (
        np.arccos(argument) / np.pi
    )

    if coverage.ndim == 0:
        return float(coverage)

    return coverage



def radial_integrand(r, p, z, u):
    """
    Evaluate the normalized blocked-flux integrand.

    r : scalar or array of stellar radii
    p : planet-to-star radius ratio
    z : center-to-center separation
    u : linear limb-darkening coefficient

    Returns the normalized blocked intensity per unit radius.
    """
    r = np.asarray(r, dtype=float)
    result = np.zeros_like(r)

    total_flux = np.pi * (1 - u / 3)

    # The integrand is explicitly zero at r = 0 because its
    # annular area factor 2*pi*r vanishes.
    positive = r > 0

    result[positive] = (
        intensity(r[positive], u)
        * 2
        * np.pi
        * r[positive]
        * annulus_coverage(r[positive], z, p)
        / total_flux
    )

    if result.ndim == 0:
        return float(result)

    return result



def limb_darkened_flux(p, z, u, n):
    """
    Calculate normalized observed flux for a limb-darkened star.

    p : planet-to-star radius ratio, with 0 < p <= 1
    z : nonnegative center-to-center separation
    u : linear limb-darkening coefficient
    n : positive even number of Simpson subintervals

    Returns normalized observed flux.
    """
    if not np.isscalar(p) or not np.isscalar(z):
        raise ValueError("p and z must be scalars.")

    p = float(p)
    z = float(z)

    if not np.isfinite(p) or not np.isfinite(z):
        raise ValueError("p and z must be finite.")

    if p <= 0 or p > 1:
        raise ValueError("p must satisfy 0 < p <= 1.")

    if z < 0:
        raise ValueError("z must be nonnegative.")

    if z == 0:
        # For a central transit, integrate only over the covered
        # interval because annuli with r > p are uncovered.
        blocked_flux = simpson(
            lambda r: radial_integrand(r, p, z, u),
            0.0,
            p,
            n
        )
    else:
        blocked_flux = simpson(
            lambda r: radial_integrand(r, p, z, u),
            0.0,
            1.0,
            n
        )

    return float(1 - blocked_flux)