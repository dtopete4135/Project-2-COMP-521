import numpy as np

# Physical constants in CGS units
PLANCK_H = 6.62607015e-27       # erg s
BOLTZMANN_K = 1.380649e-16      # erg K^-1
LIGHT_C = 2.99792458e10         # cm s^-1


def dimensionless_planck(x):
    """
    Evaluate the dimensionless Planck integrand x^3 / (e^x - 1)
    
    x: scalar or NumPy array of nonnegative values

    Returns a finite scalar or array with a limiting value of 0 at x = 0
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    positive = x > 0
    result[positive] = x[positive]**3 / np.expm1(x[positive])   # We only pass positive values through here to avoid a 0 value resulting in 0/0

    if result.ndim == 0:
        return float(result)

    return result



def planck_nu(nu, temperature):
    """
    Evaluate the Planck intensity per unit frequency.

    nu : scalar or array of nonnegative frequencies in Hz
    temperature : positive temperature in K

    Returns the spectral intensity in
    erg cm^-2 s^-1 Hz^-1 sr^-1.
    The limiting value at nu = 0 is set to 0.
    """
    nu = np.asarray(nu, dtype=float)
    result = np.zeros_like(nu)

    positive = nu > 0

    exponent = (
        PLANCK_H * nu[positive]
        / (BOLTZMANN_K * temperature)
    )

    result[positive] = (
        2 * PLANCK_H * nu[positive]**3
        / (LIGHT_C**2 * np.expm1(exponent))
    )

    if result.ndim == 0:
        return float(result)

    return result