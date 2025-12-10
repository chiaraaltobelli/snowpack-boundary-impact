import numpy as np
from thermos import (
    calc_diffusivity,
    calc_thermal_conductivity,
    get_latent_heat,
    calc_water_vapor_density,
    calc_saturation_vapor_density,
)

SAFE_T_MIN = -80.0
SAFE_T_MAX = 50.0

def calc_t_i(t_a: float, r_h: float,
             tol: float = 1e-4,
             max_iterations: int = 50) -> float:
    """
    Iteratively calculate the hydrometeor temperature (°C),
    starting with t_i = t_a.

    t_a : air temperature in °C
    r_h : relative humidity in %
    """
    if not np.isfinite(t_a) or not np.isfinite(r_h):
        # optional: comment this out once you trust it
        # print(f"[calc_t_i] bad inputs: T_a={t_a}, RH={r_h}")
        return np.nan

    # Clamp inputs
    t_a = float(np.clip(t_a, SAFE_T_MIN, SAFE_T_MAX))
    r_h = float(np.clip(r_h, 0.0, 100.0))

    # Precompute terms that don't depend on t_i
    t_i = t_a
    d = calc_diffusivity(t_a)
    lambda_t = calc_thermal_conductivity(t_a)
    rho_ta = calc_water_vapor_density(t_a, r_h)

    if not (np.isfinite(d) and np.isfinite(lambda_t) and np.isfinite(rho_ta)):
        return t_a  # fall back: no adjustment

    for _ in range(max_iterations):
        l = get_latent_heat(t_i)
        rho_sat = calc_saturation_vapor_density(t_i)

        if not (np.isfinite(l) and np.isfinite(rho_sat)):
            return t_a

        # CORRECT SIGN: cooling when rho_ta < rho_sat
        delta = (d / lambda_t) * l * (rho_ta - rho_sat)
        t_i_raw = t_a + delta

        # Under-relaxation to help convergence
        t_i_new = 0.5 * t_i + 0.5 * t_i_raw
        t_i_new = float(np.clip(t_i_new, SAFE_T_MIN, SAFE_T_MAX))

        if abs(t_i_new - t_i) < tol:
            t_i = t_i_new
            break

        t_i = t_i_new
    else:
        print(
            f"Warning: calc_t_i did not converge after {max_iterations} iterations. "
            f"Final t_i={t_i:.2f}, T_a={t_a:.2f}, RH={r_h:.1f}"
        )

    return t_i


# Precipitation phase estimation relationship
def calc_rainfall_fraction(t_i:np.ndarray, b:float, c:float) -> np.ndarray:
    """Calculate the rainfall fraction f_r using sigmoidal fit.
    Args:
        t_i (np.ndarray): An array of hydrometeor temperatures in degrees Celsius.
        b (float): Shape coefficient that controls the steepness of the transition 
                   between snow and rain. Larger values make the curve steeper.
        c (float): Shift coefficient that controls the temperature midpoint (the 
                   inflection point where f_r ≈ 0.5).
    Returns:
        np.ndarray: Array of rainfall fractions (0–1), where 0 = all snow, 1 = all rain.
    Reference:
        Harder & Pomeroy (2013) (Eq. 6)
    """    
    f_r = 1 / (1 + b * (c**t_i))
    return f_r

# Determine best fit coefficients to use when calculating rainfall fraction
def fit_bc(t_i:np.ndarray, 
    fr_observed:np.ndarray, 
    weights:np.ndarray,
    epsilon: float = 1e-6) -> tuple[float, float]:

    # Confirm data is all the same length
    if not (t_i.size == fr_observed.size == weights.size):
        raise ValueError("Ti, observed rain fraction, and weights must be the same length.")

    # Clean up data
    t_i_clean, fr_observed_clean, weights_clean = filter_clip_data(t_i, fr_observed, weights, epsilon)

    # Linearize equation 6
    X, y = linearize_data(t_i_clean, fr_observed_clean)

    # Clean/clip weights and apply weighted least squares via sqrt-weighting
    Xw, yw = apply_least_squares(weights_clean, X, y)

    # Solve for [alpha, beta] with least squares
    alpha, beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]

    # Back-transform: b = exp(alpha), c = exp(beta)
    b = float(np.exp(alpha))
    c = float(np.exp(beta))
    
    return b, c

# -----------------------------------------------------------------------------  
# HELPERS
# -----------------------------------------------------------------------------  

# Clean up and format the data
def filter_clip_data(t_i: np.ndarray,
    fr_observed: np.ndarray,
    weights: np.ndarray,
    epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert to arrays
    t_i_arr = np.asarray(t_i)
    fr_observed_array = np.asarray(fr_observed)
    weights_arr = np.asarray(weights)

    # Filter out non-finite entries in all data
    mask = np.isfinite(t_i_arr) & np.isfinite(fr_observed_array) & np.isfinite(weights_arr)
    t_i_mask = t_i_arr[mask]
    fr_observed_mask = fr_observed_array[mask]
    weights_mask = weights_arr[mask]

    # Only clip values that are exactly out of (0,1); leave small-but-positive alone
    fr_observed_clip = fr_observed_mask.copy()
    fr_observed_clip = np.where(fr_observed_clip <= 0.0, epsilon, fr_observed_clip)
    fr_observed_clip = np.where(fr_observed_clip >= 1.0, 1.0 - epsilon, fr_observed_clip)

    return t_i_mask, fr_observed_clip, weights_mask

# Linearize equation 6 -- y = ln(1/fr - 1) = ln b + (ln c) * Ti)
def linearize_data(t_i:np.ndarray, fr_observed:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones_like(t_i), t_i])
    y = np.log(1.0 / fr_observed - 1.0)
    return X, y

# Clean/clip weights and apply weighted least squares via sqrt-weighting
def apply_least_squares(weights:np.ndarray, X:np.ndarray, y:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if not np.any(w):
        sqrtw = np.ones_like(y)
    else:
        sqrtw = np.sqrt(w)
    Xw = X * sqrtw[:, None]
    yw = y * sqrtw
    return Xw, yw