import numpy as np
from thermos import (
    calc_diffusivity,
    calc_thermal_conductivity,
    get_latent_heat,
    calc_water_vapor_density,
    calc_saturation_vapor_density)

# Psychrometric energy balance
def calc_t_i(t_a: float, r_h: float, tol: float = 1e-6, max_iterations: int = 1000) -> float:
    """Iteratively calculate the hydrometeor temperature, starting with an approximation that t_i = t_a.
    Args:
        t_a (float): Air temperature in degrees Celsius.
        r_h (float): Relative humidity as a percentage.
    Returns:
        float: The hydrometeor temperature in degrees Celsius.
    Reference:
        Harder & Pomeroy (2013) (A.5)
    """
    # Get values for calculations
    t_i = t_a # Use an approximate value for t_i, setting equal to t_a
    d = calc_diffusivity(t_a) # Get the diffusivity, D
    lambda_t = calc_thermal_conductivity(t_a) # Get the thermal conductivity
    rho_ta = calc_water_vapor_density(t_a,r_h) # Get the ambient water vapor density

    # Loop until t_i is less than stated tolerance
    for _ in range(max_iterations):
        l = get_latent_heat(t_i) # Get the latent heat (sublimation or vaporization)
        rho_sat = calc_saturation_vapor_density(t_i) # Get the saturation vapor density
        t_i_new = t_a + (d / lambda_t) * l * (rho_ta - rho_sat)
        if abs(t_i_new - t_i) < tol:
            break
        t_i = t_i_new
    else:
        print(f"Warning: calc_t_i did not converge after {max_iterations} iterations.")
    
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