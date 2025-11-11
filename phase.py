import math
from thermos import (
    calc_diffusivity,
    calc_thermal_conductivity,
    get_latent_heat,
    calc_water_vapor_density,
    calc_saturation_vapor_density)

# Psychrometric energy balance
def calc_t_i(t_a: float, r_h:float) -> float:
    """Iteratively calculate the hydrometeor temperature, starting with an approximation that t_i = t_a.
    Args:
        t_a (float): Air temperature in degrees Celsius.
        r_h (float): Relative humidity as a percentage.
    Returns:
        float: The hydrometeor temperature in degrees Celsius.
    Reference:
        Harder & Pomeroy (2013) (A.5)
    """
    # Set tolerance and max iterations
    tol = 1e-3 # Convergence tolerance in degrees Celsius
    max_iterations = 100 # Avoid infinite loop

    # Get values for calculations
    t_i = t_a # Use an approximate value for t_i, setting equal to t_a
    d = calc_diffusivity(t_a) # Get the diffusivity, D
    lambda_t = calc_thermal_conductivity(t_a) # Get the thermal conductivity
    l = get_latent_heat(t_a) # Get the latent heat (sublimation or vaporization)
    rho_ta = calc_water_vapor_density(t_a,r_h) # Get the ambient water vapor density

    # Loop until t_i is less than stated tolerance
    for _ in range(max_iterations):
        rho_sat = calc_saturation_vapor_density(t_i) # Get the saturation vapor density
        t_i_new = t_a + (d / lambda_t) * l * (rho_ta - rho_sat)
        if abs(t_i_new - t_i) < tol:
            break
        t_i = t_i_new
    else:
        print(f"Warning: calc_t_i did not converge after {max_iterations} iterations.")
    
    return t_i

# Precipitation phase estimation relationship
def calc_rainfall_fraction(t_i:float, b:float, c:float) -> float:
    """Calculate the rainfall fraction f_r using sigmoidal fit.
    Args:
        t_i (float): Hydrometeor temperature in degrees Celsius.
        b (float): Shape coefficient that controls the steepness of the transition 
                   between snow and rain. Larger values make the curve steeper.
        c (float): Shift coefficient that controls the temperature midpoint (the 
                   inflection point where f_r ≈ 0.5).
    Returns:
        float: Rainfall fraction (0–1), where 0 = all snow, 1 = all rain.
    Reference:
        Harder & Pomeroy (2013) (Eq. 6)
    """    
    f_r = 1 / (1 + b * (c**t_i))
    return f_r