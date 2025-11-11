import math

# Contains constants and thermodynamic formulas necessary for implementing the Harder & Pomeroy method (2013)

#CONSTANTS
R = 8.31441 # universal gas constant (J mol^-1 K^-1)
M_W = 0.01801528 # molecular weight of water (kg mol^-1)

#FORMULAS
# Diffusivity of water vapor in air
def calc_diffusivity(t_a: float) -> float:
    """Calculate the diffusivity of water vapor in air.
    Args:
        t_a (float): Air temperature in degrees Celsius.
    Returns:
        float: Diffusivity of water vapor in air (m^2 s^-1).
    Reference:
        Harder & Pomeroy (2013) (A.6)
    Note:
        Air temperature is internally converted from °C to K before applying Eq. (A.6).
    """
    t_a_K = t_a + 273.15 # Convert C to K for absolute temperature
    d = 2.06e-5 * (t_a_K / 273.15) ** 1.75
    return d

# Ambient vapor pressure
def calc_ambient_vapor_pressure(t_a:float, r_h:float) -> float:
    """Calculate ambient (actual) vapor pressure over water.
    Args:
        t_a (float): Air temperature in degrees Celsius.
        r_h (float): Relative humidity as a percentage.
    Returns:
        float: Ambient vapor pressure over water in Pa.
    Reference:
        Harder & Pomeroy (2013) (A.7)
    Note:
        Uses a Magnus-type relation over water, then scales by RH.
    """
    e_ta_kPa = (r_h/100) * 0.611 * math.exp((17.3 * t_a) / (237.3 + t_a)) 
    e_ta = e_ta_kPa * 1000 # convert from kPa to Pa
    return e_ta

# Saturation vapor pressure
def get_saturation_vapor_pressure(t_a:float) -> float:
    """Gets the saturation vapor pressure based on the air temperature.
        If t_a <= 0 degrees Celsius, evaluate over ice.
        If t_a > 0 degrees Celsius, evaluate over water.
    Args:
        t_a (float): Air temperature in degrees Celsius.
    Returns:
        float: Saturation vapor pressure in Pa.
    Reference:
        Harder & Pomeroy (2013)
    """
    if (t_a <= 0):
        e_sat = calc_saturation_vapor_pressure_over_ice(t_a)
    else:
        e_sat = calc_saturation_vapor_pressure_over_water(t_a)
    return e_sat

def calc_saturation_vapor_pressure_over_ice(t_a:float) -> float:
    """Calculate saturation vapor pressure over ice.
    Args:
        t_a (float): Air temperature in degrees Celsius.
    Returns:
        float: Saturation vapor pressure over ice in Pa.
    Reference:
        Harder & Pomeroy (2013)
    Note:
        Uses a Magnus-type relation over ice.
    """
    e_sat_kPa = 0.611 * math.exp((22.46 * t_a) / (272.62 + t_a))
    e_sat = e_sat_kPa * 1000 # convert from kPa to Pa
    return e_sat

def calc_saturation_vapor_pressure_over_water(t_a:float) -> float:
    """Calculate saturation vapor pressure over water.
    Args:
        t_a (float): Air temperature in degrees Celsius.
    Returns:
        float: Saturation vapor pressure over water in Pa.
    Reference:
        Harder & Pomeroy (2013)
    Note:
        Uses a Magnus-type relation over water.
    """
    e_sat_kPa = 0.611 * math.exp((17.3 * t_a) / (237.3 + t_a))
    e_sat = e_sat_kPa * 1000 # convert from kPa to Pa
    return e_sat

# Water vapor density
def calc_water_vapor_density(t_a: float, r_h:float) -> float:
    """Calculate water vapor density.
    Args:
        t_a (float): Air temperature in degrees Celsius.
        r_h (float): Relative humidity as a percentage.
    Returns:
        float: Density of water vapor (kg m^-3).
    Reference:
        Harder & Pomeroy (2013) (A.8)
    Note:
        Air temperature is internally converted from °C to K before applying Eq. (A.8).
    """
    e_ta = calc_ambient_vapor_pressure(t_a, r_h) # Get the ambient vapor pressure in Pa
    t_k = t_a + 273.15 # Convert C to K for absolute temperature
    rho_ta = (M_W * e_ta) / (R * t_k) # Ideal gas law for water vapor
    return rho_ta

# Saturation vapor density
def calc_saturation_vapor_density(t_a: float) -> float:
    """Calculate saturation vapor density.
    Args:
        t_a (float): Air temperature in degrees Celsius.
    Returns:
        float: Saturation vapor density (kg m^-3).
    Reference:
        Harder & Pomeroy (2013) (A.8)
    Note:
        Internally computes saturation vapor pressure (Pa) using the appropriate
        phase relation (ice for t_a ≤ 0°C, water for t_a > 0°C), then applies Eq. (A.8).
    """
    e_sat = get_saturation_vapor_pressure(t_a) # Get the saturation vapor pressure in Pa
    t_k = t_a + 273.15 # Convert C to K for absolute temperature
    rho_sat = (M_W * e_sat) / (R * t_k) # Ideal gas law for water vapor
    return rho_sat

# Thermal conductivity of air
def calc_thermal_conductivity(t_a:float) -> float:
    """Calculate the thermal conductivity of air.
    Args:
        t_a (float): Air temperature in degrees Celsius.
    Returns:
        float: Thermal conductivity of air (J m^-1 s^-1 K^-1).
    Reference:
        Harder & Pomeroy (2013) (A.9)
    """
    lambda_t = 0.000063 * t_a + 0.00673
    return lambda_t

# Latent heat
def get_latent_heat(t_a:float) -> float:
    """Gets the latent heat value based on the air temperature.
        If t_a < 0 degrees Celsius, use sublimation.
        If t_a >= 0 degrees Celsius, use vaporization.
    Args:
        t_a (float): Air temperature in degrees Celsius.
    Returns:
        float: The latent heat (J kg^-1), either for sublimation or vaporization depending on t_a.
    Reference:
        Harder & Pomeroy (2013) (A.10, A.11)
    """
    if (t_a < 0):
        l = calc_latent_heat_sublimation(t_a)
    else:
        l = calc_latent_heat_vaporization(t_a)
    return l

def calc_latent_heat_sublimation(t_a:float) -> float:
    """Calculates the latent heat of sublimation.
    Args:
        t_a (float): Air temperature in degrees Celsius.
    Returns:
        float: Latent heat (J kg^-1).
    Reference:
        Harder & Pomeroy (2013) (A.10)
    """
    l = 1000 * (2834.1 - 0.29 * t_a - 0.004 * t_a**2) # latent heat of sublimation (T < 0 C)
    return l

def calc_latent_heat_vaporization(t_a:float) -> float:
    """Calculates the latent heat of vaporization.
    Args:
        t_a (float): Air temperature in degrees Celsius.
    Returns:
        float: Latent heat (J kg^-1).
    Reference:
        Harder & Pomeroy (2013) (A.11)
    """
    l = 1000 * (2501 - (2.361 * t_a)) # latent heat of vaporization (T >= 0 C)
    return l