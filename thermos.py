import math
import numpy as np

# Contains constants and thermodynamic formulas necessary for
# implementing the Harder & Pomeroy method (2013)

# CONSTANTS
R = 8.31441              # universal gas constant (J mol^-1 K^-1)
M_W = 0.01801528         # molecular weight of water (kg mol^-1)
EPSILON_Q = 0.622        # ratio of dry air / water vapor gas constants (for mixing ratio to vapor pressure)
CELSIUS_TO_KELVIN = 273.15  # WRF in Kelvins, formulas based in Celsius

SAFE_T_MIN = -80.0
SAFE_T_MAX = 50.0

# ---------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------

def _clamp_T(t_a: float, tmin: float = SAFE_T_MIN, tmax: float = SAFE_T_MAX) -> float:
    """Clamp temperature to a physically reasonable range and handle non-finite."""
    if not np.isfinite(t_a):
        return np.nan
    return float(np.clip(t_a, tmin, tmax))


def _safe_magnus_exp(t_a: float, a: float, b: float, c: float,
                     tmin: float, tmax: float) -> float:
    """
    Safe evaluation of Magnus-type expression:
        e_sat_kPa = a * exp(b * t / (c + t))

    with clamping on temperature and exponent.
    """
    t = _clamp_T(t_a, tmin=tmin, tmax=tmax)

    denom = c + t
    if abs(denom) < 1e-3:
        # Avoid huge exponents from division by ~0
        denom = 1e-3 if denom >= 0 else -1e-3

    exponent = (b * t) / denom
    exponent = float(np.clip(exponent, -700.0, 700.0))

    return a * math.exp(exponent)

# ---------------------------------------------------------------------
# FORMULAS
# ---------------------------------------------------------------------

# Diffusivity of water vapor in air
def calc_diffusivity(t_a: float) -> float:
    """
    Diffusivity of water vapor in air (m^2 s^-1).
    Harder & Pomeroy (2013) (A.6)
    """
    t_a_K = _clamp_T(t_a) + CELSIUS_TO_KELVIN
    d = 2.06e-5 * (t_a_K / CELSIUS_TO_KELVIN) ** 1.75
    return d


# Ambient vapor pressure
def calc_ambient_vapor_pressure(t_a: float, r_h: float) -> float:
    """
    Ambient (actual) vapor pressure over water, in Pa.
    Uses saturation over water and scales by RH.
    """
    if not np.isfinite(t_a) or not np.isfinite(r_h):
        return np.nan

    r_h = float(np.clip(r_h, 0.0, 100.0))
    e_sat = calc_saturation_vapor_pressure_over_water(t_a)  # Pa
    return (r_h / 100.0) * e_sat


# Vapor pressure from mixing ratio
def calc_vapor_pressure_from_mixing_ratio(q: float, p: float) -> float:
    """
    Vapor pressure from water vapor mixing ratio and pressure.
    e = q * p / (EPSILON_Q + q)
    """
    return q * p / (EPSILON_Q + q)


# Saturation vapor pressure dispatcher
def get_saturation_vapor_pressure(t_a: float) -> float:
    """
    Saturation vapor pressure in Pa.
    Ice for t_a <= 0°C, water for t_a > 0°C.
    """
    if t_a <= 0.0:
        return calc_saturation_vapor_pressure_over_ice(t_a)
    else:
        return calc_saturation_vapor_pressure_over_water(t_a)


def calc_saturation_vapor_pressure_over_ice(t_a: float) -> float:
    """
    Saturation vapor pressure over ice (Pa).
    Harder & Pomeroy (2013), Magnus-type relation.
    """
    # a=0.611 kPa, b=22.46, c=272.62
    e_sat_kPa = _safe_magnus_exp(t_a, a=0.611, b=22.46, c=272.62,
                                 tmin=-80.0, tmax=0.0)
    return e_sat_kPa * 1000.0


def calc_saturation_vapor_pressure_over_water(t_a: float) -> float:
    """
    Saturation vapor pressure over water (Pa).
    Harder & Pomeroy (2013), Magnus-type relation.
    """
    # a=0.611 kPa, b=17.3, c=237.3
    e_sat_kPa = _safe_magnus_exp(t_a, a=0.611, b=17.3, c=237.3,
                                 tmin=-40.0, tmax=50.0)
    return e_sat_kPa * 1000.0


# Relative humidity from T, q, p
def calc_relative_humidity_from_q2(t_a: float, q: float, p: float) -> float:
    """
    Relative humidity (%) from air temperature (°C), mixing ratio (kg/kg), and pressure (Pa).
    """
    if not (np.isfinite(t_a) and np.isfinite(q) and np.isfinite(p) and p > 0):
        return np.nan

    e = calc_vapor_pressure_from_mixing_ratio(q, p)
    e_sat = calc_saturation_vapor_pressure_over_water(t_a)

    if e_sat <= 0.0 or not np.isfinite(e_sat):
        return np.nan

    rh = 100.0 * e / e_sat
    return float(np.clip(rh, 0.0, 100.0))


# Water vapor density
def calc_water_vapor_density(t_a: float, r_h: float) -> float:
    """
    Water vapor density (kg m^-3).
    Harder & Pomeroy (2013) (A.8)
    """
    e_ta = calc_ambient_vapor_pressure(t_a, r_h)
    if not np.isfinite(e_ta):
        return np.nan

    t_k = _clamp_T(t_a) + CELSIUS_TO_KELVIN
    rho_ta = (M_W * e_ta) / (R * t_k)
    return rho_ta


# Saturation vapor density
def calc_saturation_vapor_density(t_a: float) -> float:
    """
    Saturation vapor density (kg m^-3).
    Harder & Pomeroy (2013) (A.8)
    """
    e_sat = get_saturation_vapor_pressure(t_a)
    if not np.isfinite(e_sat):
        return np.nan

    t_k = _clamp_T(t_a) + CELSIUS_TO_KELVIN
    rho_sat = (M_W * e_sat) / (R * t_k)
    return rho_sat


# Thermal conductivity of air
def calc_thermal_conductivity(t_a: float) -> float:
    """
    Thermal conductivity of air (J m^-1 s^-1 K^-1).
    Harder & Pomeroy (2013) (A.9)
    """
    t = _clamp_T(t_a)
    lambda_t = 0.000063 * t + 0.00673
    # Avoid division by zero in later calculations:
    return max(lambda_t, 1e-6)


# Latent heat dispatch
def get_latent_heat(t_a: float) -> float:
    """
    Latent heat (J kg^-1).
    Sublimation for T < 0°C, vaporization for T >= 0°C.
    """
    if t_a < 0.0:
        return calc_latent_heat_sublimation(t_a)
    else:
        return calc_latent_heat_vaporization(t_a)


def calc_latent_heat_sublimation(t_a: float) -> float:
    """
    Latent heat of sublimation (J kg^-1).
    Harder & Pomeroy (2013) (A.10)
    """
    t = _clamp_T(t_a, tmin=-80.0, tmax=0.0)
    return 1000.0 * (2834.1 - 0.29 * t - 0.004 * t**2)


def calc_latent_heat_vaporization(t_a: float) -> float:
    """
    Latent heat of vaporization (J kg^-1).
    Harder & Pomeroy (2013) (A.11)
    """
    t = _clamp_T(t_a, tmin=0.0, tmax=50.0)
    return 1000.0 * (2501.0 - 2.361 * t)
