# test_thermos.py

import math
import numpy as np
import pytest
import thermos

# ---------------------------------------------------------------------------
# calc_saturation_vapor_pressure_over_water / calc_saturation_vapor_pressure_over_ice
# ---------------------------------------------------------------------------

def test_sat_vapor_pressure_over_water_monotonic():
    """Saturation vapor pressure over water should increase with temperature."""
    temps = np.array([-5.0, 0.0, 5.0, 10.0, 20.0])  # °C
    e_vals = np.array([thermos.calc_saturation_vapor_pressure_over_water(float(t)) for t in temps])

    # Check strictly increasing
    assert np.all(np.diff(e_vals) > 0.0), f"Not monotonic: {e_vals}"


def test_get_saturation_vapor_pressure_switches_phase():
    """get_saturation_vapor_pressure should route to ice for T<=0 and water for T>0."""
    t_ice = -5.0
    t_water = 5.0

    e_ice_direct = thermos.calc_saturation_vapor_pressure_over_ice(t_ice)
    e_water_direct = thermos.calc_saturation_vapor_pressure_over_water(t_water)

    assert math.isclose(
        thermos.get_saturation_vapor_pressure(t_ice),
        e_ice_direct,
        rel_tol=1e-10,
        abs_tol=1e-10,
    )

    assert math.isclose(
        thermos.get_saturation_vapor_pressure(t_water),
        e_water_direct,
        rel_tol=1e-10,
        abs_tol=1e-10,
    )


# ---------------------------------------------------------------------------
# calc_ambient_vapor_pressure
# ---------------------------------------------------------------------------

def test_calc_ambient_vapor_pressure_consistent_with_sat_over_water():
    """calc_ambient_vapor_pressure should just be RH * e_sat_over_water."""
    t_a = 10.0  # °C
    rh = 40.0   # %

    e_sat = thermos.calc_saturation_vapor_pressure_over_water(t_a)
    expected = (rh / 100.0) * e_sat

    got = thermos.calc_ambient_vapor_pressure(t_a, rh)

    assert math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# calc_vapor_pressure_from_mixing_ratio
# ---------------------------------------------------------------------------

def test_calc_vapor_pressure_from_mixing_ratio_inverts_mixing_ratio_formula():
    """
    If we compute q from e and p using the standard formula,
    calc_vapor_pressure_from_mixing_ratio(q, p) should return ~e.
    """
    p = 80000.0  # Pa
    e_true = 1000.0  # Pa (arbitrary but small compared to p)
    EPS = thermos.EPSILON_Q

    # Forward relation: q = EPS * e / (p - e)
    q = EPS * e_true / (p - e_true)

    e_back = thermos.calc_vapor_pressure_from_mixing_ratio(q, p)

    assert math.isclose(e_back, e_true, rel_tol=1e-10, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# calc_relative_humidity_from_q2
# ---------------------------------------------------------------------------

def test_calc_relative_humidity_from_q2_recovers_known_rh_above_freezing():
    """
    Construct a self-consistent (T, RH, p, q) tuple and check that we recover RH.

    Steps:
      1) Choose T, RH, p.
      2) Compute e_sat over water at T.
      3) Compute e = (RH/100)*e_sat.
      4) Convert e,p -> q using the standard mixing-ratio formula.
      5) Feed (T, q, p) into calc_relative_humidity_from_q2 and check we get RH back.
    """
    t_a = 5.0       # °C
    rh_true = 65.0  # %
    p = 80000.0     # Pa

    # 2) saturation vapor pressure over water at t_a (Magnus over water)
    e_sat = thermos.calc_saturation_vapor_pressure_over_water(t_a)

    # 3) actual vapor pressure from RH
    e = (rh_true / 100.0) * e_sat

    # 4) mixing ratio from (e, p)
    EPS = thermos.EPSILON_Q
    q = EPS * e / (p - e)

    # 5) recover RH from (T, q, p)
    rh_got = thermos.calc_relative_humidity_from_q2(t_a, q, p)

    assert math.isfinite(rh_got)
    assert np.isclose(rh_got, rh_true, rtol=1e-4, atol=1e-3)


def test_calc_relative_humidity_from_q2_recovers_known_rh_below_freezing():
    """
    Same as above but for subfreezing T.  Here we still use e_sat_over_water
    for RH (Harder’s choice), not over ice.
    """
    t_a = -5.0      # °C
    rh_true = 45.0  # %
    p = 90000.0     # Pa

    e_sat = thermos.calc_saturation_vapor_pressure_over_water(t_a)
    e = (rh_true / 100.0) * e_sat

    EPS = thermos.EPSILON_Q
    q = EPS * e / (p - e)

    rh_got = thermos.calc_relative_humidity_from_q2(t_a, q, p)

    assert math.isfinite(rh_got)
    assert np.isclose(rh_got, rh_true, rtol=1e-4, atol=1e-3)


def test_calc_relative_humidity_from_q2_clips_above_100():
    """
    If we give a very large q that implies e > e_sat, RH should be clipped at 100%.
    """
    t_a = 0.0   # °C
    p = 80000.0 # Pa

    e_sat = thermos.calc_saturation_vapor_pressure_over_water(t_a)

    # Make e artificially 1.5 * e_sat and compute q from that.
    e = 1.5 * e_sat
    EPS = thermos.EPSILON_Q
    q = EPS * e / (p - e)

    rh_got = thermos.calc_relative_humidity_from_q2(t_a, q, p)

    assert 0.0 <= rh_got <= 100.0
    assert math.isclose(rh_got, 100.0, rel_tol=1e-6, abs_tol=1e-6)

# ---------------------------------------------------------------------------
# calc_diffusivity
# ---------------------------------------------------------------------------

def test_calc_diffusivity_positive_and_monotonic():
    """Diffusivity should be positive and increase with temperature."""
    temps = np.array([-20.0, 0.0, 10.0, 20.0, 30.0])
    d_vals = np.array([thermos.calc_diffusivity(float(t)) for t in temps])

    # All positive
    assert np.all(d_vals > 0.0), f"Non-positive diffusivity values: {d_vals}"

    # Strictly increasing with T
    assert np.all(np.diff(d_vals) > 0.0), f"Not monotonic in T: {d_vals}"


def test_calc_diffusivity_known_value_at_0C():
    """
    At 0 °C, the formula reduces to:
        d = 2.06e-5 * (273.15 / 273.15) ** 1.75 = 2.06e-5 m^2 s^-1
    """
    d0 = thermos.calc_diffusivity(0.0)
    assert math.isclose(d0, 2.06e-5, rel_tol=1e-6, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# calc_water_vapor_density
# ---------------------------------------------------------------------------

def test_calc_water_vapor_density_matches_ideal_gas_law():
    """
    calc_water_vapor_density should be consistent with:
        rho = (M_W * e) / (R * T_k)
    using the same ambient vapor pressure helper.
    """
    t_a = 10.0   # °C
    rh = 60.0    # %
    e = thermos.calc_ambient_vapor_pressure(t_a, rh)
    t_k = t_a + 273.15

    expected = (thermos.M_W * e) / (thermos.R * t_k)
    got = thermos.calc_water_vapor_density(t_a, rh)

    assert math.isfinite(got)
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_calc_water_vapor_density_zero_at_zero_rh():
    """At RH = 0%, water vapor density should be ~0."""
    t_a = 5.0
    rh = 0.0
    rho = thermos.calc_water_vapor_density(t_a, rh)

    assert rho >= 0.0
    # Allow tiny numerical noise around zero
    assert math.isclose(rho, 0.0, rel_tol=1e-12, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# calc_saturation_vapor_density
# ---------------------------------------------------------------------------

def test_calc_saturation_vapor_density_matches_sat_pressure_and_ideal_gas():
    """
    calc_saturation_vapor_density should be consistent with:
        rho_sat = (M_W * e_sat) / (R * T_k)
    where e_sat is computed via get_saturation_vapor_pressure.
    """
    for t_a in [-10.0, 0.0, 5.0, 20.0]:
        e_sat = thermos.get_saturation_vapor_pressure(t_a)
        t_k = t_a + 273.15
        expected = (thermos.M_W * e_sat) / (thermos.R * t_k)

        got = thermos.calc_saturation_vapor_density(t_a)

        assert math.isfinite(got)
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_calc_saturation_vapor_density_monotonic_with_temperature():
    """Saturation vapor density should increase with temperature."""
    temps = np.array([-10.0, 0.0, 5.0, 10.0])
    rho_vals = np.array([thermos.calc_saturation_vapor_density(float(t)) for t in temps])

    assert np.all(np.diff(rho_vals) > 0.0), f"Not monotonic in T: {rho_vals}"


# ---------------------------------------------------------------------------
# calc_thermal_conductivity
# ---------------------------------------------------------------------------

def test_calc_thermal_conductivity_linear_in_temperature():
    """
    lambda_t = 0.000063 * T + 0.00673
    Check linearity and positivity.
    """
    t1, t2 = 0.0, 20.0
    lam1 = thermos.calc_thermal_conductivity(t1)
    lam2 = thermos.calc_thermal_conductivity(t2)

    # Positivity
    assert lam1 > 0.0
    assert lam2 > 0.0

    # Expected difference from the linear formula
    expected_diff = 0.000063 * (t2 - t1)
    got_diff = lam2 - lam1

    assert math.isclose(got_diff, expected_diff, rel_tol=1e-12, abs_tol=1e-12)


def test_calc_thermal_conductivity_monotonic():
    """Thermal conductivity should increase with T over a typical range."""
    temps = np.array([-20.0, 0.0, 10.0, 20.0, 30.0])
    lam_vals = np.array([thermos.calc_thermal_conductivity(float(t)) for t in temps])

    assert np.all(np.diff(lam_vals) > 0.0), f"Not monotonic in T: {lam_vals}"


# ---------------------------------------------------------------------------
# get_latent_heat / calc_latent_heat_sublimation / calc_latent_heat_vaporization
# ---------------------------------------------------------------------------

def test_get_latent_heat_dispatches_correctly():
    """get_latent_heat should call sublimation for T < 0 and vaporization for T >= 0."""
    t_neg = -5.0
    t_pos = 5.0
    t_zero = 0.0

    l_neg_direct = thermos.calc_latent_heat_sublimation(t_neg)
    l_pos_direct = thermos.calc_latent_heat_vaporization(t_pos)
    l_zero_direct = thermos.calc_latent_heat_vaporization(t_zero)

    assert math.isclose(
        thermos.get_latent_heat(t_neg), l_neg_direct, rel_tol=1e-12, abs_tol=1e-12
    )
    assert math.isclose(
        thermos.get_latent_heat(t_pos), l_pos_direct, rel_tol=1e-12, abs_tol=1e-12
    )
    assert math.isclose(
        thermos.get_latent_heat(t_zero), l_zero_direct, rel_tol=1e-12, abs_tol=1e-12
    )


def test_latent_heat_sublimation_decreases_with_temperature():
    """Latent heat of sublimation should decrease as temperature increases (over a cold range)."""
    temps = np.array([-30.0, -20.0, -10.0, 0.0])
    l_vals = np.array([thermos.calc_latent_heat_sublimation(float(t)) for t in temps])

    # As T goes up, L should go down -> differences should be negative
    diffs = np.diff(l_vals)
    assert np.all(diffs < 0.0), f"Latent heat of sublimation not decreasing: {l_vals}"


def test_latent_heat_vaporization_decreases_with_temperature():
    """Latent heat of vaporization should decrease with temperature."""
    temps = np.array([0.0, 10.0, 20.0, 30.0])
    l_vals = np.array([thermos.calc_latent_heat_vaporization(float(t)) for t in temps])

    diffs = np.diff(l_vals)
    assert np.all(diffs < 0.0), f"Latent heat of vaporization not decreasing: {l_vals}"


def test_latent_heat_vaporization_known_value_at_0C():
    """
    At 0 °C:
        L_v = 1000 * (2501 - 2.361 * 0) = 2.501e6 J kg^-1
    """
    l0 = thermos.calc_latent_heat_vaporization(0.0)
    assert math.isclose(l0, 2.501e6, rel_tol=1e-6, abs_tol=1e-3)
