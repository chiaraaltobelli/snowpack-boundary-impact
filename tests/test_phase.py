# test_phase.py
import numpy as np
import pytest
import phase 

# -----------------------------------------------------------------------------
# calc_rainfall_fraction
# -----------------------------------------------------------------------------

def test_calc_rainfall_fraction_known_values():
    # Known parameters
    b, c = 2.0, 0.2

    # A few temps
    Ti = np.array([-5.0, 0.0, 5.0])

    # Expected via Eq. 6: f = 1 / (1 + b * c**Ti)
    expected = 1.0 / (1.0 + b * (c ** Ti))

    # Evaluate one-by-one to exercise scalar path
    got = np.array([phase.calc_rainfall_fraction(float(t), b, c) for t in Ti])

    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


# -----------------------------------------------------------------------------
# fit_bc
# -----------------------------------------------------------------------------

def test_fit_bc_recovers_bc_without_noise():
    # Synthetic truth
    b_true, c_true = 2.0, 0.2

    # Temperatures spanning a reasonable range
    Ti = np.linspace(-10, 10, 201)

    # Perfect observations from the model
    fr = 1.0 / (1.0 + b_true * (c_true ** Ti))

    # Uniform weights (e.g., precip amount)
    w = np.ones_like(Ti)

    # Wrap as xarray.DataArray-like inputs are expected; but function accepts xr.DataArray or ndarray values via .values usage.
    # Here we can pass NumPy arrays directly since fit_bc converts to np arrays internally.
    b_hat, c_hat = phase.fit_bc(Ti, fr, w)

    assert np.isclose(b_hat, b_true, rtol=1e-3, atol=1e-3)
    assert np.isclose(c_hat, c_true, rtol=1e-3, atol=1e-3)


def test_fit_bc_handles_edge_fr_clipping():
    b_true, c_true = 1.5, 0.3
    Ti = np.array([-10, -5, 0, 5, 10], dtype=float)

    # Force edge values (0 and 1) that require clipping internally
    fr = np.array([0.0, 0.01, 0.5, 0.99, 1.0], dtype=float)
    w = np.ones_like(Ti)

    b_hat, c_hat = phase.fit_bc(Ti, fr, w)

    # Should still produce a valid fit (positive b, 0 < c < 1)
    assert b_hat > 0.0
    assert 0.0 < c_hat < 1.0


def test_fit_bc_all_zero_weights_falls_back_to_unweighted():
    b_true, c_true = 2.5, 0.25
    Ti = np.linspace(-8, 8, 161)
    fr = 1.0 / (1.0 + b_true * (c_true ** Ti))

    w_zero = np.zeros_like(Ti)
    b_hat, c_hat = phase.fit_bc(Ti, fr, w_zero)

    assert np.isclose(b_hat, b_true, rtol=1e-3, atol=1e-3)
    assert np.isclose(c_hat, c_true, rtol=1e-3, atol=1e-3)


# -----------------------------------------------------------------------------
# calc_t_i (psychrometric iteration)
# -----------------------------------------------------------------------------

def test_calc_t_i_converges_to_known_fixed_point(monkeypatch):
    """
    We replace the thermodynamic helpers with simple deterministic functions so that
    the fixed-point iteration has a closed-form solution we can check.

    t_i(new) = T_a + (D / lambda_t) * L * (rho_Ta - rho_sat(t_i))

    We choose constants:
      D = 1, lambda_t = 1, and L = 1 (but L will be called each iteration)
    and a linear saturation density:
      rho_sat(t_i) = a + b * t_i

    Then the fixed point solves:
      t* = T_a + (rho_Ta - (a + b * t*))
      (1 + b) t* = T_a + rho_Ta - a
      t* = (T_a + rho_Ta - a) / (1 + b)
    """
    T_a = 0.0
    rho_Ta_const = 1.0
    a = 0.5
    b = 0.1

    # Expected fixed-point:
    t_star = (T_a + rho_Ta_const - a) / (1.0 + b)

    # Monkeypatch helpers inside phase module:
    monkeypatch.setattr(phase, "calc_diffusivity", lambda ta: 1.0)
    monkeypatch.setattr(phase, "calc_thermal_conductivity", lambda ta: 1.0)

    # L should be evaluated at current t_i, but we'll just return a constant 1.0
    monkeypatch.setattr(phase, "get_latent_heat", lambda ti: 1.0)

    # Ambient vapor density should be independent of t_i and equal to rho_Ta_const
    monkeypatch.setattr(phase, "calc_water_vapor_density", lambda ta, rh: rho_Ta_const)

    # Saturation vapor density depends linearly on t_i
    monkeypatch.setattr(phase, "calc_saturation_vapor_density", lambda ti: a + b * ti)

    # RH doesn't matter for our patched functions; pass anything
    t_i_computed = phase.calc_t_i(T_a, r_h=50.0)

    assert np.isfinite(t_i_computed)
    assert np.isclose(t_i_computed, t_star, rtol=1e-6, atol=1e-6)
