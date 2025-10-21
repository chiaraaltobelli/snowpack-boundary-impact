# tests/test_snow_frac_linear.py
import numpy as np
import xarray as xr
import plot_utils as pu

def test_snow_frac_linear_numpy_bounds_and_middle():
    T_snow, T_rain = -2.0, 4.0
    temps = np.array([-10.0, -2.0, 1.0, 4.0, 10.0])
    frac = pu.snow_frac_linear(temps, T_snow, T_rain)

    # F=1 at or below T_snow, F=0 at or above T_rain, linear in between
    expected = np.array([1.0, 1.0, (T_rain - 1.0)/(T_rain - T_snow), 0.0, 0.0])
    assert np.allclose(frac, expected, rtol=0, atol=1e-12)

def test_snow_frac_linear_xarray_preserves_shape_and_dtype():
    T_snow, T_rain = 0.0, 5.0 # new thresholds
    da = xr.DataArray([[-1.0, 0.0, 2.5, 5.0, 7.0]]) # test teh xarray pathway
    out = pu.snow_frac_linear(da, T_snow, T_rain) # should return a data array

    # shape preserved, values clipped in [0,1], endpoints correct
    assert isinstance(out, xr.DataArray) # type preserved
    assert out.shape == da.shape # shape preserved
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0 # all values between 0-1
    assert out.values[0,0] == 1.0 # below T_snow
    assert out.values[0,1] == 1.0 # at T_snow
    assert out.values[0,3] == 0.0 # at T_rain
