"""Utility functions for WRF precipitation‑phase analysis and plotting.

Highlights
----------
- Cached NetCDF opens (no repeated disk I/O).
- Shared helpers for hourly precipitation and masking.
- One generic core to compute monthly snow/liquid/frac from any phase function.
- Linear and sigmoidal (tanh) phase functions provided.
- Both figure-creating and axis-aware Cartopy plotters.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------------------------------------------------------
# Data I/O
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _open_wrf(var_name: str, current_plot_file: str, year: str, month: str) -> xr.Dataset:
    """Open a monthly WRF NetCDF once and cache it."""
    filename = f"{var_name}_{year}-{month}.nc"
    path = os.path.join(current_plot_file, filename)
    return xr.open_dataset(path, chunks={"Time": 28, "level": 39})

def load_wrf_var(var_name: str, current_plot_file: str, year: str, month: str) -> xr.Dataset | None:
    """Cached loader with friendly printout and FileNotFound handling."""
    try:
        ds = _open_wrf(var_name, current_plot_file, year, month)
        print(f"Loaded {var_name}_{year}-{month}.nc (cached)")
        return ds
    except FileNotFoundError:
        print(f"File not found: {var_name}_{year}-{month}.nc")
        return None

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _hourly_precip(ds_cnt: xr.Dataset, ds_rain: xr.Dataset, min_precip_hr: float) -> xr.DataArray:
    """Hourly precipitation (mm), masked below a threshold."""
    rain2d = ds_cnt["I_RAINNC"] * 100 + ds_rain["RAINNC"]
    raw = rain2d.diff("Time", label="upper").fillna(0)
    zero = xr.zeros_like(raw.isel(Time=0))
    precip_hr = xr.concat([zero, raw], dim="Time").transpose("Time", "south_north", "west_east")
    return precip_hr.where(precip_hr >= min_precip_hr, 0.0)

def mask_and_pack(precip_sum: xr.DataArray,
                  snow: xr.DataArray,
                  liquid: xr.DataArray,
                  frac: xr.DataArray,
                  min_total_precip: float | None = None,
                  extra_mask: xr.DataArray | np.ndarray | None = None) -> tuple[xr.DataArray, ...]:
    """
    Apply a combined mask and return (precip_sum, snow, liquid, frac).

    Parameters
    ----------
    precip_sum, snow, liquid, frac : xarray.DataArray
        Monthly fields.
    min_total_precip : float or None
        If given, mask cells where precip_sum < this value (mm).
    extra_mask : bool DataArray/ndarray or None
        Additional mask (True == mask out) to OR with the default mask.

    Returns
    -------
    tuple[xarray.DataArray, ...]
        Masked (precip_sum, snow, liquid, frac).
    """
    mask = ~np.isfinite(frac)
    if min_total_precip is not None:
        mask |= (precip_sum < min_total_precip)
    if extra_mask is not None:
        mask |= extra_mask
    return (precip_sum.where(~mask),
            snow.where(~mask),
            liquid.where(~mask),
            frac.where(~mask))

# -----------------------------------------------------------------------------
# Microphysics-based fraction
# -----------------------------------------------------------------------------

def compute_microphysics_snow_frac(current_plot_file: str,
                                   year: str,
                                   month: str,
                                   min_precip_month: float = 1.0,
                                   epsilon: float = 1e-6):
    """Monthly totals using native WRF microphysics accumulators.

    Steps
    -----
    1. Load RAINNC, SNOWNC, HAILNC, GRAUPELNC, I_RAINNC.
    2. Correct rollover (rain2d = I_RAINNC*100 + RAINNC).
    3. Diff & sum over Time to get monthly totals.
    4. Compute liquid = precip - (snow+hail+graupel), and frac = snow/(precip+ε).
    5. Mask where precip < min_precip_month or frac non-finite.
    """
    ds_rain  = load_wrf_var("RAINNC",    current_plot_file, year, month)
    ds_snow  = load_wrf_var("SNOWNC",    current_plot_file, year, month)
    ds_hail  = load_wrf_var("HAILNC",    current_plot_file, year, month)
    ds_graup = load_wrf_var("GRAUPELNC", current_plot_file, year, month)
    ds_cnt   = load_wrf_var("I_RAINNC",  current_plot_file, year, month)

    rain2d = ds_cnt["I_RAINNC"] * 100 + ds_rain["RAINNC"]
    precip = rain2d.diff("Time", label="upper").fillna(0).sum("Time")
    snow   = ds_snow["SNOWNC"].diff("Time").clip(min=0).sum("Time")
    hail   = ds_hail["HAILNC"].diff("Time").clip(min=0).sum("Time")
    graup  = ds_graup["GRAUPELNC"].diff("Time").clip(min=0).sum("Time")

    liquid = precip - (snow + hail + graup)
    frac   = snow / (precip + epsilon)

    return mask_and_pack(precip, snow, liquid, frac, min_total_precip=min_precip_month)

# -----------------------------------------------------------------------------
# Phase functions (linear / sigmoid)
# -----------------------------------------------------------------------------

def snow_frac_linear(Tc: xr.DataArray | np.ndarray,
                     T_snow: float,
                     T_rain: float) -> xr.DataArray | np.ndarray:
    """Piecewise-linear phase fraction. F=1 for T<=T_snow, F=0 for T>=T_rain."""
    span = T_rain - T_snow
    return xr.where(Tc <= T_snow, 1.0,
           xr.where(Tc >= T_rain, 0.0,
                    (T_rain - Tc) / span))

def _solve_sigmoid_params(T_snow: float,
                          T_rain: float,
                          F_snow: float = 0.95,
                          F_rain: float = 0.05) -> tuple[float, float]:
    """Parameters (b, c) for F(T)=0.5*[1 - tanh(b*(T - c))] matching anchors."""
    if T_rain <= T_snow:
        raise ValueError("T_rain must be > T_snow")
    if not (0 < F_snow < 1 and 0 < F_rain < 1):
        raise ValueError("F_snow/F_rain must be in (0,1)")
    c = 0.5 * (T_snow + T_rain)
    z_s = 1 - 2 * F_snow
    z_r = 1 - 2 * F_rain
    b1 = np.arctanh(z_s) / (T_snow - c)
    b2 = np.arctanh(z_r) / (T_rain - c)
    b  = 0.5 * (b1 + b2)
    return b, c

def snow_frac_sigmoid(Tc: xr.DataArray | np.ndarray,
                      T_snow: float = -5.0,
                      T_rain: float = 5.0,
                      F_snow: float = 0.95,
                      F_rain: float = 0.05) -> xr.DataArray | np.ndarray:
    """Sigmoidal (tanh) snow fraction in [0,1]."""
    b, c = _solve_sigmoid_params(T_snow, T_rain, F_snow, F_rain)
    return np.clip(0.5 * (1 - np.tanh(b * (Tc - c))), 0.0, 1.0)

# -----------------------------------------------------------------------------
# Generic monthly phase calculator
# -----------------------------------------------------------------------------

def _compute_phase_snow_frac(current_plot_file: str,
                             year: str,
                             month: str,
                             min_precip_hr: float,
                             phase_func: Callable[[xr.DataArray], xr.DataArray],
                             epsilon: float = 1e-6,
                             min_precip_month: float | None = None):
    """Build monthly fields using a provided phase function."""
    ds_t2   = load_wrf_var("T2",       current_plot_file, year, month)
    ds_cnt  = load_wrf_var("I_RAINNC", current_plot_file, year, month)
    ds_rain = load_wrf_var("RAINNC",   current_plot_file, year, month)

    precip_hr = _hourly_precip(ds_cnt, ds_rain, min_precip_hr)
    T_c = (ds_t2["T2"] - 273.15).transpose("Time", "south_north", "west_east")

    frac_hr = phase_func(T_c).clip(min=0.0, max=1.0)

    snow        = (precip_hr * frac_hr).sum("Time")
    liquid      = (precip_hr * (1 - frac_hr)).sum("Time")
    precip_sum  = precip_hr.sum("Time")
    frac_map    = snow / (precip_sum + epsilon)

    month_thresh = min_precip_hr if min_precip_month is None else min_precip_month
    return mask_and_pack(precip_sum, snow, liquid, frac_map,
                         min_total_precip=month_thresh)

def compute_linear_snow_frac(current_plot_file: str,
                             year: str,
                             month: str,
                             min_precip_hr: float = 0.1,
                             TSNOW_THRESHOLD: float = 0.0,
                             TRAIN_THRESHOLD: float = 5.0,
                             epsilon: float = 1e-6,
                             min_precip_month: float | None = None):
    """Monthly fields using the piecewise-linear phase function."""
    return _compute_phase_snow_frac(
        current_plot_file, year, month, min_precip_hr,
        phase_func=lambda T: snow_frac_linear(T, TSNOW_THRESHOLD, TRAIN_THRESHOLD),
        epsilon=epsilon, min_precip_month=min_precip_month
    )

def compute_sigmoidal_snow_frac(current_plot_file: str,
                                year: str,
                                month: str,
                                min_precip_hr: float = 0.1,
                                T_snow: float = -5.0,
                                T_rain: float = 5.0,
                                F_snow: float = 0.95,
                                F_rain: float = 0.05,
                                epsilon: float = 1e-6,
                                min_precip_month: float | None = None):
    """Monthly fields using the tanh-based sigmoidal phase function."""
    return _compute_phase_snow_frac(
        current_plot_file, year, month, min_precip_hr,
        phase_func=lambda T: snow_frac_sigmoid(T, T_snow, T_rain, F_snow, F_rain),
        epsilon=epsilon, min_precip_month=min_precip_month
    )

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_cartopy(lons, lats, data, title,
                 cmap="viridis", vmin=None, vmax=None,
                 alpha=0.6, resolution="110m"):
    """Quick standalone Cartopy map (creates and shows a Figure)."""
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cartomap(ax, lons, lats, data, title, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.add_feature(cfeature.RIVERS.with_scale(resolution), edgecolor="blue", linewidth=0.5)
    ax.add_feature(cfeature.LAKES.with_scale(resolution),  facecolor="lightblue", edgecolor="blue", linewidth=0.5)
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())

    plt.tight_layout()
    plt.show()

def cartomap(ax, lons, lats, data, title,
             cmap="viridis", vmin=None, vmax=None, alpha=0.6,
             add_colorbar=True, cbar_label=None, cbar_kwargs=None):
    """
    Draw a Cartopy pcolormesh on an existing axes.

    Returns
    -------
    mesh : QuadMesh
    cb   : Colorbar or None
    """
    mesh = ax.pcolormesh(lons, lats, data,
                         transform=ccrs.PlateCarree(),
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="auto", alpha=alpha, zorder=1)
    cb = None
    if add_colorbar:
        ck = {} if cbar_kwargs is None else dict(cbar_kwargs)
        cb = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, **ck)
        cb.set_label(cbar_label or title)

    feat_kw = dict(zorder=2, linewidth=0.6)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), **feat_kw)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linestyle=":", **feat_kw)
    ax.add_feature(cfeature.STATES.with_scale("110m"), edgecolor="black", linewidth=0.5, zorder=2)

    ax.set_title(title)
    return mesh, cb

def plot_hexbin(x, y,
                gridsize: int = 50,
                cmap: str = "viridis",
                reduce_C_function=None,
                mincnt: int = 1,
                extent=None,
                xlabel: str = "X",
                ylabel: str = "Y",
                title: str = "",
                overlay_one_one: bool = False,
                log_counts: bool = False,
                trim_zeros_ones: bool = True,
                trim_pct: float | None = 99.5):
    """Hexbin scatter with optional log coloring and edge trimming.

    Returns
    -------
    fig, ax, hb : matplotlib Figure, Axes, PolyCollection
    """
    x = np.asarray(x); y = np.asarray(y)

    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    if trim_zeros_ones:
        edge = np.isclose(x, 0) | np.isclose(x, 1) | np.isclose(y, 0) | np.isclose(y, 1)
        x, y = x[~edge], y[~edge]

    fig, ax = plt.subplots(figsize=(8, 6))

    if log_counts and reduce_C_function is None:
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, bins='log',
                       mincnt=mincnt, extent=extent, edgecolors="none")
    else:
        norm = LogNorm(vmin=max(mincnt, 1)) if log_counts else None
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap,
                       reduce_C_function=reduce_C_function,
                       mincnt=mincnt, extent=extent, edgecolors="none",
                       norm=norm)

    if trim_pct is not None:
        arr = hb.get_array()
        if np.ma.isMaskedArray(arr):
            arr = arr.compressed()
        vmax = np.percentile(arr, trim_pct)
        hb.set_clim(vmax=vmax)

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count" if reduce_C_function is None else "Aggregated value")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if overlay_one_one:
        mn = min(ax.get_xlim()[0], ax.get_ylim()[0])
        mx = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([mn, mx], [mn, mx], "k--", zorder=2)

    plt.tight_layout()
    plt.show()
    return fig, ax, hb

__all__ = [
    "load_wrf_var",
    "compute_microphysics_snow_frac",
    "compute_linear_snow_frac",
    "compute_sigmoidal_snow_frac",
    "snow_frac_sigmoid", "snow_frac_linear",
    "plot_cartopy", "cartomap", "plot_hexbin",
    "mask_and_pack",
]
