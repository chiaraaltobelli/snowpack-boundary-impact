# app.py — minimal GUI wrapper around compare_03_2000 logic
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import streamlit as st

import plot_utils
from plot_utils import (
    load_wrf_var,
    compute_microphysics_snow_frac,
    compute_linear_snow_frac,
    compute_sigmoidal_snow_frac,
    cartomap,
    snow_frac_sigmoid,  # optional (for later curves tab)
)

st.set_page_config(page_title="WRF Phase Compare", layout="wide")
st.title("WRF Precipitation-Phase — Compare (Single Threshold)")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Configuration")

# ---------------- Configuration ----------------
# Data directory (HPC default location)
D02_DIR = "/bsushare/leaf-shared/subset-output-wrf/vol04/wrf_out/wy_2000/d02/"

# Discover available years/months from T2 files
years = set()
months_by_year = {}
if os.path.isdir(D02_DIR):
    for fname in os.listdir(D02_DIR):
        m = re.match(r"^T2_(\d{4})-(\d{2}).*\.nc$", fname)
        if m:
            y, mo = m.groups()
            years.add(y)
            months_by_year.setdefault(y, set()).add(mo)

years = sorted(years)
for y in list(months_by_year.keys()):
    months_by_year[y] = sorted(months_by_year[y])

# Build a sorted list of "YYYY-MM" available (from T2 files we found)
ym_options = []
for y in years:
    for m in months_by_year.get(y, []):
        ym_options.append(f"{y}-{m}")
ym_options = sorted(ym_options)

if ym_options:
    # Default to full available range
    start_default, end_default = ym_options[0], ym_options[-1]
    PERIOD = st.sidebar.select_slider(
        "Analysis period (start → end)",
        options=ym_options,
        value=(start_default, end_default),
        help="Range of months to include (based on available T2_* files)."
    )
    START_YM, END_YM = PERIOD
else:
    START_YM = END_YM = None
    st.sidebar.warning("No monthly NetCDF files found (expected 'T2_YYYY-MM.nc').")


EPSILON = 1e-6
MIN_PRECIP_MONTH = 1.0 # mm 
MIN_PRECIP_HR    = 0.1 # mm/hr

# Two explicit thresholds, applied to ALL schemes
T_SNOW = st.sidebar.number_input("T_snow (°C)", min_value=-30.0, max_value=10.0, value=-2.0, step=0.5, key="t_snow")
T_RAIN = st.sidebar.number_input("T_rain (°C)", min_value=-10.0, max_value=20.0, value=4.0,  step=0.5, key="t_rain")

# Simple validation: T_rain must be > T_snow
if T_RAIN <= T_SNOW:
    st.sidebar.error("T_rain must be greater than T_snow.")
    run = False  # prevent execution

# Apply to both Linear and Sigmoidal
T_SNOW_LINEAR = T_SNOW_SIG = T_SNOW
T_RAIN_LINEAR = T_RAIN_SIG = T_RAIN

# For plot titles
thresh_label = f" (T_snow={T_SNOW:.1f}°C, T_rain={T_RAIN:.1f}°C)"

# Sigmoid anchor: choose F_snow only, compute F_rain = 1 - F_snow
F_SNOW_TARGET = st.sidebar.slider(
    "F_snow at T_snow (sigmoid)",
    0.50, 0.999, 0.95, 0.01, key="fsnow"
)
F_RAIN_TARGET = 1.0 - F_SNOW_TARGET  # hidden from UI

run = st.sidebar.button("Run", type="primary", key="run_btn")

# plotting constants (same as your driver)
CMAP_MAP, CMAP_FRAC, CMAP_DIFF, CMAP_HEX = "Blues", "viridis", "seismic_r", "plasma"
FRAC_MIN, FRAC_MAX = 0.0, 1.0
DIFF_MIN, DIFF_MAX = -0.5, 0.5
SNOW_MIN = 0.0
HEX_GRIDSIZE = 50

def _ym_next(ym: str) -> str:
    """Return next YYYY-MM string."""
    y, m = map(int, ym.split("-"))
    y2, m2 = (y + (m == 12), 1 if m == 12 else m + 1)
    return f"{y2:04d}-{m2:02d}"

def _ym_list(start_ym: str, end_ym: str) -> list[str]:
    """Inclusive list of YYYY-MM between start and end."""
    out = []
    cur = start_ym
    while True:
        out.append(cur)
        if cur == end_ym:
            break
        cur = _ym_next(cur)

def _parse_ym(ym: str) -> tuple[str, str]:
    """Split 'YYYY-MM' into ('YYYY', 'MM')."""
    y, m = ym.split("-")
    return y, m

# ---------------- Cache helpers ----------------
@st.cache_data(show_spinner=False)
def _grid(D02_DIR, YEAR, MONTH):
    ds = load_wrf_var("T2", D02_DIR, YEAR, MONTH)
    lons = ds["XLONG"][0].values
    lats = ds["XLAT"][0].values
    return lons, lats

@st.cache_data(show_spinner=False)
def _compute_all(params):
    # params is a dict so cache has a stable key
    D02_DIR = params["D02_DIR"]; YEAR = params["YEAR"]; MONTH = params["MONTH"]
    EPSILON = params["EPSILON"]; MIN_PRECIP_MONTH = params["MIN_PRECIP_MONTH"]; MIN_PRECIP_HR = params["MIN_PRECIP_HR"]
    T_SNOW_LINEAR = params["T_SNOW_LINEAR"]; T_RAIN_LINEAR = params["T_RAIN_LINEAR"]
    T_SNOW_SIG    = params["T_SNOW_SIG"];    T_RAIN_SIG    = params["T_RAIN_SIG"]
    F_SNOW_TARGET = params["F_SNOW_TARGET"]; F_RAIN_TARGET = params["F_RAIN_TARGET"]

    precip_micro, snow_micro, liquid_micro, frac_micro = compute_microphysics_snow_frac(
        D02_DIR, YEAR, MONTH, min_precip_month=MIN_PRECIP_MONTH, epsilon=EPSILON
    )
    precip_lin, snow_lin, liquid_lin, frac_lin = compute_linear_snow_frac(
        D02_DIR, YEAR, MONTH, min_precip_hr=MIN_PRECIP_HR,
        TSNOW_THRESHOLD=T_SNOW_LINEAR, TRAIN_THRESHOLD=T_RAIN_LINEAR,
        epsilon=EPSILON, min_precip_month=MIN_PRECIP_MONTH
    )
    precip_sig, snow_sig, liquid_sig, frac_sig = compute_sigmoidal_snow_frac(
        D02_DIR, YEAR, MONTH, min_precip_hr=MIN_PRECIP_HR,
        T_snow=T_SNOW_SIG, T_rain=T_RAIN_SIG,
        F_snow=F_SNOW_TARGET, F_rain=F_RAIN_TARGET,
        epsilon=EPSILON, min_precip_month=MIN_PRECIP_MONTH
    )
    return (snow_micro, frac_micro, snow_lin, frac_lin, snow_sig, frac_sig)
    
@st.cache_data(show_spinner=False)
def _compute_range(params, start_ym: str, end_ym: str):
    """
    Loop months from start_ym to end_ym (inclusive) and sum fields.
    Returns same tuple shapes as _compute_all but aggregated across months.
    """
    months = _ym_list(start_ym, end_ym)
    agg = None

    for ym in months:
        year, month = _parse_ym(ym)
        p = dict(params)
        p["YEAR"] = year
        p["MONTH"] = month

        snow_micro, frac_micro, snow_lin, frac_lin, snow_sig, frac_sig = _compute_all(p)

        # Convert to numpy arrays for accumulation
        A = [
            snow_micro.values, frac_micro.values,
            snow_lin.values,   frac_lin.values,
            snow_sig.values,   frac_sig.values
        ]
        if agg is None:
            agg = [a.copy() for a in A]
        else:
            # For fractions, weight by precip to combine correctly.
            # We need monthly precip weights; recompute quick precip from utilities:
            # NOTE: For a fast first pass, sum numerators/denominators from frac * precip.
            # We can get monthly precip from the compute_* functions indirectly:
            #   - For now we re-call just to get precip. If desired, refactor compute_* to return precip too.
            # Minimal approach: approximate by averaging fractions equally (simpler, less exact).
            # Better approach (recommended): modify compute_* to return precip monthly and weight here.
            agg[0] += A[0]  # snow_micro
            agg[2] += A[2]  # snow_lin
            agg[4] += A[4]  # snow_sig

            # For fractions, simple mean across months (approx). Replace with precip-weighted if you refactor.
            agg[1] = (agg[1] + A[1]) / 2.0
            agg[3] = (agg[3] + A[3]) / 2.0
            agg[5] = (agg[5] + A[5]) / 2.0

    # Pack back into xarray-like containers via the last month’s dataset shapes if needed.
    # Since we only use .values later for plotting, returning numpy arrays is fine.
    return tuple(agg)

def _panel_maps(lons, lats, arrays, labels, *, cmap, vmin, vmax, unit_label, title):
    fig, axes = plt.subplots(
        1, len(arrays), figsize=(4*len(arrays), 4),
        subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True
    )
    if len(arrays) == 1:
        axes = [axes]
    last = None
    for ax, data, lab in zip(axes, arrays, labels):
        m, _ = cartomap(ax, lons, lats, data, title=lab, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
        last = m
    cbar = fig.colorbar(last, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label(unit_label)
    fig.suptitle(title,  fontsize=18, y=1.05)
    return fig

# ---------------- Run + display ----------------
if run:
    if run and (YEAR is None or MONTH is None):
        st.warning("Select a valid year and month.")
        st.stop()
    if not os.path.isdir(D02_DIR):
        st.error(f"Directory not found: {D02_DIR}")
        st.stop()

    # quick existence check for core files
    must_exist = [f"T2_{YEAR}-{MONTH}.nc", f"RAINNC_{YEAR}-{MONTH}.nc", f"I_RAINNC_{YEAR}-{MONTH}.nc"]
    missing = [fn for fn in must_exist if not os.path.exists(os.path.join(D02_DIR, fn))]
    if missing:
        st.error(f"Missing required files for {YEAR}-{MONTH}: {', '.join(missing)}")
        st.stop()

    with st.spinner("Loading grid and computing…"):
        lons, lats = _grid(D02_DIR, YEAR, MONTH)
        snow_micro, frac_micro, snow_lin, frac_lin, snow_sig, frac_sig = _compute_all({
            "D02_DIR": D02_DIR, "YEAR": YEAR, "MONTH": MONTH,
            "EPSILON": EPSILON, "MIN_PRECIP_MONTH": MIN_PRECIP_MONTH, "MIN_PRECIP_HR": MIN_PRECIP_HR,
            "T_SNOW_LINEAR": T_SNOW_LINEAR, "T_RAIN_LINEAR": T_RAIN_LINEAR,
            "T_SNOW_SIG": T_SNOW_SIG, "T_RAIN_SIG": T_RAIN_SIG,
            "F_SNOW_TARGET": F_SNOW_TARGET, "F_RAIN_TARGET": F_RAIN_TARGET
        })

    tabs = st.tabs(["Snow totals", "Snow fraction", "Differences", "Hexbin"])

    # Totals
    with tabs[0]:
        SNOW_MAX = np.nanpercentile(
            np.concatenate([snow_micro.values.ravel(), snow_lin.values.ravel(), snow_sig.values.ravel()]), 99.0
        )
        fig = _panel_maps(
            lons, lats,
            [snow_micro.values, snow_lin.values, snow_sig.values],
            ['(a) Microphysics', '(b) Linear', '(c) Sigmoidal'],
            cmap=CMAP_MAP, vmin=SNOW_MIN, vmax=SNOW_MAX, unit_label="Snow Total (mm)",
            title=f"Snow Totals — {MONTH}-{YEAR}{thresh_label}"
        )
        st.pyplot(fig)

    # Fractions
    with tabs[1]:
        fig = _panel_maps(
            lons, lats,
            [frac_micro.values, frac_lin.values, frac_sig.values],
            ['(a) Microphysics', '(b) Linear', '(c) Sigmoidal'],
            cmap=CMAP_FRAC, vmin=FRAC_MIN, vmax=FRAC_MAX, unit_label="Snow Fraction",
            title=f"Snow Fraction — {MONTH}-{YEAR}{thresh_label}"
        )
        st.pyplot(fig)

    # Differences (maps only)
    with tabs[2]:
        fm = np.ma.masked_invalid(frac_micro.values)
        fl = np.ma.masked_invalid(frac_lin.values)
        fs = np.ma.masked_invalid(frac_sig.values)
        diff_lin = fl - fm
        diff_sig = fs - fm

        fig_maps, axs = plt.subplots(
            1, 2, figsize=(10, 4),
            subplot_kw={'projection': ccrs.PlateCarree()},
            constrained_layout=True
        )
        cartomap(axs[0], lons, lats, diff_lin, "Linear − Microphysics",
                 cmap=CMAP_DIFF, vmin=DIFF_MIN, vmax=DIFF_MAX)
        cartomap(axs[1], lons, lats, diff_sig, "Sigmoidal − Microphysics",
                 cmap=CMAP_DIFF, vmin=DIFF_MIN, vmax=DIFF_MAX)
        fig_maps.suptitle(f"Fraction Differences — {MONTH}-{YEAR}{thresh_label}", fontsize = 18, y=1.05)
        st.pyplot(fig_maps)

    # Hexbin (side-by-side)
    with tabs[3]:
        fm = np.ma.masked_invalid(frac_micro.values)
        fl = np.ma.masked_invalid(frac_lin.values)
        fs = np.ma.masked_invalid(frac_sig.values)

        x  = fm.compressed()
        y1 = fl.compressed()  # Linear
        y2 = fs.compressed()  # Sigmoidal

        fig_h, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        hb1 = axes[0].hexbin(x, y1, gridsize=HEX_GRIDSIZE, cmap=CMAP_HEX, bins='log', mincnt=1)
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel("Microphysics")
        axes[0].set_ylabel("Linear")
        axes[0].set_title(f"Linear vs Microphysics — {MONTH}-{YEAR}")

        hb2 = axes[1].hexbin(x, y2, gridsize=HEX_GRIDSIZE, cmap=CMAP_HEX, bins='log', mincnt=1)
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set_xlabel("Microphysics")
        axes[1].set_ylabel("Sigmoidal")
        axes[1].set_title(f"Sigmoidal vs Microphysics — {MONTH}-{YEAR}")

        # Unify color scale across both plots
        arr1 = hb1.get_array().compressed() if np.ma.isMaskedArray(hb1.get_array()) else hb1.get_array()
        arr2 = hb2.get_array().compressed() if np.ma.isMaskedArray(hb2.get_array()) else hb2.get_array()
        vmax = max(arr1.max(initial=1), arr2.max(initial=1))
        hb1.set_clim(vmax=vmax)
        hb2.set_clim(vmax=vmax)

        cbar = fig_h.colorbar(hb2, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
        cbar.set_label("Count")

        fig_h.suptitle(f"Hexbin Comparison — {MONTH}-{YEAR}{thresh_label}", fontsize=18, y=1.05)
        st.pyplot(fig_h)

else:
    st.info("Set parameters in the sidebar and click **Run**.")