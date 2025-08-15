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
st.title("Precipitation Phase Comparison Dashboard")

# ---------------- Configuration ----------------
st.sidebar.header("Configuration")

# Fixed data directory (HPC default location)
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

# Build a sorted list of "YYYY-MM" available
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
    # Ensure start <= end in case user drags in reverse
    if START_YM > END_YM:
        START_YM, END_YM = END_YM, START_YM
else:
    START_YM = END_YM = None
    st.sidebar.warning("No monthly NetCDF files found (expected 'T2_YYYY-MM.nc').")

# Backend defaults (kept out of UI)
EPSILON = 1e-6
MIN_PRECIP_MONTH = 1.0  # mm
MIN_PRECIP_HR    = 0.1  # mm/hr

# Two explicit thresholds, applied to ALL schemes
T_SNOW = st.sidebar.number_input("T_snow (°C)", min_value=-30.0, max_value=10.0, value=-2.0, step=0.5, key="t_snow")
T_RAIN = st.sidebar.number_input("T_rain (°C)", min_value=-10.0, max_value=20.0, value=4.0,  step=0.5, key="t_rain")

# Apply to both Linear and Sigmoidal
T_SNOW_LINEAR = T_SNOW_SIG = T_SNOW
T_RAIN_LINEAR = T_RAIN_SIG = T_RAIN

# For plot titles
thresh_label = f" (T_snow={T_SNOW:.1f}°C, T_rain={T_RAIN:.1f}°C)"

# Sigmoid anchor: choose F_snow only, compute F_rain = 1 - F_snow
F_SNOW_TARGET = st.sidebar.slider("F_snow at T_snow (sigmoid)", 0.50, 0.999, 0.95, 0.01, key="fsnow")
F_RAIN_TARGET = 1.0 - F_SNOW_TARGET  # hidden from UI

run = st.sidebar.button("Run", type="primary", key="run_btn")

# plotting constants
CMAP_MAP, CMAP_FRAC, CMAP_DIFF, CMAP_HEX = "Blues", "viridis", "seismic_r", "plasma"
FRAC_MIN, FRAC_MAX = 0.0, 1.0
DIFF_MIN, DIFF_MAX = -0.5, 0.5
SNOW_MIN = 0.0
HEX_GRIDSIZE = 50

# ---------------- Month helpers ----------------
def _ym_next(ym: str) -> str:
    """Return next YYYY-MM string."""
    y, m = map(int, ym.split("-"))
    y2, m2 = (y + (m == 12), 1 if m == 12 else m + 1)
    return f"{y2:04d}-{m2:02d}"

def _ym_list(start_ym: str, end_ym: str):
    """Inclusive list of YYYY-MM between start and end."""
    out = []
    cur = start_ym
    while True:
        out.append(cur)
        if cur == end_ym:
            break
        cur = _ym_next(cur)
    return out

def _parse_ym(ym: str):
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
    """
    Return monthly fields including precip so we can do precip-weighted aggregation:
    (precip_micro, snow_micro, liquid_micro, frac_micro,
     precip_lin,   snow_lin,   liquid_lin,   frac_lin,
     precip_sig,   snow_sig,   liquid_sig,   frac_sig)
    """
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
    return (
        precip_micro, snow_micro, liquid_micro, frac_micro,
        precip_lin,   snow_lin,   liquid_lin,   frac_lin,
        precip_sig,   snow_sig,   liquid_sig,   frac_sig
    )

@st.cache_data(show_spinner=False)
def _compute_range(params, start_ym: str, end_ym: str):
    """
    Loop months from start_ym to end_ym (inclusive) and aggregate:
      - Sum precip & snow for each scheme
      - Recompute fraction = snow_sum / (precip_sum + EPSILON)
    Returns the six arrays used by the UI:
      (snow_micro, frac_micro, snow_lin, frac_lin, snow_sig, frac_sig)
    """
    months = _ym_list(start_ym, end_ym)

    # Running sums
    pm_sum = pl_sum = ps_sum = None   # precip micro/lin/sig
    sm_sum = sl_sum = ss_sum = None   # snow   micro/lin/sig

    for ym in months:
        year, month = _parse_ym(ym)
        p = dict(params)
        p["YEAR"] = year
        p["MONTH"] = month

        (precip_micro, snow_micro, _liquid_micro, _frac_micro,
         precip_lin,   snow_lin,   _liquid_lin,   _frac_lin,
         precip_sig,   snow_sig,   _liquid_sig,   _frac_sig) = _compute_all(p)

        # Convert to numpy
        pm = np.asarray(precip_micro); sm = np.asarray(snow_micro)
        pl = np.asarray(precip_lin);   sl = np.asarray(snow_lin)
        ps = np.asarray(precip_sig);   ss = np.asarray(snow_sig)

        if pm_sum is None:
            pm_sum, sm_sum = pm.copy(), sm.copy()
            pl_sum, sl_sum = pl.copy(), sl.copy()
            ps_sum, ss_sum = ps.copy(), ss.copy()
        else:
            pm_sum += pm; sm_sum += sm
            pl_sum += pl; sl_sum += sl
            ps_sum += ps; ss_sum += ss

    # Recompute precip-weighted fractions
    EPS = params["EPSILON"]
    frac_micro = sm_sum / (pm_sum + EPS)
    frac_lin   = sl_sum / (pl_sum + EPS)
    frac_sig   = ss_sum / (ps_sum + EPS)

    # Return the same six arrays the rest of the app expects
    return (sm_sum, frac_micro, sl_sum, frac_lin, ss_sum, frac_sig)

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
    fig.suptitle(title, fontsize=18, y=1.05)
    return fig

# ---------------- Run + display ----------------
if run:
    # Hard stop on invalid thresholds
    if T_RAIN <= T_SNOW:
        st.error("T_rain must be greater than T_snow.")
        st.stop()

    if START_YM is None or END_YM is None:
        st.warning("Select a valid start and end month.")
        st.stop()
    if not os.path.isdir(D02_DIR):
        st.error(f"Directory not found: {D02_DIR}")
        st.stop()

    # Check that required files exist for all months in the range
    missing = []
    for ym in _ym_list(START_YM, END_YM):
        y, m = _parse_ym(ym)
        for fn in (f"T2_{y}-{m}.nc", f"RAINNC_{y}-{m}.nc", f"I_RAINNC_{y}-{m}.nc"):
            if not os.path.exists(os.path.join(D02_DIR, fn)):
                missing.append(fn)
    if missing:
        st.error("Missing required files in the selected range: " + ", ".join(sorted(set(missing))))
        st.stop()

    with st.spinner("Loading grid and computing…"):
        # Use the start month grid (assumed static grid)
        sy, sm = _parse_ym(START_YM)
        lons, lats = _grid(D02_DIR, sy, sm)

        params = {
            "D02_DIR": D02_DIR,
            "YEAR": sy, "MONTH": sm,  # placeholders; _compute_range overrides per month
            "EPSILON": EPSILON,
            "MIN_PRECIP_MONTH": MIN_PRECIP_MONTH,
            "MIN_PRECIP_HR": MIN_PRECIP_HR,
            "T_SNOW_LINEAR": T_SNOW_LINEAR, "T_RAIN_LINEAR": T_RAIN_LINEAR,
            "T_SNOW_SIG": T_SNOW_SIG,       "T_RAIN_SIG": T_RAIN_SIG,
            "F_SNOW_TARGET": F_SNOW_TARGET, "F_RAIN_TARGET": F_RAIN_TARGET
        }

        snow_micro, frac_micro, snow_lin, frac_lin, snow_sig, frac_sig = _compute_range(
            params, START_YM, END_YM
        )

    range_label = f"{START_YM} → {END_YM}"

    tabs = st.tabs(["Snow totals", "Snow fraction", "Differences", "Hexbin"])

    # Totals
    with tabs[0]:
        SNOW_MAX = np.nanpercentile(
            np.concatenate([np.asarray(snow_micro).ravel(),
                            np.asarray(snow_lin).ravel(),
                            np.asarray(snow_sig).ravel()]), 99.0
        )
        fig = _panel_maps(
            lons, lats,
            [np.asarray(snow_micro), np.asarray(snow_lin), np.asarray(snow_sig)],
            ['(a) Microphysics', '(b) Linear', '(c) Sigmoidal'],
            cmap=CMAP_MAP, vmin=SNOW_MIN, vmax=SNOW_MAX, unit_label="Snow Total (mm)",
            title=f"Snow Totals — {range_label}{thresh_label}, fontsize=18, y=1.05"
        )
        st.pyplot(fig)

    # Fractions
    with tabs[1]:
        fig = _panel_maps(
            lons, lats,
            [np.asarray(frac_micro), np.asarray(frac_lin), np.asarray(frac_sig)],
            ['(a) Microphysics', '(b) Linear', '(c) Sigmoidal'],
            cmap=CMAP_FRAC, vmin=FRAC_MIN, vmax=FRAC_MAX, unit_label="Snow Fraction",
            title=f"Snow Fraction — {range_label}{thresh_label}, fontsize=18, y=1.05"
        )
        st.pyplot(fig)

    # Differences (maps only)
    with tabs[2]:
        fm = np.ma.masked_invalid(np.asarray(frac_micro))
        fl = np.ma.masked_invalid(np.asarray(frac_lin))
        fs = np.ma.masked_invalid(np.asarray(frac_sig))
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
        fig_maps.suptitle(f"Fraction Differences — {range_label}{thresh_label}", fontsize=18, y=1.05)
        st.pyplot(fig_maps)

    # Hexbin (side-by-side)
    with tabs[3]:
        fm = np.ma.masked_invalid(np.asarray(frac_micro))
        fl = np.ma.masked_invalid(np.asarray(frac_lin))
        fs = np.ma.masked_invalid(np.asarray(frac_sig))

        x  = fm.compressed()
        y1 = fl.compressed()  # Linear
        y2 = fs.compressed()  # Sigmoidal

        fig_h, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        hb1 = axes[0].hexbin(x, y1, gridsize=HEX_GRIDSIZE, cmap=CMAP_HEX, bins='log', mincnt=1)
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel("Microphysics")
        axes[0].set_ylabel("Linear")
        axes[0].set_title(f"Linear vs Microphysics — {range_label}")

        hb2 = axes[1].hexbin(x, y2, gridsize=HEX_GRIDSIZE, cmap=CMAP_HEX, bins='log', mincnt=1)
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1].set_xlabel("Microphysics")
        axes[1].set_ylabel("Sigmoidal")
        axes[1].set_title(f"Sigmoidal vs Microphysics — {range_label}")

        # Unify color scale across both plots
        arr1 = hb1.get_array().compressed() if np.ma.isMaskedArray(hb1.get_array()) else hb1.get_array()
        arr2 = hb2.get_array().compressed() if np.ma.isMaskedArray(hb2.get_array()) else hb2.get_array()
        vmax = max(arr1.max(initial=1), arr2.max(initial=1))
        hb1.set_clim(vmax=vmax)
        hb2.set_clim(vmax=vmax)

        cbar = fig_h.colorbar(hb2, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
        cbar.set_label("Count")

        fig_h.suptitle(f"Hexbin Comparison — {range_label}{thresh_label}", fontsize=18, y=1.05)
        st.pyplot(fig_h)

else:
    st.info("Choose a period, set thresholds, and click **Run**.")
