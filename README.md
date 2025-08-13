# README

## Cold Rain and Snow: The Influence of Cloud Microphysical Parameterizations on Hydrologic Predictions in Mountain Landscapes
## Project Purpose and Overview

This code suite supports the study "Cold Rain and Snow: The Influence of Cloud Microphysical Parameterizations on Hydrologic Predictions in Mountain Landscapes" by implementing and comparing precipitation phase partitioning methods (PPM). It automates both WRF microphysics schemes and common temperature-based PPMs to split rain and snow, allowing quantification of how different parameterizations influence snow accumulation, ablation, and transition‑band processes in mountain watershed simulations.

Folder contents:
1. **`iterative-03-2000.ipynb`**
2. **`compare-03-2000.ipynb`**
3. **`plot_utils.py`**
4. **`app.py`** ***FUTURE***

---

## 1. `iterative-03-2000.ipynb`

**Purpose:**

* A multi-threshold driver notebook that iterates over user‑defined temperature‑thresholds (e.g., 0 °C–5 °C) and schemes (microphysics, linear, sigmoidal) to produce grouped figures for multiple analyses.

  * Computes microphysics, linear, and sigmoidal snow fractions
  * Generates grouped plots, including:
    * Snow total maps
    * Snow fraction maps
    * Difference maps
    * Hexbin scatter comparisons
    * Histograms of fraction distributions
    * Phase‐partition curves
    * Transition‑band precipitation analysis

**Configuration Section:**

```python
# ==========================
# CONFIG
# ==========================
D02_DIR = "/bsushare/leaf-shared/subset-output-wrf/vol04/wrf_out/wy_2000/d02/"
YEAR, MONTH = "2000", "03"
EPSILON = 1e-6

# Temperature-threshold definitions
THRESHOLD_SOURCES = [
    {"name": "McCabe and Wolock (1999)", "T_snow":  0.0, "T_rain": 5.0},
    {"name": "Dai (2008)",              "T_snow": -2.0, "T_rain": 4.0},
]

# Partitioning schemes: (label, function)
SCHEME_FUNCS = [
    ("Microphysics", lambda: compute_microphysics_snow_frac(
        D02_DIR, YEAR, MONTH,
        min_precip_month=MIN_PRECIP_MONTH,
        epsilon=EPSILON
    )),
    ("Linear",       lambda T_snow, T_rain: compute_linear_snow_frac(
        D02_DIR, YEAR, MONTH,
        min_precip_hr=MIN_PRECIP_HR,
        TSNOW_THRESHOLD=T_snow,
        TRAIN_THRESHOLD=T_rain,
        epsilon=EPSILON,
        min_precip_month=MIN_PRECIP_MONTH
    )),
    ("Sigmoidal",    lambda T_snow, T_rain: compute_sigmoidal_snow_frac(
        D02_DIR, YEAR, MONTH,
        min_precip_hr=MIN_PRECIP_HR,
        T_snow=T_snow,
        T_rain=T_rain,
        F_snow=F_SNOW_TARGET,
        F_rain=F_RAIN_TARGET,
        epsilon=EPSILON,
        min_precip_month=MIN_PRECIP_MONTH
    )),
]

MIN_PRECIP_MONTH = 1.0   # mm for monthly mask
MIN_PRECIP_HR    = 0.1   # mm/hr for hourly filter
F_SNOW_TARGET    = 0.95
F_RAIN_TARGET    = 0.05

CMAP_MAP, CMAP_FRAC, CMAP_DIFF, CMAP_HEX = "Blues", "viridis", "seismic_r", "plasma"
FRAC_MIN, FRAC_MAX = 0.0, 1.0
DIFF_MIN, DIFF_MAX = -0.5, 0.5
SNOW_MIN          = 0.0
HEX_GRIDSIZE, HEX_TRIM_PCT = 50, 99.0

```

**Usage:**

1. Install dependencies: `numpy`, `xarray`, `matplotlib`, `cartopy`, `streamlit` etc.
2. Open in Jupyter and run all cells.
3. Add new entries to `THRESHOLD_SOURCES` to include other temperature definitions.

---

## 2. `compare-03-2000.ipynb`

**Purpose:**

* A simpler driver notebook for viewing a single set of thresholds (e.g. −2 to 4 °C).
* Computes and displays:

  * Side‑by‑side snow total maps (microphysics vs PPM linear vs PPM sigmoidal).
  * Side‑by‑side snow fraction maps.
  * Combined difference map/hexbins figure.
  * Standalone histograms and phase‑curve overlay.
  * Precipitation‑weighted temperature histogram.

**Key Configuration:**

```python
# ==========================
# CONFIG
# ==========================
D02_DIR = "/bsushare/leaf-shared/subset-output-wrf/vol04/wrf_out/wy_2000/d02/"
YEAR, MONTH = "2000", "03"
EPSILON = 1e-6

# Masks
MIN_PRECIP_MONTH = 1.0   # mm for monthly fields
MIN_PRECIP_HR    = 0.1   # mm/hr for hourly filter in linear/sigmoid paths

# Linear thresholds
T_SNOW_LINEAR = -2.0
T_RAIN_LINEAR = 4.0

# Sigmoidal anchors
T_SNOW_SIG    = -2.0
T_RAIN_SIG    = 4.0
F_SNOW_TARGET = 0.95
F_RAIN_TARGET = 0.05

# Colormaps
CMAP_MAP  = "Blues"       # totals
CMAP_FRAC = "viridis"     # 0–1 fraction
CMAP_DIFF = "seismic_r"   # differences
CMAP_HEX  = "plasma"

# Fixed limits
FRAC_MIN, FRAC_MAX = 0.0, 1.0
DIFF_MIN, DIFF_MAX = -0.5, 0.5
SNOW_MIN           = 0.0   # SNOW_MAX will be set from data

# Hexbin
HEX_GRIDSIZE = 50
HEX_TRIM_PCT = 99.0
```

**Usage:**

1. Open the notebook and run.
2. Modify threshold constants to compare different transition bands.

---

## 3. `plot_utils.py`

**Purpose:**

* Core utilities for WRF I/O, snow‐fraction computation, and mapping routines.

**Main Features:**

* **Cached I/O** with `@lru_cache` for fast repeated NetCDF reads.
* **`compute_microphysics_snow_frac`**: uses WRF microphysics accumulators.
* **Phase functions**:

  * `snow_frac_linear` (piecewise-linear)
  * `snow_frac_sigmoid` (tanh)
* **Generic `_compute_phase_snow_frac`** for monthly accumulations using any phase function.
* **`compute_linear_snow_frac`** and **`compute_sigmoidal_snow_frac`** wrappers.
* **Plot helpers**:

  * `cartomap` for Cartopy pcolormesh with coastlines/borders.
  * `plot_panels` (in notebook) for dynamic multi‑panel plots.
  * `plot_hexbin` for scatter hexbin visualizations.

**Usage:**

* Imported by both notebooks to handle data loading, masking, computation, and plotting.

---
## 4. `app.py` *(FUTURE — Streamlit GUI)*

**Purpose:**

Will provide a web-based interface (via Streamlit) to run the precipitation-phase analysis without editing notebook code.

Planned features:
- Dataset directory (`D02_DIR`) selection
- Year & month controls
- Temperature thresholds (`T_snow`, `T_rain`)
- Target snow/liquid fractions (`F_snow`, `F_rain`)
- Scheme selection (microphysics, linear, sigmoidal)
- Plot type selection
- Inline rendering of plots in browser

**Planned Usage:**

1. Activate the conda environment:
   ```bash
   conda activate snowpack-model
2. Run:
   ```bash
   streamlit run app.py --server.port <PORT> --server.address 0.0.0.0
3. Use VS Code’s PORTS panel to forward <PORT> and open in your browser.
---

# Prerequisites

* Python 3.8+
* Packages: `numpy`, `xarray`, `matplotlib`, `cartopy`, `functools`, `typing`

# Directory Structure

```
./
├── iterative-03-2000.ipynb
├── compare-03-2000.ipynb
└── plot_utils.py
```
