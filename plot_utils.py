import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_wrf_var(var_name: str, current_plot_file: str, year: str, month: str) -> xr.Dataset:
    """
    Load a WRF NetCDF file as an xarray.Dataset with Dask chunking.

    Parameters:
        var_name: Base name of the WRF file (e.g. "RAINNC" or "T2").
        current_plot_file: Directory containing the NetCDF files.
        year: Four-digit year string, e.g. "2000".
        month: Two-digit month string, e.g. "03".

    Returns:
        An xarray.Dataset if found, or None if missing.
    """
    filename = f"{var_name}_{year}-{month}.nc"
    path = os.path.join(current_plot_file, filename)
    try:
        ds = xr.open_dataset(path, chunks={"Time": 28, "level": 39})
        print(f"Loaded {filename} with Dask chunks")
        return ds
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

def compute_microphysics_snow_frac(current_plot_file: str,
                                  year: str,
                                  month: str,
                                  min_precip: float = 1.0,
                                  epsilon: float = 1e-6):
    """
    Compute monthly precipitation, snow, liquid and snow-fraction using WRF microphysics.

    Steps:
      1. Load RAINNC, SNOWNC, HAILNC, GRAUPELNC, I_RAINNC.
      2. Correct rainfall rollover: rain2d = I_RAINNC*100 + RAINNC.
      3. Time-integrate each field to monthly totals.
      4. liquid = precip - (snow + hail + graupel).
      5. snow_frac = snow / (precip + epsilon).
      6. Mask where precip < min_precip or fraction invalid.

    Returns:
        precip, snow, liquid, snow_fraction (all xarray.DataArray)
    """
    ds_rain  = load_wrf_var("RAINNC",   current_plot_file, year, month)
    ds_snow  = load_wrf_var("SNOWNC",   current_plot_file, year, month)
    ds_hail  = load_wrf_var("HAILNC",   current_plot_file, year, month)
    ds_graup = load_wrf_var("GRAUPELNC",current_plot_file, year, month)
    ds_cnt   = load_wrf_var("I_RAINNC", current_plot_file, year, month)

    rain2d = ds_cnt["I_RAINNC"] * 100 + ds_rain["RAINNC"]
    precip = rain2d.diff("Time", label="upper").fillna(0).sum("Time")
    snow   = ds_snow["SNOWNC"].diff("Time").clip(min=0).sum("Time")
    hail   = ds_hail["HAILNC"].diff("Time").clip(min=0).sum("Time")
    graup  = ds_graup["GRAUPELNC"].diff("Time").clip(min=0).sum("Time")
    liquid = precip - (snow + hail + graup)
    frac   = snow / (precip + epsilon)

    mask = (precip < min_precip) | (~np.isfinite(frac))
    return (precip.where(~mask),
            snow.where(~mask),
            liquid.where(~mask),
            frac.where(~mask))

def compute_linear_snow_frac(current_plot_file: str,
                             year: str,
                             month: str,
                             min_precip_hr: float = 0.1,
                             TSNOW_THRESHOLD: float = 0.0,
                             TRAIN_THRESHOLD: float = 5.0,
                             epsilon: float = 1e-6):
    """
    Compute monthly precipitation, snow, liquid and snow-fraction using a linear 0–5 °C transition.

    Steps:
      1. Load T2, I_RAINNC, RAINNC.
      2. Correct rollover, diff→fillna to get hourly precip (mm/hr).
      3. Mask hourly precip < min_precip_hr.
      4. Convert T2→°C and transpose to match precip dims.
      5. Define snow_frac_lin: 1 if T≤TSNOW, 0 if T≥TRAIN, linear in between.
      6. Multiply precip by fraction and sum over Time.
      7. frac_map = snow_lin / (precip_sum + epsilon).
      8. Mask where precip_sum < min_precip_hr or frac invalid.

    Returns:
        precip_sum, snow_lin, liquid_lin, frac_map (all xarray.DataArray)
    """
    ds_t2   = load_wrf_var("T2",     current_plot_file, year, month)
    ds_cnt  = load_wrf_var("I_RAINNC",current_plot_file, year, month)
    ds_rain = load_wrf_var("RAINNC", current_plot_file, year, month)

    rain2d = ds_cnt["I_RAINNC"] * 100 + ds_rain["RAINNC"]
    raw    = rain2d.diff("Time", label="upper").fillna(0)
    zero   = xr.zeros_like(raw.isel(Time=0))
    precip_hr = xr.concat([zero, raw], dim="Time") \
                 .transpose("Time","south_north","west_east")
    precip_hr = precip_hr.where(precip_hr >= min_precip_hr, 0.0)

    t2c = (ds_t2["T2"] - 273.15) \
           .transpose("Time","south_north","west_east")
    span = TRAIN_THRESHOLD - TSNOW_THRESHOLD
    snow_frac_lin = xr.where(
        t2c <= TSNOW_THRESHOLD, 1.0,
        xr.where(
            t2c >= TRAIN_THRESHOLD, 0.0,
            (TRAIN_THRESHOLD - t2c) / span
        )
    )

    snow_lin   = (precip_hr * snow_frac_lin).sum("Time")
    liquid_lin = (precip_hr * (1 - snow_frac_lin)).sum("Time")
    precip_sum = precip_hr.sum("Time")
    frac_map   = snow_lin / (precip_sum + epsilon)

    mask = (precip_sum < min_precip_hr) | (~np.isfinite(frac_map))
    return (precip_sum.where(~mask),
            snow_lin.where(~mask),
            liquid_lin.where(~mask),
            frac_map.where(~mask))

def plot_cartopy(lons, lats, data, title,
                 cmap="viridis", vmin=None, vmax=None,
                 alpha=0.6, resolution="110m"):
    """
    Plot a 2D field on a map via Cartopy.

    Parameters:
      lons, lats: 2D lon/lat arrays.
      data: 2D numpy or masked array.
      title: colorbar label & plot title.
      cmap, vmin, vmax, alpha: passed to pcolormesh.
      resolution: '110m', '50m', etc. for features.
    """
    fig = plt.figure(figsize=(10,6))
    ax  = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(lons, lats, data,
                         transform=ccrs.PlateCarree(),
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="auto", alpha=alpha, zorder=1)
    cb = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
    cb.set_label(title)

    feat_kw = dict(zorder=2, linewidth=0.8)
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), **feat_kw)
    ax.add_feature(cfeature.BORDERS.with_scale(resolution), linestyle=":", **feat_kw)
    ax.add_feature(cfeature.STATES.with_scale(resolution), edgecolor="black", **feat_kw)
    ax.add_feature(cfeature.RIVERS.with_scale(resolution), edgecolor="blue", **feat_kw)
    ax.add_feature(cfeature.LAKES.with_scale(resolution),
                   facecolor="lightblue", edgecolor="blue", **feat_kw)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.set_extent([lons.min(), lons.max(),
                   lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())
    ax.set_title(title)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    plt.tight_layout()
    plt.show()

def plot_hexbin(x, y,
                gridsize: int = 50,
                cmap: str = "viridis",
                reduce_C_function=None,
                mincnt: int = 1,
                extent=None,
                xlabel: str = "X",
                ylabel: str = "Y",
                title: str = "",
                overlay_one_one: bool = False):
    """
    Create a hexbin comparison plot.

    Parameters:
        x, y: 1D numeric arrays.
        gridsize: hexbin resolution in x‑direction.
        cmap: colormap.
        reduce_C_function: aggregation function per bin (None for counts).
        mincnt: minimum points in a bin to display.
        extent: (xmin,xmax,ymin,ymax) limits.
        xlabel, ylabel, title: axis labels and title.
        overlay_one_one: if True, draw a 1:1 dashed line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(
        x, y,
        C=None if reduce_C_function is None else None,
        gridsize=gridsize,
        cmap=cmap,
        reduce_C_function=reduce_C_function,
        mincnt=mincnt,
        extent=extent,
        edgecolors="none"
    )
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
