import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_wrf_var(var_name, current_plot_file, year, month):
    filename = f"{var_name}_{year}-{month}.nc"
    path = os.path.join(current_plot_file, filename)
    try:
        ds = xr.open_dataset(path, chunks={"Time": 28, "level": 39}) # may need adjusting
        print(f"Loaded {filename} with Dask chunks")
        return ds
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

def fix_accum_reset(accum, threshold=100.0):
    # these operations will now build a lazy Dask graph
    d = accum.diff(dim="Time")
    d = xr.where(d < 0, d + threshold, d)
    return d.sum(dim="Time")

def plot_cartopy(lons, lats, data, title,
                 cmap="viridis", vmin=None, vmax=None, alpha=0.6):
    fig = plt.figure(figsize=(10,6))
    ax  = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(lons, lats, data,
                         transform=ccrs.PlateCarree(),
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="auto", alpha=alpha,
                         zorder=1)
    cb = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
    cb.set_label(title)

    feat_kw = dict(zorder=2, linewidth=0.8)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), **feat_kw)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"),
                   linestyle=":", **feat_kw)
    ax.add_feature(cfeature.STATES.with_scale("110m"),
                   edgecolor="black", **feat_kw)
    ax.add_feature(cfeature.RIVERS.with_scale("110m"),
                   edgecolor="blue", **feat_kw)
    ax.add_feature(cfeature.LAKES.with_scale("110m"),
                   facecolor="lightblue", edgecolor="blue", **feat_kw)

    # draw gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.set_extent([lons.min(), lons.max(),
                   lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())
    ax.set_title(title)
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Latitude (°N)")

    plt.tight_layout()
    plt.show()
