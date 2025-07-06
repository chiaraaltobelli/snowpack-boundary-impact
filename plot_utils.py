import os
import xarray as xr
import numpy as np
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === HELPER TO LOAD WRF VARIABLE ===
def load_wrf_var(var_name, base_dir, year, month):
    fn = f"{var_name}_{year}-{month}.nc"
    path = os.path.join(base_dir, fn)
    try:
        ds = xr.open_dataset(path)
        print(f"Loaded {fn}")
        return ds
    except FileNotFoundError:
        print(f"File not found: {fn}")
        return None

# === ACCUMULATION WITH ROLLOVER FIX ===
# def fix_accum_reset(accum, threshold=100.0):
#     d = accum.diff(dim='Time')
#     d = xr.where(d < 0, d + threshold, d)
#     return d.sum(dim='Time')

def fix_accum_reset(accum, threshold=100.0):
    t0 = time.time()
    d = accum.diff(dim="Time")
    print(f"- diff took {time.time()-t0:.1f}s")

    t1 = time.time()
    d2 = xr.where(d < 0, d + threshold, d)
    print(f"- where took {time.time()-t1:.1f}s")

    t2 = time.time()
    out = d2.sum(dim="Time")
    print(f"- sum took {time.time()-t2:.1f}s")

    return out    

# === HELPER TO PLOT DATA WITH GEOGRAPHIC FEATURES ===
#TODO make geographic features optional
def plot_cartopy(lons, lats, data, title,
                 cmap="viridis", vmin=None, vmax=None, alpha=0.6):
    fig = plt.figure(figsize=(10,6))
    ax  = plt.axes(projection=ccrs.PlateCarree())

    # === PLOT GRIDDED DATA ON GEOAXES ====
    mesh = ax.pcolormesh(
        lons, lats, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", alpha=alpha,
        zorder=1
    )
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label(title)

    # === ADD GEOGRPAHIC FEATURES ====
    #TODO - review lake and river display 
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

    # === ADD BLACK FRAME AROUND GEOAXES ===
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1)

    # === ADD GRIDLINES AND LABELS ===
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    # === ZOOM TO DOMAIN AND LABEL ===
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())
    ax.set_title(title)
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Latitude (°N)")

    plt.tight_layout()
    plt.show()
