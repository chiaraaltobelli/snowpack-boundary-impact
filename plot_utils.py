import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def load_wrf_var(var_name: str, current_plot_file: str, year: str, month: str) -> xr.Dataset:
    """
    Load a WRF NetCDF file by variable name, year, and month using xarray with Dask chunks.

    Parameters:
        var_name: The WRF variable prefix (e.g., "T2", "RAINNC").
        current_plot_file: Directory path containing the NetCDF files.
        year: Four-digit year string (e.g., "2000").
        month: Two-digit month string (e.g., "03").

    Returns:
        An xarray.Dataset with Dask chunking or None if the file is not found.
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


def plot_cartopy(
    lons: np.ndarray,
    lats: np.ndarray,
    data: np.ma.MaskedArray,
    title: str,
    cmap: str = "viridis",
    vmin: float = None,
    vmax: float = None,
    alpha: float = 0.6
) -> None:
    """
    Render a 2D field on a geographic map using Cartopy.

    Parameters:
        lons: 2D array of longitudes.
        lats: 2D array of latitudes.
        data: 2D masked array of field values to plot.
        title: Plot title (also used as colorbar label).
        cmap: Colormap name for the field.
        vmin: Minimum data value for color scaling.
        vmax: Maximum data value for color scaling.
        alpha: Transparency of the plotted field.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lons, lats, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", alpha=alpha, zorder=1
    )
    cb = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
    cb.set_label(title)

    feat_kw = dict(zorder=2, linewidth=0.8)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), **feat_kw)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linestyle=":", **feat_kw)
    ax.add_feature(cfeature.STATES.with_scale("110m"), edgecolor="black", **feat_kw)
    ax.add_feature(cfeature.RIVERS.with_scale("110m"), edgecolor="blue", **feat_kw)
    ax.add_feature(cfeature.LAKES.with_scale("110m"), facecolor="lightblue", edgecolor="blue", **feat_kw)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.set_extent([
        np.nanmin(lons), np.nanmax(lons),
        np.nanmin(lats), np.nanmax(lats)
    ], crs=ccrs.PlateCarree())

    ax.set_title(title)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    plt.tight_layout()
    plt.show()


def plot_hexbin(
    x: np.ndarray,
    y: np.ndarray,
    C: np.ndarray = None,
    gridsize: int = 50,
    cmap: str = "viridis",
    reduce_C_function: callable = np.size,
    mincnt: int = 1,
    extent: tuple = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    overlay_one_one: bool = False
) -> None:
    """
    Create a hexbin comparison between two fields, with optional 1:1 overlay.

    Parameters:
        x, y: 1D data arrays for the horizontal and vertical axes.
        C: Optional 1D array of values to aggregate in each hexbin.
        gridsize: Number of hexagons in the x-direction.
        cmap: Colormap for the hexbin counts or aggregated values.
        reduce_C_function: Function to aggregate C within each bin (e.g., np.mean).
        mincnt: Minimum count in a bin to be displayed.
        extent: (xmin, xmax, ymin, ymax) tuple for plotting limits.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Plot title.
        overlay_one_one: If True, draw a dashed 1:1 reference line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(
        x, y, C=C,
        gridsize=gridsize,
        cmap=cmap,
        reduce_C_function=reduce_C_function,
        mincnt=mincnt,
        extent=extent,
        edgecolors='none'
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count' if C is None else 'Aggregated value')

    if overlay_one_one:
        mn = np.nanmin([x.min(), y.min()])
        mx = np.nanmax([x.max(), y.max()])
        ax.plot([mn, mx], [mn, mx], '--', color='black', lw=1, label='1:1')
        ax.legend(loc='upper left')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
