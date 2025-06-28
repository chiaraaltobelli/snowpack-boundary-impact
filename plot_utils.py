import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

# Pre-download necessary Natural Earth shapefiles so Cartopy features will render
for resolution, category, name in [
    ('110m', 'physical', 'coastline'),
    ('110m', 'physical', 'rivers_lake_centerlines'),
    ('110m', 'physical', 'lakes'),
]:
    shpreader.natural_earth(resolution=resolution, category=category, name=name)


def plot_with_state_outline(lons, lats, data_array, title,
                            cmap="viridis", vmin=None, vmax=None,
                            alpha=0.6):
    """
    Plot a 2D array using Cartopy with U.S. state outlines and geographic features
    (coastlines, borders, states, rivers, lakes) from the built-in Natural Earth data.

    Parameters:
        lons, lats: 2D arrays of longitude and latitude.
        data_array: 2D numpy or xarray.DataArray.
        title: Plot title.
        cmap: colormap name.
        vmin, vmax: colorbar limits.
        alpha: transparency of the data layer (0–1).
    """
    fig = plt.figure(figsize=(10, 6))
    ax  = plt.axes(projection=ccrs.PlateCarree())

    # Plot data
    im = ax.pcolormesh(
        lons, lats, data_array,
        cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=alpha, shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1 # plot the data before adding features
    )
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cb.set_label(title)

    # Add geographic features
    feat_kw = dict(zorder=2, linewidth=0.8)
    ax.add_feature(cfeature.COASTLINE, **feat_kw)
    ax.add_feature(cfeature.BORDERS, linestyle=':', **feat_kw)
    ax.add_feature(cfeature.STATES, edgecolor='black', **feat_kw)
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', **feat_kw)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='blue', **feat_kw)

    # Zoom in on domain and render plot
    ax.set_extent([
        np.min(lons), np.max(lons),
        np.min(lats), np.max(lats)
    ], crs=ccrs.PlateCarree())

    ax.set_title(title)
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Latitude (°N)")

    plt.tight_layout()
    plt.show()
