import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_with_state_outline(lons, lats, data_array, title, cmap="viridis", vmin=None, vmax=None):
    """
    Plot a 2D array using Cartopy with U.S. state outlines and geographic features, including rivers and lakes.
    
    Parameters:
        lons, lats: 2d arrays of longitude and latitude.
        data_array (2D np.ndarray or xarray.DataArray): The gridded data to plot.
        title (str): Title for the plot.
        cmap (str): Colormap to use (default is viridis).
        vmin, vmax (float): Color limits.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # === PLOT DATA ===
    im = ax.pcolormesh(
        lons, lats, data_array, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        shading="auto", 
        transform=ccrs.PlateCarree()
        )
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.2)
    cbar.set_label(title)

    # === ADD GEOGRAPHIC DATA ===
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':', linewidth=1)
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='blue', linewidth=1)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='lightblue', edgecolor='blue', linewidth=1)

    # === AXES SETTINGS ===
    ax.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)])
    ax.set_title(title)
    ax.set_xlabel("Longitude (°W)")           
    ax.set_ylabel("Latitude (°N)")

    plt.tight_layout()
    plt.show()