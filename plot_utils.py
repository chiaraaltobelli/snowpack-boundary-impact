import matplotlib.pyplot as plt
import geopandas as gpd

def plot_with_state_outline(data_array, title, cmap="viridis", vmin=None, vmax=None,
                            shapefile_path=None, state_name="Idaho"):
    """
    Plot a 2D array with optional overlay of a state outline from a shapefile.
    
    Parameters:
        data_array (2D np.ndarray or xarray.DataArray): The gridded data to plot.
        title (str): Title for the plot.
        cmap (str): Colormap to use.
        vmin, vmax (float): Color limits.
        shapefile_path (str): Path to the shapefile (.shp).
        state_name (str): Name of the state to overlay (must match shapefile 'NAME' column).
    """
    plt.figure(figsize=(10, 6))
    im = plt.imshow(data_array, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=title)
    plt.title(title)
    plt.xlabel("Grid (west-east)")
    plt.ylabel("Grid (south-north)")

    if shapefile_path is not None:
        try:
            gdf = gpd.read_file(shapefile_path)
            state_geom = gdf[gdf['NAME'] == state_name]
            state_geom.boundary.plot(ax=plt.gca(), edgecolor='black', linewidth=1)
        except Exception as e:
            print(f"Failed to load or overlay shapefile: {e}")

    plt.tight_layout()
    plt.show()
