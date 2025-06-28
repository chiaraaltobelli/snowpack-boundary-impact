import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_cartopy(lons, lats, data, title,
                 cmap="viridis", vmin=None, vmax=None, alpha=0.6):
    fig = plt.figure(figsize=(10,6))
    ax  = plt.axes(projection=ccrs.PlateCarree())

    # 1) plot your gridded data on that GeoAxes
    mesh = ax.pcolormesh(
        lons, lats, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", alpha=alpha,
        zorder=1
    )
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label(title)

    # 2) add Natural Earth features
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

    # 3) put a thin black frame back around the GeoAxes
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1)

    # 4) optionally draw gridlines with labels
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    # 5) zoom to your data domain & finish labeling
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())
    ax.set_title(title)
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Latitude (°N)")

    plt.tight_layout()
    plt.show()
