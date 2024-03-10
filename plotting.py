""" Shared plotting functions for the WSRA mean square slope notebooks """

from typing import Optional

import cartopy
import cmocean
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Colormap
from matplotlib.collections import QuadMesh, PathCollection, PatchCollection
from matplotlib.contour import QuadContourSet, ContourSet
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Arc


# Define default plot keyword arguments
default_ocean_kwargs = {'color': 'white'}
default_land_kwargs = {'color':'whitesmoke', 'zorder':3, 'alpha':0.4}
default_coast_kwargs = {'edgecolor': 'grey', 'linewidth': 0.5, 'zorder': 4}
default_intensity_cmap = mpl.cm.get_cmap('YlOrRd', 7)
default_wsra_marker_size = 5
default_storm_colors = {
    'earl': 'rebeccapurple',
    'fiona': 'orchid',
    'ian': 'steelblue',
    'julia': 'teal',
    'idalia': 'darkseagreen',
    'lee': 'cadetblue',
    # 'atomic': 'midnightblue'
    'atomic': 'deepskyblue'
}
default_storm_labels = {
    'earl': 'Earl (2022)',
    'fiona': 'Fiona (2022)',
    'ian': 'Ian (2022)',
    'julia': 'Julia (2022)',
    'idalia': 'Idalia (2023)',
    'lee': 'Lee (2023)',
    'atomic': 'ATOMIC (2020)',
}

default_drifter_marker_kwargs = {
    'microswift': {
        'label': 'microSWIFT',
        'color': 'grey',
        'edgecolor': 'k',
        's': 60,
        'marker': 'o',
        'zorder':  5,
    },
    'swift': {
        'label': 'SWIFT',
        'color': 'grey',
        'edgecolor': 'k',
        's': 60,
        'marker': 'o',
        'zorder':  5,
    },
    'spotter': {
        'label': 'Spotter',
        'color': 'goldenrod',
        'edgecolor': 'k',
        's': 70,
        'marker': 'p',
        'zorder': 5,
    },
    'dwsd': {
        'label': '(A)DWSD',
        'color': 'dodgerblue',
        'edgecolor': 'k',
        's': 60,
        'marker': 'o',
        'zorder': 5,
    }
}


def configure_figures() -> None:
    plt.rcParams.update({'font.size': 12})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = 'Helvetica'


def create_inset_colorbar(plot_handle, ax, bounds=None, **kwargs):
    # bounds = [x0, y0, width, height]
    if bounds is None:
        bounds = [0.93, 0.5, 0.02, 0.45]
    cax = ax.inset_axes(bounds, axes_class=mpl.axes.Axes)
    cbar = plt.colorbar(plot_handle, cax=cax, **kwargs)
    return cbar, cax


def plot_base_chart(
    ax: GeoAxes,
    extent: np.ndarray,
    ocean_kwargs=default_ocean_kwargs,
    land_kwargs=default_land_kwargs,
    coast_kwargs=default_coast_kwargs,
) -> GeoAxes:
    # Initialize the figure, crop it based on extent, and add gridlines
    ax.set_extent(extent)
    ax.set_aspect('equal')
    gridlines = ax.gridlines(draw_labels=True, dms=False,
                             x_inline=False, y_inline=False) # zorder=0)
    gridlines.top_labels = False
    gridlines.left_labels = False
    gridlines.right_labels = True

    # Add the ocean, land, coastline, and border features
    ax.add_feature(cartopy.feature.OCEAN, **ocean_kwargs)
    ax.add_feature(cartopy.feature.LAND, **land_kwargs)
    ax.add_feature(cartopy.feature.COASTLINE, **coast_kwargs)

    return ax


def plot_wsra_track(
    wsra_ds: pd.DataFrame,
    ax: GeoAxes,
    color_column_name: Optional[str] = None,
    **kwargs,
) -> PathCollection:
    if color_column_name:
        color_column = wsra_ds[color_column_name]
    else:
        color_column = None

    if 'color' not in kwargs:
        kwargs['color'] = default_storm_colors[wsra_ds.attrs['storm_name']]
    if 's' not in kwargs:
        kwargs['s'] = default_wsra_marker_size
    if 'label' not in kwargs:
        kwargs['label'] = default_storm_labels[wsra_ds.attrs['storm_name']]

    plot = ax.scatter(wsra_ds['longitude'],
                      wsra_ds['latitude'],
                      c=color_column,
                      **kwargs)
    return plot


def plot_colocated_drifter(
    wsra_ds: pd.DataFrame,
    drifter_type: str,
    ax: GeoAxes,
    color_column_name: Optional[str] = None,
    **kwargs,
) -> PathCollection:

    drifter_kwargs = default_drifter_marker_kwargs[drifter_type]
    for keyword in drifter_kwargs.keys():
        if keyword in kwargs:
            kwargs[keyword] = drifter_kwargs[keyword]

    prefix = drifter_type + '_'

    plot = ax.scatter(wsra_ds[prefix + 'longitude'],
                      wsra_ds[prefix + 'latitude'],
                      c=color_column_name,
                      **drifter_kwargs,
                      **kwargs)

    # plot = ax.plot(wsra_ds[prefix + 'longitude'],
    #                 wsra_ds[prefix + 'latitude'])
                    #   c=color_column_name,
                    #   **drifter_kwargs,
                    #   **kwargs)
    return plot


def plot_drifter_track(
    drifter_df: pd.DataFrame,
    ax: GeoAxes,
    color_column_name: Optional[str] = None,
    first_only=True,
    **kwargs,
) -> PathCollection:
    if first_only:
        drifter_df_plot = get_multiindex_first(drifter_df)
    else:
        drifter_df_plot = drifter_df

    if color_column_name:
        color_column = drifter_df_plot[color_column_name]
    else:
        color_column = None

    plot = ax.scatter(drifter_df_plot['longitude'],
                      drifter_df_plot['latitude'],
                      c=color_column,
                      **kwargs)
    return plot



def plot_best_track(
    pts_gdf: gpd.GeoDataFrame,
    lin_gdf: gpd.GeoDataFrame,
    windswath_gdf: gpd.GeoDataFrame,
    ax: GeoAxes,
    intensity_cmap: Colormap = default_intensity_cmap,
) -> GeoAxes:
    lin_gdf.plot(
        color='k',
        zorder=2,
        ax=ax
    )

    windswath_gdf[windswath_gdf['RADII'] == 64.0].plot(
        facecolor='dimgrey',
        alpha=0.3,
        ax=ax,
    )

    windswath_gdf[windswath_gdf['RADII'] == 50.0].plot(
        facecolor='darkgrey',
        alpha=0.3,
        ax=ax,
    )

    windswath_gdf[windswath_gdf['RADII'] == 34.0].plot(
        facecolor='lightgrey',
        alpha=0.3,
        ax=ax,
    )

    # Plot the best track points; color and label by intensity
    pts_gdf.plot(
        column='saffir_simpson_int',
        cmap=intensity_cmap,
        vmin=-1.5,
        vmax=5.5,
        edgecolor='k',
        zorder=4,
        markersize=200,
        alpha=1.0,
        ax=ax,
    )

    for x, y, label in zip(pts_gdf.geometry.x, pts_gdf.geometry.y, pts_gdf['saffir_simpson_label']):
        ax.annotate(
            label,
            xy=(x, y),
            annotation_clip=True,
            ha='center',
            va='center',
            zorder=10,
            fontsize=9,
            bbox=dict(boxstyle='circle,pad=0', fc='none', ec='none')
        )

    return ax

    #TODO:
    # counter = 0
    # for x, y, label in zip(pts_gdf.geometry.x, pts_gdf.geometry.y, pts_gdf.index):
    #     if counter % 3 == 0:
    #         ax.annotate(label.strftime('%Y-%m-%d %HZ') + '     ', xy=(x, y), annotation_clip=True, ha='right', va='center', zorder=10, fontsize=11,
    #             bbox=dict(boxstyle='circle,pad=0', fc='none', ec='none'))
    #     counter += 1


def tile_frequency(frequency_1d, length):
    return np.tile(frequency_1d, (length, 1))


def plot_wsra_frequency_spectrum(wsra_ds, ax, spectral_var_name='frequency_wave_spectrum', mean=True, **kwargs):
    if mean:
        spectral_var = wsra_ds[spectral_var_name].mean(axis=0)
        frequency = wsra_ds['frequency']
    else:
        spectral_var = wsra_ds[spectral_var_name]
        frequency = tile_frequency(wsra_ds['frequency'],
                                   len(spectral_var))

    plot = ax.plot(
       frequency.T,
       spectral_var.T,
       **kwargs,
   )
    return plot

def plot_colocated_drifter_spectrum(wsra_ds, drifter_type, ax, spectral_var_name='energy_density', mean=True, **kwargs):
    prefix = drifter_type + '_'

    if mean:
        spectral_var = wsra_ds[prefix + spectral_var_name].mean(dim='time')
        frequency = wsra_ds[prefix + 'frequency']
    else:
        spectral_var = wsra_ds[prefix + spectral_var_name]

        frequency = tile_frequency(wsra_ds[prefix + 'frequency'],
                                   len(spectral_var))
    plot = ax.plot(
       frequency.T,
       spectral_var.T,
       **kwargs,
    )
    return plot

def get_storm_color(storm_name):
    return default_storm_colors[storm_name]


def get_storm_label(storm_name):
    return default_storm_labels[storm_name]


def get_drifter_color(drifter_type):
    return default_drifter_marker_kwargs[drifter_type]['color']


def get_multiindex_first(drifter_df, level=0):
    return drifter_df.groupby(level=level).first()
