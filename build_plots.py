import datadotworld
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
import pyproj
import hdbscan

from shapely.geometry import MultiPoint, Point
from shapely.ops import transform
from toolz import curry

EARTH_RADIUS = 6371.0 # In kilometers.
sns.set_style('darkgrid')

# In order to draw a buffer properly I'll need to apply a map projection.
# This is because, unfortunately, the earth isn't flat.
# OR - if it is - we're doing "meters" wrong.
def geographic_buffer(geometry, distance):
    
    # In general picking a map projection amounts to picking the least worst 
    # solution. One fairly standard approach is to perform a universal 
    # transverse mercator (UTM) projection. The standard definition divides 
    # these into zones, but it's easy to recenter the central meridian at the 
    # centroid of the geometry (in this case the AF base) and perform the 
    # projection there to minimize distance distortion. The lon_0 keyword 
    # argument sets the central meridian. Note longitude is x.
    utm = pyproj.Proj(
        proj='utm',
        ellps='WGS84',
        lon_0=geometry.centroid.x
    )
    lonlat = pyproj.Proj(init='EPSG:4326')
    
    utm2lonlat = curry(pyproj.transform)(utm,lonlat)
    lonlat2utm = curry(pyproj.transform)(lonlat,utm)
    
    # First, put the geometry into UTM.
    geometry_utm = transform(lonlat2utm, geometry)
    # Draw the buffer in UTM coordinates, with meters as the units.
    geometry_utm_buffered = geometry_utm.buffer(distance*1000)
    # Transform back into lon/lat.
    geometry_buffered = transform(utm2lonlat, geometry_utm_buffered)
    
    return geometry_buffered

def main():
    """ Constructs all of the plots in the notebook as PNG files.
    """

    ufo_sightings_dataset = datadotworld.load_dataset(
        'timothyrenner/ufo-sightings'
    )

    ufo_sightings = \
        ufo_sightings_dataset \
            .dataframes['nuforc_reports'] \
            .rename(columns={"shape": "reported_shape"}) \
            .drop('city_location', axis=1)

    # Clean out the non-geocoded reports.
    non_coded_reports = ufo_sightings.city_longitude.isnull()
    ufo_sightings = ufo_sightings.loc[~non_coded_reports,:]

    std_dev_km = 5.0

    num_sightings = ufo_sightings.shape[0]

    # Jitter the latitudes and longitudes with a random normal with standard 
    # deviations at 5 kilometers. This is definitely _not_ the most studious way
    # to perform the jitter because the standard deviation has a systematic bias
    # due to the coordinate system, but it's not the worst thing in the world.
    ufo_sightings.loc[:,'latitude'] = np.random.normal(
        ufo_sightings.loc[:,'city_latitude'], 
        std_dev_km / EARTH_RADIUS, 
        num_sightings
    )
    ufo_sightings.loc[:,'longitude'] = np.random.normal(
        ufo_sightings.loc[:,'city_longitude'],
        std_dev_km / EARTH_RADIUS,
        num_sightings
    )

    # Convert the sightings to a geo data frame.
    ufo_sightings_geo = gpd.GeoDataFrame(
        ufo_sightings,
        geometry=list(
            MultiPoint(
                ufo_sightings.loc[:,['longitude','latitude']].values
            )
        ),
        crs={"init": "EPSG:4326"}
    )

    usa = gpd.read_file('data/external/cb_2016_us_state_500k.shp')
    # Remove non-continental US states.
    usa = usa[~usa.STUSPS.isin(['VI', 'AK', 'HI', 'PR', 'GU', 'MP', 'AS'])]
    sightings_conus = ufo_sightings_geo.state.isin(usa.STUSPS)

    ufo_sightings_geo = ufo_sightings_geo.loc[sightings_conus,:]

    ############################# UFO SIGHTINGS ################################
    print("Plotting the sightings.")
    # Albers Equal Area is pretty standard for US projections.
    proj = gplt.crs.AlbersEqualArea(
        central_longitude=-98, 
        central_latitude=39.5
    )

    # For some weirdo reason I have to set the ylim manually.
    # Reference: http://www.residentmar.io/geoplot/examples/usa-city-elevations.html
    ylim = (-1647757.3894385984, 1457718.4893930717)

    fig,ax = plt.subplots(subplot_kw={'projection':proj}, figsize=(16,12))
    gplt.polyplot(usa, projection=proj,
                       ax=ax,
                       linewidth=0.5,
                       facecolor='lightgray',
                       alpha=0.1)
    gplt.pointplot(ufo_sightings_geo, 
                   ax=ax, 
                   projection=proj, 
                   s=0.75,
                   alpha=0.25,
                   legend=True,
                   legend_values=[0, 10, 100, 1000],
                   legend_kwargs={'loc':'lower right'})
    ax.set_ylim(ylim)
    ax.set_title("UFO Sightings in the United States")
    fig.savefig('data/plots/ufo_sightings.png')
    print("Done plotting sightings.")
    ############################################################################

    bases = pd.read_csv('data/external/military_bases.csv')

    air_force_bases = \
        bases[bases.branch == 'Air Force']\
        [['branch','latitude','longitude']]\
        .drop_duplicates()\
        .reset_index()\
        .drop('index',axis=1)

    air_force_base_vicinities = [
        geographic_buffer(Point(row.longitude, row.latitude), 150.0)
        for _,row in air_force_bases.iterrows()
    ]

    air_force_bases_geo = gpd.GeoDataFrame(
        air_force_bases,
        geometry=air_force_base_vicinities,
        crs={"init": "EPSG:4326"}
    )

    # Isolate the rows that are inside the continental US.
    air_force_bases_geo = \
        air_force_bases_geo[
            air_force_bases_geo.intersects(usa.geometry.cascaded_union)
        ]
    
    ######################## UFO SIGHTINGS AF BASES ############################
    print("Plotting the sightings with Air Force bases.")
    # Albers Equal Area is pretty standard for US projections.
    proj = gplt.crs.AlbersEqualArea(
        central_longitude=-98, 
        central_latitude=39.5
    )

    # For some weirdo reason I have to set the ylim manually.
    # Reference: http://www.residentmar.io/geoplot/examples/usa-city-elevations.html
    fig,ax = plt.subplots(subplot_kw={'projection':proj}, figsize=(16,12))

    ylim = (-1647757.3894385984, 1457718.4893930717)
    gplt.polyplot(
        usa, 
        projection=proj,
        ax=ax,
        linewidth=0.5,
        facecolor='lightgray',
        alpha=0.1
    )
    gplt.pointplot(
        ufo_sightings_geo, 
        ax=ax, 
        projection=proj, 
        s=0.75,
        alpha=0.5
    )
    gplt.polyplot(
        air_force_bases_geo,
        ax=ax, 
        projection=proj, 
        linewidth=0.5,
        facecolor='none',
        edgecolor='red'
    )
    ax.set_ylim(ylim)
    ax.set_title("UFO Sightings in the United States")
    fig.savefig('data/plots/ufo_sightings_air_force_bases.png')
    print("Done plotting sightings with Air Force bases.")
    ############################################################################

    # Add the date of the sighting.
    ufo_sightings_geo.loc[:,'date'] = \
        pd.to_datetime(ufo_sightings_geo.date_time).dt.to_period("D")

    sightings_by_day = \
        ufo_sightings_geo\
            .groupby('date')\
            .agg({'report_link':'count'})\
            .rename(columns={'report_link':'report_count'})

    sightings_by_day.loc[:,'day_of_year'] = sightings_by_day.index.dayofyear

    sightings_by_day_of_year = \
        sightings_by_day\
            .groupby('day_of_year')\
            .agg({'report_count':'sum'})\
            .reset_index()
    
    ###################### UFO SIGHTINGS BY DAY OF YEAR ########################
    print("Plotting sightings by day of the year.")
    fig,ax = plt.subplots(figsize=(15,9))

    ax.plot(
        sightings_by_day_of_year.day_of_year.values,
        sightings_by_day_of_year.report_count
    )

    ax.set_title(
        "UFO Sightings by Day of Year", 
        fontsize=20, 
        fontweight="bold"
    )
    ax.set_xlabel(
        "Day of Year",
        fontsize=18
    )
    ax.set_ylabel(
        "Number of UFO Sightings",
        fontsize=18
    )

    ax.annotate(
        "Guess what day this is.",
        xy=(185, 1353),
        xytext=(215, 1353),
        arrowprops={
            "facecolor": "black",
            "shrink": 0.15
        },
        verticalalignment="center",
        fontsize=15,
        fontweight="bold"
    )

    ax.tick_params(labelsize=12, which='both')
    fig.savefig('data/plots/ufo_sightings_day_of_year.png')
    print("Done plotting sightings by day of the year.")
    ############################################################################

    ufo_sighting_coordinates = \
        np.radians(ufo_sightings_geo.loc[:,['latitude', 'longitude']].values)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50, 
        min_samples=1, 
        metric="haversine", 
        cluster_selection_method='leaf'
    )

    # This will take a while.
    ufo_sightings_geo.loc[:,'cluster_label'] = \
        clusterer.fit_predict(ufo_sighting_coordinates)

    cluster_polygons = [
        MultiPoint(group.loc[:,['longitude', 'latitude']].values).convex_hull
        for label,group in ufo_sightings_geo.groupby('cluster_label')
        if label != -1
    ]
    
    ###################### UFO SIGHTINGS WITH CLUSTERS #########################
    print("Plotting sightings with clusters.")
    # Albers Equal Area is pretty standard for US projections.
    proj = gplt.crs.AlbersEqualArea(
        central_longitude=-98, 
        central_latitude=39.5
    )

    # For some weirdo reason I have to set the ylim manually.
    # Reference: http://www.residentmar.io/geoplot/examples/usa-city-elevations.html
    fig,ax = plt.subplots(subplot_kw={'projection':proj}, figsize=(16,12))

    ylim = (-1647757.3894385984, 1457718.4893930717)
    gplt.polyplot(
        usa, 
        projection=proj,
        ax=ax,
        linewidth=0.5,
        facecolor='lightgray',
        alpha=0.1
    )
    gplt.polyplot(
        gpd.GeoSeries(
            [p for p in cluster_polygons if p.type == "Polygon"], 
            crs={"init": "EPSG:4326"}
        ),
        ax=ax,
        projection=proj,
        linewidth=0.5,
        facecolor='red',
        alpha=0.3
    )
    gplt.pointplot(
        ufo_sightings_geo, 
        ax=ax, 
        projection=proj, 
        s=0.75,
        alpha=0.5
    )
    ax.set_ylim(ylim)
    ax.set_title("UFO Sightings in the United States")

    fig.savefig('data/plots/ufo_sightings_with_clusters.png')
    print("Done plotting sightings with clusters.")
    ############################################################################

if __name__ == "__main__":
    main()