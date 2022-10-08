# Daymet dataset

[Daymet](https://daac.ornl.gov/DAYMET) is a daily surface weather data on a 1-km grid for North America. We use the variable `maximum temperature` at the station level for evaluation (please [get data here](https://daac.ornl.gov/cgi-bin/dataset_lister.pl?p=32)). The period is from 2010 to 2021.

## Data processing

### Impute missing values

```python
import numpy as np
from netCDF4 import Dataset

temp0 = Dataset('daymet_v4_stnxval_tmax_na_2021.nc', "r", format="NETCDF4")
temp0 = temp0.variables
both = [bytes.decode(s) for s in np.frombuffer(temp0['station_name'][:].data, dtype='S255')]
for t in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
    temp = Dataset('daymet_v4_stnxval_tmax_na_20{}.nc'.format(t), "r", format="NETCDF4")
    temp = temp.variables
    str0 = [bytes.decode(s) for s in np.frombuffer(temp['station_name'][:].data, dtype='S255')]
    both = set(both).intersection(str0)

for t in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
    print('20{}'.format(t))
    temp = Dataset('daymet_v4_stnxval_tmax_na_20{}.nc'.format(t), "r", format="NETCDF4")
    temp = temp.variables
    str_new = [bytes.decode(s) for s in np.frombuffer(temp['station_name'][:].data, dtype='S255')]
    idx = [str_new.index(x) for x in both]
    mat = temp['obs'][:].data[idx, :]
    mat[mat == -9999] = np.nan
    pos = np.where(np.isnan(mat))

    rank = 20
    d = 1
    lambda0 = 0.1
    rho = 0.1
    season = 1
    maxiter = 50
    mat_hat, _, _, _ = notmf(mat, mat, rank, d, lambda0, rho, season, maxiter)
    mat_new = mat.copy()
    mat_new[pos] = mat_hat[pos]
    np.savez_compressed('daymet_tmax_na_20{}.npz'.format(t), mat_new)
```

where the `notmf` algorithm is available at [tracebase](https://github.com/xinychen/tracebase) project.

### Get the `lon` and `lat` of stations

There are 6,289 stations in total.

```python
temp = Dataset('daymet_v4_stnxval_tmax_na_2021.nc', "r", format="NETCDF4")
temp = temp.variables
idx = [str_new.index(x) for x in both]
station = np.zeros((len(both), 2))
station[:, 0] = temp['stn_lon'][:].data[idx]
station[:, 1] = temp['stn_lat'][:].data[idx]
np.savez_compressed('stations.npz', station)
```
where the first column corresponds to `lon`, and the second column corresponds to `lat`.

### Use the processed data in our work

```python
mat = np.load('daymet_tmax_na_2010.npz')['arr_0']
for t in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
    mat = np.append(mat, np.load('daymet_tmax_na_20{}.npz'.format(t))['arr_0'], axis = 1)
mat.shape
```
where the `shape` is `(6289, 4380)` over the 12 years (i.e., 4,380 days) from 2010 to 2021.

## Data visualization

### Filter stations and open data

```python
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

gdf = gpd.read_file("states.shp")
stations = np.load('stations.npz')['arr_0']
num = stations.shape[0]
pos = np.zeros((num, len(gdf['geometry'])))
for n in range(num):
    pt = Point(stations[n, 0], stations[n, 1])
    for r in range(len(gdf['geometry'])):
        if pt.within(gdf['geometry'][r]) == True:
            pos[n, r] = 1
        elif pt.within((gdf['geometry'][r])) == False:
            pos[n, r] = 0
pos = np.sum(pos, axis = 1)
station_us = stations[np.where(pos == 1)]

mat = np.load('daymet_tmax_na_2010.npz')['arr_0']
for t in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
    mat = np.append(mat, np.load('daymet_tmax_na_20{}.npz'.format(t))['arr_0'], axis = 1)
mat = mat[np.where(pos == 1)[0], :]
mat.shape
```

### Visualize data

```python
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

fig = plt.figure(figsize = (8, 4))
ax = fig.subplots(1)

gdf = gpd.read_file("states.shp")
gdf.plot(facecolor = 'white', edgecolor = 'black', linewidth = 1, ax = ax)

lon_lat = []
for i in range(station_us.shape[0]):
    lon_lat.append(Point(station_us[i, 0], station_us[i, 1]))
df = {'temp': np.mean(mat, axis = 1), 'geometry': lon_lat}

merged = gpd.GeoDataFrame(df, crs = "EPSG:4326")
merged.plot('temp', cmap = 'RdYlGn_r', markersize = 50, 
            legend = True, legend_kwds = {'shrink': 0.618}, ax = ax)

plt.xticks([])
plt.yticks([])
for _, spine in ax.spines.items():
    spine.set_visible(False)
plt.show()
fig.savefig("usa_temp_spatial_dist.png", bbox_inches = "tight")
```
