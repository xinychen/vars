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
