# Sea Surface Temperature Dataset

Due to the data size limitation, it is hard to push the original dataset here. But it is also possible to follow [this blog post](https://medium.com/p/21a6324df563) for
- downloading `sst.wkmean.1990-present.nc`,
- visualizing the temperature data.

## Visualization

- Visualize the distribution of mean temperature.

```python
from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

levs = np.arange(16, 29, 0.05)
jet = ["blue", "#007FFF", "cyan","#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"] 
cm = LinearSegmentedColormap.from_list('my_jet', jet, N = len(levs))

plt.rcParams['font.size'] = 12
fig = plt.figure(figsize = (7, 4))
mask = netcdf.NetCDFFile('lsmask.nc', 'r').variables['mask'].data[0, :, :]
mask = mask.astype(float)
mask[mask == 0] = np.nan
temp = netcdf.NetCDFFile('sst.wkmean.1990-present.nc', 'r').variables
plt.contourf(temp['lon'].data, temp['lat'].data, 
             np.mean(temp['sst'].data[: 1565, :, :], axis = 0) / 100 * mask, 
             levels = 20, linewidths = 1, vmin = 0, cmap = cm)
plt.xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
plt.yticks(np.arange(-60, 90, 30), ['60S', '30S', 'EQ', '30N', '60N'])
cbar = plt.colorbar(fraction = 0.022)
plt.show()
fig.savefig("mean_temperature.pdf", bbox_inches = "tight")
```

- Visualize the time series of mean temperature.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf
import time

data = netcdf.NetCDFFile('sst.wkmean.1990-present.nc', 'r').variables
tensor = data['sst'].data[:, :, :] / 100
T, M, N = tensor.shape
mat = np.zeros((M * N, T))
for t in range(T):
    mat[:, t] = tensor[t, :, :].reshape([M * N])

plt.rcParams['font.size'] = 11
fig = plt.figure(figsize = (8, 0.8))
ax = fig.add_subplot(1, 1, 1)
plt.plot(np.mean(mat, axis = 0), color = 'red', linewidth = 2, alpha = 0.6)
plt.axhline(y = np.mean(np.mean(mat)), color = 'gray', alpha = 0.5, linestyle='dashed')
plt.xticks(np.arange(1, 1565 + 1, 52 * 2), np.arange(1990, 2020 + 1, 2))
plt.grid(axis = 'both', linestyle='dashed', linewidth = 0.1, color = 'gray')
ax.tick_params(direction = "in")
ax.set_xlim([0, 1565])
plt.show()
fig.savefig("mean_temperature_time_series.pdf", bbox_inches = "tight")
```
