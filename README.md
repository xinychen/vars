# vars
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![repo size](https://img.shields.io/github/repo-size/xinychen/vars.svg)](https://github.com/xinychen/vars/archive/master.zip)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/vars.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/vars)

<h6 align="center">Made by Xinyu Chen • :globe_with_meridians: <a href="https://xinychen.github.io">https://xinychen.github.io</a></h6>

The scientific question is how to **discover dynamic patterns from spatiotemporal data**. We utilize the Vector Autoregression (VAR) as a basic tool to explore the spatiotemporal data in real-world applications.

<br>

## Data Sets

### JERICHO-E-usage dataset

- **Prepare `shapefile` data**
  - Download `shapefile` from [GISCO statistical unit dataset](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts), please set `Year: NUTS 2016`, `File format: SHP`, `Geometry type: Polygons (RG)`, `Scale: 03M`, and `Coordinate reference system: EPSG: 4326`.
  - Use the following code to visualize the map:

```python
import geopandas as gpd
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (14, 8))
ax = fig.subplots(1)
shape = gpd.read_file("NUTS_RG_03M_2016_4326.shp")
shape_de = shape[(shape['CNTR_CODE'] == 'DE') & (shape['LEVL_CODE'] == 2)]
shape_de.plot(cmap = 'YlOrRd_r', ax = ax)
plt.xticks([])
plt.yticks([])
for _, spine in ax.spines.items():
    spine.set_visible(False)
plt.show()
```

- **Prepare the energy consumption data**
  - Download `.zip` from [JERICHO-E-usage dataset](https://springernature.figshare.com/collections/Time_series_of_useful_energy_consumption_patterns_for_energy_system_modeling/5245457).
  - Open the dataset from the folder `JERICHO-E-usage/Singleindex files/Residential/nuts2_hourly_res_Space Heat_kw.csv`.
