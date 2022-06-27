# var
Time-varying reduced-rank vector autoregression with tensor factorization.

## Datasets

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
