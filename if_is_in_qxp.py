"""
此程序用于：
    根据青藏高原shp生成0.25°×0.25°的二分值文件，以表征哪些格点位于高原上
"""


import numpy as np
import shapefile
import shapely.geometry as geometry
import time
import xarray as xr


time_start = time.time()
area = [40, 160, 0, 70]  # 与ERA5数据同尺寸

file = xr.open_dataset(r'D:\meteo_data\ERA5\500\19900501.nc')

lon = file.variables['longitude'].data[:]
lat = file.variables['latitude'].data[:]
lon, lat = np.meshgrid(lon, lat)
lonl = np.where(lon == area[0])[1][0]
lonr = np.where(lon == area[1])[1][0]
latd = np.where(lat == area[2])[0][0]
latu = np.where(lat == area[3])[0][0]

in_qxp = np.zeros_like(lon)
shp_file = shapefile.Reader(r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\shp\青藏高原边界数据总集\TPBoundary_new(2021)\TPBoundary_new(2021).shp', encoding='gbk')
shp_file = shp_file.shapeRecords()[0].shape
for m in range(latu, latd):
    for n in range(lonl, lonr):
        if geometry.Point(lon[m, n], lat[m, n]).within(geometry.shape(shp_file)):
            in_qxp[m, n] = 1
np.savetxt(r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\tibetan.txt', in_qxp, fmt='%d', delimiter=',')

time_end = time.time()
print('totally cost = %.4fs' % (time_end - time_start))
