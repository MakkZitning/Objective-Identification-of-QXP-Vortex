"""
此程序用于：
    画每一个识别到的低涡的每个时次叠加实时路径
"""


import cartopy.crs as crs
import cartopy.feature as cfeat
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps
import datetime
import geopandas
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import multiprocessing as multi
import numpy as np
import os
import pandas as pd
import salem
import time as tm
import xarray as xr

from atm_routine import create_submap
from metools import metools as mt
from vortex.clustering import clustering as vclst
import vortex_main as vm
from vortex.traversal import traversal as vtrav
from shearline.clustering import clustering as sclst
import shearline_main as sm
from shearline.trace import trace as strace
from shearline.traversal import traversal as strav


proj = crs.PlateCarree()
glbl = {}


## 设置青藏高原边界 ##
in_qxp = np.loadtxt(r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\tibetan.txt', delimiter=',')
## 读入经纬度常量 ##
lon, lat = mt.coords()
## 设置遍历范围 ##
nl = vm.nl
nr = vm.nr
md = vm.md
mu = vm.mu
## 设置切变线识别参数 ##
# 切变点判定阈值参数 #
r_1 = sm.r_1
twind_1 = sm.twind_1
tangl_1 = sm.tangl_1
# 切变线聚类参数 #
s_eps = sm.s_eps
s_mins = sm.s_mins
s_tless = sm.s_tless
s_tcdist = sm.s_tcdist
s_tdisp = sm.s_tdisp
## 设置低涡识别参数 ##
# 小尺度低涡中心判定阈值参数 #
r_s = vm.r_s
dtz_s = vm.dtz_s
t1_s = vm.t1_s
t2_s = vm.t2_s
t3_s = vm.t3_s
# 大尺度低涡中心判定阈值参数 #
r_x = vm.r_x
dtz_x = vm.dtz_x
t1_x = vm.t1_x
t2_x = vm.t2_x
t3_x = vm.t3_x
# 低涡聚类参数 #
v_eps = vm.v_eps
v_mins = vm.v_mins


## 创建地图底图 ##
def create_basemap():
    fig = plt.figure(1, figsize=(14, 12.25))
    gs = gridspec.GridSpec(nrows=1, ncols=1,
                           hspace=0.06, wspace=0.04)
    ax = plt.subplot(gs[0, 0], projection=proj)

    # 设定画图区域 #
    # extent = [70, 140, 15, 55]  # lon-lat: 69, 137, 16, 56 // 46, 154, 6, 64
    extent = [65, 145, 15, 55]  # lon-lat: 69, 137, 16, 56 // 46, 154, 6, 64
    # extent = [70, 110, 20, 40]  # lon-lat: 69, 137, 16, 56 // 46, 154, 6, 64
    ax.set_extent(extent, crs=proj)

    # 添加边界线 #
    shp = {
        r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\shp\青藏高原边界数据总集\TPBoundary_new(2021)\TPBoundary_new(2021).shp':
            [0.9, '--', 'none', 'grey', 0.25],  # [0.9, '--', 'none', 'dimgrey', 0.2]
        r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\shp\natural earth\50m_physical\50m_coastline.shp':
            [0.15, '--', 'grey', 'none', 1],
        r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\shp\中国\中国国界.shp':
            [0.6, '--', 'grey', 'none', 1],
        # r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\shp\中国\中国省界.shp':
        #     [0.6, '--', 'grey', 'none', 1],
        r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\shp\中国\中国九段线.shp':
            [1.8, '-.', 'grey', 'none', 1],
    }  # 在该字典里添加shp路径为键，边界线线宽、线形、颜色为值
    for _ in shp:
        border = cfeat.ShapelyFeature(Reader(_).geometries(), crs=proj)
        ax.add_feature(border,
                            linewidth=shp[_][0],
                            linestyle=shp[_][1],
                            edgecolor=shp[_][2],
                            facecolor=shp[_][3],
                            alpha=shp[_][4])

    # 添加经纬度刻度 #
    y_axis = np.arange(extent[2], extent[3] + 5, 5)
    x_axis = np.arange(extent[0], extent[1] + 10, 10)
    ax.set_yticks(y_axis, crs=proj)
    ax.set_xticks(x_axis, crs=proj)

    # 设置经纬度刻度格式（带不带°N或者°E） #
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # 添加南海小图 #
    ax_sub = fig.add_axes(
        [0.70, 0.16, 0.23, 0.27],
        projection=proj
    )
    _ax_s = create_submap.create_submap(ax, ax_sub, shp)

    return ax


## 作图 ##
def main():
    def draw():
        ax = create_basemap()

        # 绘制路径
        xlon = dbd.loc[(year, nflag), 'lon']
        ylat = dbd.loc[(year, nflag), 'lat']
        ax.plot(xlon, ylat, color='r', linewidth=0.4, linestyle='-', transform=proj, zorder=100)  # 绘制路径
        ax.scatter(xlon, ylat, color='k', s=0.4, marker='.', transform=proj, zorder=500)  # 绘制各时刻点

        # 标注信息
        ax.text(0.015, 0.95, fontsize=16, color='k',
                s='%d - %d' % (year, nflag),
                bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.8, 'pad': 1}, transform=ax.transAxes)
        ax.text(0.035, 0.85, fontsize=16, color='darkgrey',
                s='Starting on %d-%02d-%02d-%02d, %.2f°N, %.2f°E\n' %
                  (year,
                   dbd.loc[(year, nflag), :].head(1).month.values[0],
                   dbd.loc[(year, nflag), :].head(1).day.values[0],
                   dbd.loc[(year, nflag), :].head(1).hour.values[0],
                   dbd.loc[(year, nflag), :].head(1).lat.values[0],
                   dbd.loc[(year, nflag), :].head(1).lon.values[0]),
                bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.0, 'pad': 1}, transform=ax.transAxes)
        # i = 0
        # for idx, data in dbd.loc[(year, nflag), :].iterrows():
        #     ax.text(0.018, 0.90 - i, fontsize=6, color='dimgrey',
        #             s='%02d-%02d-%02d  %.2f, %.2f\n' % (data.month, data.day, data.hour, data.lat, data.lon),
        #             bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.8, 'pad': 1}, transform=ax.transAxes)
        #     i += 0.025

        # 存图设定
        os.makedirs(r'D:\图\识别结果\%d' % year, exist_ok=True)
        plt.savefig(r'D:\图\识别结果\%d' + r'\%d-%d--%02d.%02d.%02d.png' % (year, nflag, mm, dd, hh), dpi=800, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # 基础数据预备
    dbb = pd.read_csv(r'C:\Users\maizh\OneDrive\meteo_data\Data of my Thesis\vortex database\database_brief.csv', index_col=[0, 1])
    dbd = pd.read_csv(r'C:\Users\maizh\OneDrive\meteo_data\Data of my Thesis\vortex database\database_detail.csv', index_col=[0, 1])
    stat = pd.read_csv(r'C:\Users\maizh\OneDrive\meteo_data\Data of my Thesis\vortex database\statistic.csv', index_col=[0, 1])

    # 画图
    savefig_dir = r'C:\Users\maizh\OneDrive\桌面\毕业论文\图\杂图'  # 默认存图位置

    for year in range(2019, 2020):
        for nflag in range(dbd.loc[(year, slice(None)), :].index[-1][1] + 1):
            main()

    # 运行结束标志
    print('***************\n******End******\n***************')
