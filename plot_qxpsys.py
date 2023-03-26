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
## 设置起止时间（北京时） ##
year = 2000
monlist = ['4', '5', '6', '7', '8', '9']
date_begin = datetime.date(year, 5, 3)
date_end = datetime.date(year, 5, 3)
hour_range = range(18, 19, 1)


## 判断是否位于高原边界线内，若是则返回1 ##
def is_in_qxp(k):  # <-- (lat, lon)
    m = np.where(lat == k[0])[0][0]
    n = np.where(lon == k[1])[1][0]
    if in_qxp[m, n] == 1:
        return 1
    return 0


## 读取nc数据 ##
def read_data(yy, mm, dd, hh):
    ds = xr.open_dataset(r'D:\meteo_data\ERA5\500\%s%02d%02d.nc' % (yy, mm, dd))

    vo = ds['vo'].isel(time=hh) * 1e4
    u = ds['u'].isel(time=hh)
    v = ds['v'].isel(time=hh)
    z = ds['z'].isel(time=hh) / 9.8

    return vo, u, v, z


## 识别切变线 ##
def identify_shearline(u, v):
    # 得到切变点 #
    glbl['s_pot'], glbl['s_pott'] = strav(mu=mu, md=md, nl=nl, nr=nr,
                                          u=u, v=v,
                                          lat=lat, lon=lon,
                                          r=r_1, t_wind=twind_1, t_angl=tangl_1)
    # 切变点聚类 #
    if len(glbl['s_pot']) != 0:
        glbl['s_clst'], glbl['s_cent'] = sclst(data=glbl['s_pot'], eps=s_eps, min_samples=s_mins, t_less=s_tless,
                                               t_cdist=s_tcdist,
                                               t_disp=s_tdisp)
        glbl['s_endp'], glbl['shearline'] = strace(cluster=glbl['s_clst'])


## 识别低涡 ##
def identify_vortex(u, v, z):
    # 得到潜在低涡中心点 #
    glbl['v_pot'] = vtrav(mu=mu, md=md, nl=nl, nr=nr,
                          u=u, v=v, z=z,
                          lat=lat, lon=lon,
                          r_s=r_s, dtz_s=dtz_s, t1_s=t1_s, t2_s=t2_s, t3_s=t3_s,
                          r_x=r_x, dtz_x=dtz_x, t1_x=t1_x, t2_x=t2_x, t3_x=t3_x)
    # 低涡中心点聚类 #
    if len(glbl['v_pot']) != 0:
        glbl['v_clst'], glbl['v_cent'] = vclst(data=glbl['v_pot'], eps=v_eps, min_samples=v_mins)


## 创建地图底图 ##
def create_basemap():
    fig = plt.figure(1, figsize=(14, 12.25))
    gs = gridspec.GridSpec(nrows=1, ncols=1,
                           hspace=0.06, wspace=0.04)
    ax_main = plt.subplot(gs[0, 0], projection=proj)

    # 设定画图区域 #
    # extent = [70, 140, 15, 55]  # lon-lat: 69, 137, 16, 56 // 46, 154, 6, 64
    # extent = [65, 145, 15, 55]  # lon-lat: 69, 137, 16, 56 // 46, 154, 6, 64
    extent = [70, 110, 20, 40]  # lon-lat: 69, 137, 16, 56 // 46, 154, 6, 64
    ax_main.set_extent(extent, crs=proj)

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
        ax_main.add_feature(border,
                            linewidth=shp[_][0],
                            linestyle=shp[_][1],
                            edgecolor=shp[_][2],
                            facecolor=shp[_][3],
                            alpha=shp[_][4])

    # 添加经纬度刻度 #
    y_axis = np.arange(extent[2], extent[3] + 5, 5)
    x_axis = np.arange(extent[0], extent[1] + 10, 10)
    ax_main.set_yticks(y_axis, crs=proj)
    ax_main.set_xticks(x_axis, crs=proj)

    # 设置经纬度刻度格式（带不带°N或者°E） #
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax_main.xaxis.set_major_formatter(lon_formatter)
    ax_main.yaxis.set_major_formatter(lat_formatter)

    # 网格线 #
    # yy = np.arange(extent[2], extent[3] + 0.25, 0.25)
    # xx = np.arange(extent[0], extent[1] + 0.25, 0.25)
    #
    # gl = ax_main.gridlines(crs=proj, draw_labels=True, linewidth=0.05, color='black', alpha=1.0, linestyle='-', ylocs=yy, xlocs=xx)
    # gl.top_labels = False
    # gl.bottom_labels = False
    # gl.left_labels = False
    # gl.right_labels = False
    
    # 添加南海小图 #
    ax_sub = fig.add_axes(
        [0.70, 0.16, 0.23, 0.27],
        projection=proj
    )
    _ax_s = create_submap.create_submap(ax_main, ax_sub, shp)
    
    return ax_main


## 相关绘图设置 ##
def plot(vo, u, v, z, ax, yy, mm, dd, hh, flag):
    # 生成n个随机颜色的色表 #
    def colorlist_generator(n):
        import random

        rangelist = [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
        ]
        n = int(n)
        colors_list = []
        for _i in range(n):
            color = "#"  # 一个7位（"#"+6个16进制数字）的表示颜色的字符串
            for _j in range(6):
                color += rangelist[random.randint(0, 14)]  # 用randint生成0到14的随机整数作为索引
            colors_list.append(color)
        return colors_list

    # 绘制等值填色图 #
    def draw_contourf(data):
        levels = np.linspace(1, 5, 5)
        colormap = cmaps.WhiteYellowOrangeRed  # ncl色表:cmaps.BlueWhiteOrangeRed, matplotlib色表:plt.cm.Reds

        ctrf = ax.contourf(lon, lat, data,
                           levels=levels, cmap=colormap, extend='max', transform=proj)
        cbar = plt.colorbar(ctrf, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04, aspect=60, shrink=0.75,
                            extendrect=False, format='%.2g')
        cbar.set_label('Relative Vorticity ($10^4$$s^-$$^1$)', size='large', loc='center')

    # 绘制等值线 #
    def draw_contour(data, colors, spacing=10):
        # levels = np.arange(5, 60010, spacing)
        # ctr1 = ax.contour(lon, lat, data, levels=levels, colors=colors, alpha=0.7, linewidths=0.5, transform=proj)
        levels = np.arange(0, 6000, spacing)
        ctr2 = ax.contour(lon, lat, data, levels=levels, colors=colors, alpha=0.7, linewidths=0.25, transform=proj)
        ax.clabel(ctr2, colors=colors, fontsize=4, fmt='%d', use_clabeltext=True)

    # 绘制风标 #
    def draw_wind(density=8, wind_type=0, alpha=1.0):
        if wind_type == 0:
            ax.barbs(x=lon[::density, ::density], y=lat[::density, ::density],
                     u=u[::density, ::density], v=v[::density, ::density],
                     linewidth=1, flagcolor='black', alpha=alpha, length=4.5, fill_empty=False,
                     pivot='tip', barb_increments=dict(half=3.89, full=7.78, flag=38.88),
                     sizes=dict(spacing=0.2, height=0.6, width=0.4), transform=proj, zorder=1000)
        elif wind_type == 1:
            ax.quiver(x=lon[::density, ::density], y=lat[::density, ::density],
                      u=u[::density, ::density], v=v[::density, ::density], pivot='tip',
                      scale=750, alpha=alpha, color='black', transform=proj, zorder=1000)
        else:
            print('select the correct wind_type you want to draw')
            quit()

    # 基础打点 #
    def draw_somne_point(data, facecolors, edgecolors, size, marker):
        x = []
        y = []
        for i in data:
            x.append(i[1])
            y.append(i[0])
        ax.scatter(x, y, facecolors=facecolors, edgecolors=edgecolors, s=size, marker=marker, transform=proj)

    # 绘制切变点 #
    def draw_shearpoint():
        # draw_somne_point(data=glbl['s_pot'], facecolors=glbl['s_pott'], edgecolors='none', size=24, marker='.')
        draw_somne_point(data=glbl['s_pot'], facecolors='blue', edgecolors='none', size=16, marker='.')
        
        ax.text(0.01, 0.92, s='a', fontsize=28, color='k', transform=ax.transAxes,
                bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 1},
                zorder=2000
                )

    # 绘制切变线簇 #
    def draw_scluster():
        clist_fixed = ['#2A752B', '#713375', '#2E84D6', '#5863D7', '#138D84', '#44EEFB', '#CD52FA', '#E6EC29', '#19C68C', '#D2CA37', '#5D6DDD', '#B9CB9D', '#5F826D']
        for i in range(len(glbl['s_clst'])):
            draw_somne_point(data=glbl['s_clst'][i], facecolors=clist_fixed[i], edgecolors='none', size=8, marker='D')
            # draw_somne_point(data=glbl['s_clst'][i], facecolors=clist[i], edgecolors='none', size=8, marker='D')
            
        ax.text(0.01, 0.92, s='b', fontsize=28, color='k', transform=ax.transAxes,
                bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 1},
                zorder=3000
                )

    # 绘制切变线簇的端点 #
    def draw_endpoint():
        for i in glbl['s_endp']:
            draw_somne_point(data=i, facecolors='red', edgecolors='none', size=32, marker='o')
            
        ax.text(0.01, 0.92, s='c', fontsize=28, color='k', transform=ax.transAxes,
                bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 1},
                zorder=4000
                )

    # 绘制切变线 #
    def draw_shearline():
        for i in range(len(glbl['shearline'])):
            for j in range(len(glbl['shearline'][i]) - 1):
                ax.plot(
                    (glbl['shearline'][i][j][1], glbl['shearline'][i][j + 1][1]),
                    (glbl['shearline'][i][j][0], glbl['shearline'][i][j + 1][0]),
                    color='sienna', linewidth=2.0, linestyle='-', transform=proj)
                
        ax.text(0.01, 0.92, s='d', fontsize=28, color='k', transform=ax.transAxes,
                bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 1},
                zorder=5000
                )

    # 绘制低涡特征点 #
    def draw_vortexpoint():
        draw_somne_point(data=glbl['v_pot'], facecolors='lime', edgecolors='none', size=24, marker='.')

    # 绘制位于qxp内的低涡特征点 #
    def draw_insider():
        insider = []
        for i in glbl['v_clst']:
            for j in i:
                if is_in_qxp(j):
                    insider.append(j)
        draw_somne_point(data=insider, facecolors='red', edgecolors='none', size=24, marker='.')

    # 绘制低涡簇 #
    def draw_vcluster():
        clist_fixed = ['#C47B1E', '#BB69FD', '#B7D695', '#B8BB9C', '#E4738E', '#539E75', '#D949CB', '#76589A', '#DAFC69', '#BEB5CA', '#DC1C63', '#12B4CF']

        for i in range(len(glbl['v_clst'])):
            # draw_somne_point(data=glbl['v_clst'][i], facecolors=clist_fixed[i], edgecolors='none', size=24, marker='.')  # 固定颜色予不同簇
            draw_somne_point(data=glbl['v_clst'][i], facecolors=clist[i], edgecolors='none', size=24, marker='.')  # 随机颜色予不同簇

    # 绘制低涡中心 #
    def draw_vortexcenter():
        # for i in range(len(glbl['v_clst'])):
        #     draw_somne_point(data=glbl['v_cent'], facecolors=clist[i], edgecolors=clist[i], size=800, marker='x')
        draw_somne_point(data=glbl['v_cent'], facecolors='blue', edgecolors='blue', size=800, marker='x')

    # 标示低涡中心坐标 #
    def label_vortexcenter():
        for i in range(len(glbl['v_cent'])):
            flag_inside = 0
            for j in glbl['v_clst'][i]:
                if flag_inside == 0:
                    for k in j:
                        if is_in_qxp(j) and flag_inside == 0:
                            text = '%.2f, %.2f' % (glbl['v_cent'][i][0], glbl['v_cent'][i][1])
                            ax.text(glbl['v_cent'][i][1], glbl['v_cent'][i][0] + 2.5, s=text, fontsize=10, weight='bold', color='blue', transform=proj, ha='center',
                                    bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.4, 'pad': 1}
                                    )
                            flag_inside = 1

    # 设置标题 #
    def set_title():
        ax.set_title('500-hPa', fontsize=14, loc='left')
        ax.set_title('%s-%02d-%02d-%02d:00 (UTC+8)' % (yy, mm, dd, hh), fontsize=14, loc='right')

    # 设置序号 #
    def set_figlabel(text='0', size=28, alpha=1):
        if text != '0':
            ax.text(0.01, 0.92, s=text, fontsize=28, color='k', transform=ax.transAxes,
                    bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 1},
                    zorder=5000
                    )

    def draw_track():
        # 绘制路径
        xlon = dbd.loc[(year, nflag), 'lon']
        ylat = dbd.loc[(year, nflag), 'lat']
        ax.plot(xlon, ylat, color='r', linewidth=2, linestyle='-', transform=proj, zorder=9000)  # 绘制路径
        # ax.scatter(xlon, ylat, color='k', s=1, marker='.', transform=proj, zorder=9000)  # 绘制各时刻点

        ax.text(0.0125, 0.925, fontsize=24, color='red',
                s='%d - %02d' % (year, nflag),
                bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 1.0, 'pad': 1}, transform=ax.transAxes, zorder=9999)

    dbd = pd.read_csv(r'C:\Users\maizh\OneDrive\meteo_data\Data of my Thesis\vortex database\database_detail.csv', index_col=[0, 1])
    nflag = 8
    # draw_track()  # 画移动路径

    # mask数据
    # shp = shp.loc[
    #     # (shp['省'] == '新疆维吾尔自治区') |
    #     (shp['省'] == '西藏自治区') |
    #     (shp['省'] == '青海省') |
    #     (shp['省'] == '四川省') |
    #     (shp['省'] == '甘肃省') |
    #     (shp['省'] == '陕西省') |
    #     (shp['省'] == '山西省') |
    #     (shp['省'] == '内蒙古自治区')
    # ]

    # 绘制vo填色 #
    # draw_contourf(data=vo)

    # 绘制z等值线 #
    # shp = geopandas.read_file(r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\shp\中国\中国省界.shp')
    # z = z.salem.roi(shape=shp)
    draw_contour(data=z, colors='purple', spacing=10)  # spacing=20
    shp = geopandas.read_file(r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\shp\青藏高原边界数据总集\TPBoundary_new(2021)\TPBoundary_new(2021).shp')
    z = z.salem.roi(shape=shp)
    draw_contour(data=z, colors='purple', spacing=5)

    # 绘制风场 #
    # shp = geopandas.read_file(r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\shp\中国\中国省界.shp')
    # u = u.salem.roi(shape=shp)
    # v = v.salem.roi(shape=shp)
    draw_wind(density=6, wind_type=0, alpha=1.0)

    # 绘制低涡或切变线相关 #
    if 's_pot' in glbl:  # 切变线
        if len(glbl['s_pot']) != 0:
            clist = colorlist_generator(len(glbl['s_clst']))  # 生成随机色表
            print(clist)

            draw_shearpoint()  # 绘制切变点
            draw_scluster()  # 绘制切变点聚类
            draw_endpoint()  # 绘制切变端点
            draw_shearline()  # 绘制切变线
    if 'v_pot' in glbl:  # 低涡
        if len(glbl['v_pot']) != 0:
            clist = colorlist_generator(len(glbl['v_clst']))  # 生成随机色表
            print(clist)

            draw_vortexpoint()  # 绘制低涡特征点
            draw_insider()  # 仅绘制位于qxp内的低涡特征点
            # draw_vcluster()  # 绘制不同簇的低涡特征点
            draw_vortexcenter()  # 绘制聚类中心（低涡中心）
            label_vortexcenter()  # 标示聚类中心（低涡中心）

            #  年鉴低涡
            # nianjian = (33.5, 82.0)
            # nianjian = (33.0, 91.1)
            # draw_somne_point(data=[nianjian], facecolors='red', edgecolors='red', size=800, marker='x')
            # ax.text(nianjian[1], nianjian[0] - 2.5, s='%.1f, %.1f' % nianjian, fontsize=10, weight='bold', color='red', transform=proj, ha='center',
            #         bbox={'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.4, 'pad': 1}
            #         )
            #  年鉴低涡路径
            # xlon = [100.0, 102.0, 109.5, 112.5, 116.0, 120.3, 124.1, 131.2]
            # ylat = [35.8, 37.0, 35.4, 36.7, 36.1, 36.7, 37.7, 41.0]
            # ax.plot(xlon, ylat, color='r', linewidth=1.2, linestyle='-', transform=proj, zorder=100)  # 绘制路径
            # ax.scatter(xlon, ylat, color='k', s=1.6, marker='.', transform=proj, zorder=500)  # 绘制各时刻点

    set_title()

    set_figlabel(text='0', size=28, alpha=1)

    plt.savefig(r'D:\图\识别结果\500-hPa_%s.%02d.%02d.%02d(UTC+8).png' % (yy, mm, dd, hh), bbox_inches='tight', dpi=800)  # dpi=50
    plt.close()


## 控制程序开关 ##
def main(yy, mm, dd, hh, ff):
    time_utc0 = datetime.datetime(yy, mm, dd, hh) - datetime.timedelta(hours=8)  # 将输入的UTC+8转换为UTC+0以便读取数据
    yy = time_utc0.year
    mm = time_utc0.month
    dd = time_utc0.day
    hh = time_utc0.hour
    vo, u, v, z = read_data(yy=yy, mm=mm, dd=dd, hh=hh)

    # identify_shearline(u.data, v.data)  # 识别切变线
    identify_vortex(u.data, v.data, z.data)  # 识别低涡

    ax = create_basemap()
    dtime = datetime.datetime(yy, mm, dd, hh) + datetime.timedelta(hours=8)  # 转换回UTC+8
    plot(vo, u, v, z, ax=ax, yy=dtime.year, mm=dtime.month, dd=dtime.day, hh=dtime.hour, flag=ff)


if __name__ == '__main__':
    time_start = tm.time()

    # 根据选取的起止时间决定是否使用多进程，以避免无谓开销 #
    if hour_range.stop - hour_range.start > 1:
        multi.freeze_support()
        pool = multi.Pool(processes=multi.cpu_count())
        # pool = multi.Pool(processes=3)
        for delta_day in range((date_end - date_begin).days + 1):
            date = date_begin + datetime.timedelta(days=delta_day)
            YY = date.year
            MM = date.month
            DD = date.day
            if any(m in str(MM) for m in monlist):
                for HH in hour_range:
                    pool.apply_async(func=main, args=(YY, MM, DD, HH, 0,))
            else:
                continue
        pool.close()
        pool.join()
    elif hour_range.stop - hour_range.start == 1:
        for delta_day in range((date_end - date_begin).days + 1):
            date = date_begin + datetime.timedelta(days=delta_day)
            YY = date.year
            MM = date.month
            DD = date.day
            if any(m in str(MM) for m in monlist):
                for HH in hour_range:
                    main(yy=YY, mm=MM, dd=DD, hh=HH, ff=1)
            else:
                continue

    time_end = tm.time()
    print('totally cost = %.4fs' % (time_end - time_start))
