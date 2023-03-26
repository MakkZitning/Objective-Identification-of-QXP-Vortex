import datetime
from math import ceil, log
import multiprocessing as multi
import numpy as np
from scipy.spatial.distance import cdist
import time as tm
import xarray as xr

from metools import metools as mt
from shearline.clustering import clustering
from shearline.trace import trace
from shearline.traversal import traversal


""""""" 设置青藏高原边界 """""""
in_qxp = np.loadtxt(r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\tibetan.txt', delimiter=',')
""""""" 读入经纬度常量 """""""
lon, lat = mt.coords()
""""""" 设置遍历范围 """""""
nl = mt.trans(coor=46, lon=lon)
nr = mt.trans(coor=154, lon=lon)
md = mt.trans(coor=6, lat=lat)
mu = mt.trans(coor=64, lat=lat)
""""""" 设置切变线识别参数 """""""
""" 切变点判定阈值参数 """
r_1 = 12  # 切变点判断半径 fixed=12
twind_1 = 7  # 判断半径两侧风速阈值
tangl_1 = 75  # 判断半径两侧风向阈值
""" 切变线聚类参数 """
s_eps = 0.5  # fixed=0.5
s_mins = 3  # fixed=3
s_tless = 10  # 剔除元素少于该值的簇
s_tcdist = 2
s_tdisp = 10
""""""" 设置起止时间 """""""
year_range = range(2018, 2019)


""""""" 创建切变线类 """""""
class Shearline:
    # 定义切变线类
    # nflag：切变线的编号，-1默认表示未编号
    # loc：包含切变线中心的经纬度以及经纬度对应下标，{'lat': xx, 'lon': xx, 'm': xx, 'n': xx}
    # time：切变线时次
    # pot：组成该切变线的潜在切变点

    def __init__(self, loc, time, pot):
        self.nflag = -1
        self.loc = loc
        self.time = time
        self.pot = pot


""""""" 判断是否位于高原边界线内 """""""
def is_in_qxp(k):
    insider = 0
    outsider = 0
    flag = 0
    t = ceil(len(k.pot) * 1 / 6)
    s = len(k.pot) - t
    for i in k.pot:
        if (25 <= i[0] <= 40) and (73 <= i[1] <= 105):
            insider = insider + 1
        else:
            outsider = outsider + 1
        if insider >= t:
            flag = 1
            break
        if outsider > s:
            break
    return flag


""""""" 路径追踪主体 """""""
def main(year):
    def read_data():
        nonlocal time, u, v

        file_dir = r'D:\meteo_data\ERA5\500\%s%02d%02d.nc' % (year, month, day)
        file = xr.open_dataset(file_dir)

        time = file.time.data[hour].astype('datetime64[ms]').astype('O')
        u = file.variables['u'].data[hour, :, :]
        v = file.variables['v'].data[hour, :, :]

    def trace_track():
        nonlocal time, u, v, num, past, track

        # 切变线移动路径追踪
        # num:编号指针
        # ng:新编号个数

        cluster = {}
        now = []

        # 若上一时次无切变线，开始新编号
        if len(past) == 0:
            # 遍历得到潜在切变点
            pot, pott = traversal(mu=mu, md=md, nl=nl, nr=nr,
                                  u=u, v=v,
                                  lat=lat, lon=lon,
                                  r=r_1, t_wind=twind_1, t_angl=tangl_1)
            if len(pot) != 0:
                # 将潜在切变点进行聚类
                clst_pot, clst_cent = clustering(data=pot, eps=s_eps, min_samples=s_mins, t_less=s_tless,
                                                 t_cdist=s_tcdist,
                                                 t_disp=s_tdisp)
                clst_endp, clst_pot = trace(cluster=clst_pot)
                for i in range(len(clst_cent)):
                    cluster[clst_cent[i]] = clst_pot[i]
                # 记录该时次的所有切变线簇
                for i, j in cluster.keys():
                    if len(cluster[i, j]) != 0:
                        now.append(Shearline(loc={'lat': i, 'lon': j}, time=time, pot=cluster[i, j]))
                # 为本时次切变线进行编号（仅新编号）
                ng = 0
                for nn in now:
                    track[num + ng] = []
                    track[num + ng].append(nn)
                    nn.nflag = num + ng
                    ng = ng + 1
                num = num + ng
        # 若上一时次有切变线，判断是重复编号或是新编号
        elif len(past) != 0:
            # 遍历得到潜在切变点
            pot, pott = traversal(mu=mu, md=md, nl=nl, nr=nr,
                                  u=u, v=v,
                                  lat=lat, lon=lon,
                                  r=r_1, t_wind=twind_1, t_angl=tangl_1)
            if len(pot) != 0:
                # 将潜在切变点进行聚类
                clst_pot, clst_cent = clustering(data=pot, eps=s_eps, min_samples=s_mins, t_less=s_tless,
                                                 t_cdist=s_tcdist,
                                                 t_disp=s_tdisp)
                print(len(clst_pot))
                clst_endp, clst_pot = trace(cluster=clst_pot)
                for i in range(len(clst_cent)):
                    cluster[clst_cent[i]] = clst_pot[i]
                print(len(clst_pot))
                # 记录该时次的所有切变线簇
                for i, j in cluster.keys():
                    if len(cluster[i, j]) != 0:
                        now.append(Shearline(loc={'lat': i, 'lon': j}, time=time, pot=cluster[i, j]))
                # 为本时次切变线进行编号（有旧编号，可能有新编号）
                ng = 0
                if len(now) != 0:
                    # 先将上一时次切变线与本时次切变线匹配(取上一时次与本时次相关性最大者)
                    # 当距离较近时，考虑：
                    ## 生命史较长者优先继承
                    dist = {}
                    corr = {}
                    for pp in range(len(past)):
                        for nn in range(len(now)):
                            # print(now[nn].pot)
                            dist[(pp, nn)] = np.mean([np.min(i) for i in cdist(past[pp].pot, now[nn].pot)])
                            corr[(pp, nn)] = 1 / ((dist[(pp, nn)] + 1) ** 2) - log(len(track[past[pp].nflag]))
                            # print('%.3f,%.3f--%.3f,%.3f  dist: %.3f  corr: %.3f' %
                            #       (past[pp].loc['lat'], past[pp].loc['lon'],
                            #        now[nn].loc['lat'], now[nn].loc['lon'],
                            #        dist[(pp, nn)], corr[(pp, nn)]))
                    for pp in range(len(past)):
                        max_corr = -10000
                        min_nn = -1
                        for nn in range(len(now)):
                            if now[nn].nflag == -1:
                                if corr[(pp, nn)] > max_corr and dist[(pp, nn)] <= 5.0:
                                    max_corr = corr[(pp, nn)]
                                    min_nn = nn
                        if min_nn != -1:
                            track[past[pp].nflag].append(now[min_nn])
                            now[min_nn].nflag = past[pp].nflag
                    # 再将无匹配的本时次切变线赋予新编号
                    for nn in now:
                        if nn.nflag == -1:
                            track[num + ng] = []
                            track[num + ng].append(nn)
                            nn.nflag = num + ng
                            ng = ng + 1
                    num = num + ng
        # 保留有编号的切变线到下一时次
        past = []
        for nn in now:
            if nn.nflag != -1:
                past.append(nn)
        # 输出识别结果至屏幕
        print('time: ', time, '      number of shearline clusters: ', len(cluster))
        print('--------------------------------------------------------------')
        if len(now) != 0:
            print(' LAT            LON           FLAG')
        for i in now:
            print('%.2f' % i.loc['lat'], '        ', '%.2f' % i.loc['lon'], '        ', i.nflag)
        print('\n')

    time = None
    u = None
    v = None
    num = 0
    past = []
    track = {}

    date_begin = datetime.date(year, 4, 15)
    date_end = datetime.date(year, 9, 30)
    hour_range = range(0, 24, 1)
    monlist = ['4', '5', '6', '7', '8', '9']

    # 循环主体
    for delta_day in range((date_end - date_begin).days + 1):
        date = date_begin + datetime.timedelta(days=delta_day)
        year = date.year
        month = date.month
        day = date.day
        if any(m in str(month) for m in monlist):
            for hour in hour_range:
                read_data()
                trace_track()
        else:
            continue

    return track


""""""" 存储切变线路径 """""""
def write_file(tt):
    year = tt[0][0].time.year
    f = open(r'C:\Users\maizh\OneDrive\桌面\毕业论文\切变线\original data' + '\\' + str(year) + '.txt', 'w')
    num = 0
    for k in tt.keys():
        mom = datetime.datetime.strptime(str(tt[k][0].time), '%Y-%m-%d %H:%M:%S').month
        if len(tt[k]) >= 12 and is_in_qxp(k=tt[k][0]) == 1 and mom in [5, 6, 7, 8, 9]:
            f.write(str(num))
            f.write('\n')
            for i in tt[k]:
                f.write(str(num))  # 存储切变线编号
                f.write(',')
                f.write('%.6f' % (i.loc['lat']))  # 存储切变点坐标纬度
                f.write(',')
                f.write('%.6f' % (i.loc['lon']))  # 存储切变点坐标经度
                f.write(',')
                f.write(str(i.time))  # 存储切变线时间
                f.write(',')
                f.write(str(len(i.pot)))  # 存储潜在切变点数量
                f.write(',')
                for j in i.pot:
                    f.write(str(j[0]))  # 存储潜在切变点坐标纬度
                    f.write('-')
                    f.write(str(j[1]))  # 存储潜在切变点坐标经度
                    f.write(' ')
                f.write('\n')
            num = num + 1
    f.close()


if __name__ == '__main__':
    time_start = tm.time()

    # 根据选取的起止时间决定是否使用多进程，以避免无谓开销
    FF = -1
    if year_range.stop - year_range.start > 1:
        FF = 0
    elif year_range.stop - year_range.start == 1:
        FF = 1
    if FF == 0:
        multi.freeze_support()
        pool = multi.Pool(processes=12)
        for yy in year_range:
            pool.apply_async(func=main, args=(yy,), callback=write_file)
        pool.close()
        pool.join()
    elif FF == 1:
        for yy in year_range:
            TRACK = main(year_range.start)
            write_file(TRACK)

    time_end = tm.time()
    print('totally cost = %.4fs' % (time_end - time_start))
