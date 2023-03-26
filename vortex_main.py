import datetime
import multiprocessing as multi
import numpy as np
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import sys
import time as tm
from tqdm import tqdm
import xarray as xr

from metools import metools as mt
from vortex.clustering import clustering
from vortex.traversal import traversal

## 设置青藏高原边界 ##
in_qxp = np.loadtxt(r'C:\Users\maizh\OneDrive\meteo_data\Geo_Data\tibetan.txt', delimiter=',')
## 读入经纬度常量 ##
lon, lat = mt.coords()
## 设置遍历范围 ##
nl = mt.trans(coor=46, lon=lon)  # 46
nr = mt.trans(coor=154, lon=lon)  # 154
md = mt.trans(coor=6, lat=lat)  # 6
mu = mt.trans(coor=64, lat=lat)  # 64
## 设置低涡识别参数 ##
# 小尺度低涡中心判定阈值参数 #
r_s = 4  # 闭合等高线判断半径 fixed=4
dtz_s = 5  # 闭合等高线值 fixed=5
t1_s = 6  # 要求达标旋臂数（高度值递增） fixed=6
t2_s = 8  # 要求达标旋臂数（闭合等高线） fixed=8
t3_s = 6  # 要求达标旋臂数（气旋式风场） fixed=6
# 大尺度低涡中心判定阈值参数 #
r_x = 6  # 闭合等高线判断半径 fixed=6
dtz_x = 10  # 闭合等高线值 fixed=10
t1_x = 8  # 要求达标旋臂数（高度值递增） fixed=8
t2_x = 8  # 要求达标旋臂数（闭合等高线） fixed=8
t3_x = 6  # 要求达标旋臂数（气旋式风场） fixed=6
# 低涡聚类参数 #
v_eps = 1  # fixed=1
v_mins = 4  # fixed=4
# 识别时间区间 #
year_range = range(1990, 2020)  # fixed=(1990, 2020)
begin = (4, 15)  # fixed=(4, 15)
end = (9, 30)  # fixed=(9, 30)
monlist = ['4', '5', '6', '7', '8', '9']


## 创建低涡类 ##
class Vortex:
    # 定义低涡类
    # nflag：低涡的编号，-1默认表示未编号
    # loc：包含低涡中心的经纬度以及经纬度对应下标，{'lat': xx, 'lon': xx, 'm': xx, 'n': xx}
    # time：低涡时次
    # pot：组成该低涡的潜在低涡中心点

    def __init__(self, loc, time, pot):
        self.nflag = -1
        self.loc = loc
        self.time = time
        self.pot = pot


## 判断是否位于高原边界线内，若是则返回1 ##
def is_in_qxp(k):
    for _ in k.pot:
        m = np.where(lat == _[0])[0][0]
        n = np.where(lon == _[1])[1][0]
        if in_qxp[m, n] == 1:
            return 1
    return 0


## 预先检查缺失的nc文件 ##
def check_missing():
    file_dir = r'D:\meteo_data\ERA5\500'
    for root, dirs, files in os.walk(file_dir):
        flag = 0
        for year in year_range:
            date_begin = datetime.date(year, begin[0], begin[1])
            date_end = datetime.date(year, end[0], end[1])
            for delta_day in range((date_end - date_begin).days + 1):
                date = date_begin + datetime.timedelta(days=delta_day)
                year = date.year
                month = date.month
                day = date.day
                if '%s%02d%02d.nc' % (year, month, day) not in files:
                    flag = 1
                    print('missing file %s%02d%02d.nc' % (year, month, day))
        if flag:
            sys.exit(0)


## 路径追踪主体 ##
def main(*args):
    # 读取nc文件并提取变量
    def read_data():
        nonlocal time, u, v, z

        ds = xr.open_dataset(r'D:\meteo_data\ERA5\500\%s%02d%02d.nc' % (year, month, day))

        time = ds.time.data[hour].astype('datetime64[ms]').astype('O')
        time += datetime.timedelta(hours=8)  # 转换为UTC+8
        u = ds.variables['u'].data[hour, :, :]
        v = ds.variables['v'].data[hour, :, :]
        z = ds.variables['z'].data[hour, :, :] / 9.8

    # 将未编号的潜在低涡中心点簇按最近距离归一化
    def now_the_closests_to_one(_now):
        print('----------------------------------------------------', file=debug)
        pool_set = []
        for i in range(len(_now)):
            for j in range(i + 1, len(_now)):
                if (_now[i].nflag == -1) and (_now[j].nflag == -1):
                    dist_closest = np.min([np.min(_) for _ in cdist(_now[i].pot, _now[j].pot)])
                    dist_center = np.sqrt((_now[i].loc['lon'] - _now[j].loc['lon']) ** 2 +
                                          (_now[i].loc['lat'] - _now[j].loc['lat']) ** 2)
                    if (dist_closest <= 2.0) and (dist_center <= 4.0):  # 两个潜在低涡中心点簇若同时满足(1).最近潜在低涡中心点足够近；(2).潜在低涡中心足够近
                        pool_set.append({i, j})  # 则将两个潜在低涡中心点簇归为一簇
        for _ in _now:
            print('now without flag before:  %-3d (%6.3f %7.3f)' % (_.nflag, _.loc['lat'], _.loc['lon']), file=debug)
        print('pool_set before:', pool_set, file=debug)
        for i in range(len(pool_set)):
            for j in range(i + 1, len(pool_set)):
                if len(pool_set[i] & pool_set[j]) > 0:
                    pool_set[i] = pool_set[i] | pool_set[j]
                    pool_set[j] = set([])
        print('pool_set after:', pool_set, file=debug)
        discard = []
        for _set in pool_set:
            if len(_set) > 0:
                _lat = 0.0
                _lon = 0.0
                _pot = []
                count = 0
                for _ in _set:
                    discard.append(_now[_])
                    print('now to be discarded: %-3d (%6.3f %7.3f)' % (_now[_].nflag, _now[_].loc['lat'], _now[_].loc['lon']), file=debug)
                    count += 1
                    _lat += _now[_].loc['lat']
                    _lon += _now[_].loc['lon']
                    _pot = _pot + _now[_].pot
                _lat /= count
                _lon /= count
                _now.append(Vortex(loc={'lat': _lat, 'lon': _lon}, time=time, pot=_pot))
                print('now append new merged vortex:', _lat, _lon, file=debug)
        _now = list(set(_now) - set(discard))
        # 将本时次的now根据pot数量进行排序，pot数量越多者先编号
        for i in range(len(_now)):
            for j in range(i + 1, len(_now)):
                if len(_now[i].pot) <= len(_now[j].pot):
                    tmp = _now[i]
                    _now[i] = _now[j]
                    _now[j] = tmp
                elif len(_now[i].pot) == len(_now[j].pot):
                    if _now[i].loc['lat'] < _now[j].loc['lat']:
                        tmp = _now[i]
                        _now[i] = _now[j]
                        _now[j] = tmp
        return _now

    def trace_track():
        nonlocal time, u, v, z, num, past, track

        # 低涡移动路径追踪
        # num:编号指针
        # ng:新编号个数（临时编号偏移指针）

        cluster = {}
        now = []

        print('===================', file=debug)
        print(time, file=debug)
        print('===================', file=debug)
        print('----------------------------------------------------', file=debug)

        # 若上一时次无低涡，开始新编号
        if len(past) == 0:
            # 遍历得到潜在低涡中心点
            pot = traversal(mu=mu, md=md, nl=nl, nr=nr,
                            u=u, v=v, z=z,
                            lat=lat, lon=lon,
                            r_s=r_s, dtz_s=dtz_s, t1_s=t1_s, t2_s=t2_s, t3_s=t3_s,
                            r_x=r_x, dtz_x=dtz_x, t1_x=t1_x, t2_x=t2_x, t3_x=t3_x)
            if len(pot) != 0:
                # 将潜在低涡中心点进行聚类得到低涡中心点簇
                clst_pot, clst_cent = clustering(data=pot, eps=v_eps, min_samples=v_mins)
                for i in range(len(clst_cent)):
                    cluster[clst_cent[i]] = clst_pot[i]
                # 记录该时次的所有低涡簇
                for i, j in cluster.keys():
                    now.append(Vortex(loc={'lat': i, 'lon': j}, time=time, pot=cluster[i, j]))
                # 将未编号的潜在低涡中心点簇按最近距离归一化
                now = now_the_closests_to_one(now)
                # 为本时次低涡进行编号（只有新编号）
                ng = 0
                for nn in now:
                    track[num + ng] = []
                    track[num + ng].append(nn)
                    nn.nflag = num + ng
                    ng = ng + 1
                num = num + ng
        # 若上一时次有低涡，判断是重复编号还是开始新编号
        elif len(past) != 0:
            # 遍历得到潜在低涡中心点
            pot = traversal(mu=mu, md=md, nl=nl, nr=nr,
                            u=u, v=v, z=z,
                            lat=lat, lon=lon,
                            r_s=r_s, dtz_s=dtz_s, t1_s=t1_s, t2_s=t2_s, t3_s=t3_s,
                            r_x=r_x, dtz_x=dtz_x, t1_x=t1_x, t2_x=t2_x, t3_x=t3_x)
            if len(pot) != 0:
                # 将潜在低涡中心点进行聚类得到低涡中心点簇
                clst_pot, clst_cent = clustering(data=pot, eps=v_eps, min_samples=v_mins)
                for i in range(len(clst_cent)):
                    cluster[clst_cent[i]] = clst_pot[i]
                # 记录该时次的所有低涡簇
                for i, j in cluster.keys():
                    now.append(Vortex(loc={'lat': i, 'lon': j}, time=time, pot=cluster[i, j]))
                # 为本时次低涡进行编号（1.有旧编号+有新编号；2.只有旧编号；3.只有新编号）
                for pp in range(len(past)):
                    #
                    print('--------------------------', file=debug)
                    print('past: %-3d (%6.3f %7.3f)' % (past[pp].nflag, past[pp].loc['lat'], past[pp].loc['lon']), file=debug)
                    print('--------------------------', file=debug)
                    print('now before:', [len(_.pot) for _ in now], file=debug)
                    # 将本时次的now根据pot数量进行排序，pot数量越多者获得优先判断权
                    for i in range(len(now)):
                        for j in range(i + 1, len(now)):
                            if len(now[i].pot) < len(now[j].pot):
                                tmp = now[i]
                                now[i] = now[j]
                                now[j] = tmp
                            elif len(now[i].pot) == len(now[j].pot):
                                if now[i].loc['lat'] < now[j].loc['lat']:
                                    tmp = now[i]
                                    now[i] = now[j]
                                    now[j] = tmp
                    print('now after:', [len(_.pot) for _ in now], file=debug)
                    #
                    idx_c2p = []  # 记录与pp接近的nn编号
                    flag = 0  # 标记是否有重合度呈明显继承关系而直接继承pp的nn
                    for nn in range(len(now)):
                        if now[nn].nflag == -1:
                            overlap = len(set(past[pp].pot) & set(now[nn].pot))  # nn与pp潜在低涡中心点的重叠数量
                            overlap_pp = overlap / len(past[pp].pot)
                            overlap_nn = overlap / len(now[nn].pot)
                            print('overlap of past: %.2f' % overlap_pp, file=debug)
                            print('overlap of now: %.2f' % overlap_nn, file=debug)
                            if overlap_pp + overlap_nn >= 0.5:  # 若nn与pp重合度呈明显继承关系则直接继承
                                now[nn].nflag = past[pp].nflag
                                track[past[pp].nflag].append(now[nn])
                                print('now the best choice: %-3d (%6.3f %7.3f)' % (now[nn].nflag, now[nn].loc['lat'], now[nn].loc['lon']), file=debug)
                                flag = 1
                                break
                            else:  # 若nn与pp重合度不够则判断距离是否足够近
                                dist_closest = np.min([np.min(_) for _ in cdist(past[pp].pot, now[nn].pot)])
                                dist_center = np.sqrt((past[pp].loc['lon'] - now[nn].loc['lon']) ** 2 +
                                                      (past[pp].loc['lat'] - now[nn].loc['lat']) ** 2)
                                print('min distance and center distance to now  %-3d (%6.3f %7.3f): %.2f %.2f' %
                                      (now[nn].nflag, now[nn].loc['lat'], now[nn].loc['lon'], dist_closest, dist_center),
                                      file=debug)
                                if (dist_closest <= 2.0) and (dist_center <= 4.0):  # 与pp足够近的nn
                                    idx_c2p.append(nn)
                    if len(idx_c2p) != 0 and flag == 0:  # 两（多）个与pp足够近且均不明显继承于pp的nn
                        print('idx_c2p:', end=' ', file=debug)
                        for _ in idx_c2p:
                            print('(%6.3f %7.3f)' % (now[_].loc['lat'], now[_].loc['lon']), file=debug)
                        loc_c2p = [(now[_].loc['lat'], now[_].loc['lon']) for _ in idx_c2p]
                        idx_c2p_clst = DBSCAN(eps=4.0,
                                              min_samples=1,
                                              algorithm='auto').fit_predict(loc_c2p)  # 与pp接近的nn之间按接近程度分簇
                        print('idx_c2p_clst:', idx_c2p_clst, file=debug)
                        count = {}
                        for _ in idx_c2p_clst:
                            if _ not in count:
                                count[_] = 1
                            else:
                                count[_] += 1
                        max_value = np.max([_ for _ in count.values()])  # 与pp接近的nn之间按接近程度分簇的最多元素值
                        idx_c2n = []
                        for _ in count.keys():
                            if count[_] == max_value:
                                idx_c2n.append(_)
                        clst_c2n = {}  # 取与pp接近的nn之间按接近程度分簇的最多元素的簇
                        for i in idx_c2n:
                            these_idx = np.where(idx_c2p_clst == i)[0]
                            _lat = np.mean([now[idx_c2p[j]].loc['lat'] for j in these_idx])
                            _lon = np.mean([now[idx_c2p[j]].loc['lon'] for j in these_idx])
                            _pot = []
                            for _ in these_idx:
                                _pot = _pot + now[idx_c2p[_]].pot
                            clst_c2n[tuple(these_idx)] = Vortex(loc={'lat': _lat, 'lon': _lon}, time=time, pot=_pot)
                        print('clst_c2n:', end=' ', file=debug)
                        for _ in clst_c2n:
                            print('%s(%6.3f %7.3f)' % (_, clst_c2n[_].loc['lat'], clst_c2n[_].loc['lon']), file=debug)
                        min_dist_pn = 2.0
                        idx_best_clst = []
                        for idx_list in clst_c2n.keys():
                            dist = np.min([np.min(_) for _ in cdist(past[pp].pot, clst_c2n[idx_list].pot)])
                            if dist <= min_dist_pn:
                                min_dist_pn = dist
                                idx_best_clst.append(idx_list)  # 挑选出继承pp的nn聚合体
                        print('idx_best_clst', idx_best_clst, file=debug)
                        for _ in now:
                            print('now before changing: %-3d (%6.3f %7.3f)' % (_.nflag, _.loc['lat'], _.loc['lon']), file=debug)
                        # 根据潜在中心间的距离进一步选出最佳匹配簇
                        _lat = 0.0
                        _lon = 0.0
                        _pot = []
                        count = 0
                        for i in idx_best_clst:
                            for j in i:
                                count += 1
                                _lat += now[idx_c2p[j]].loc['lat']
                                _lon += now[idx_c2p[j]].loc['lon']
                                _pot = _pot + now[idx_c2p[j]].pot
                        _lat /= count
                        _lon /= count
                        best_choice = Vortex(loc={'lat': _lat, 'lon': _lon}, time=time, pot=_pot)
                        best_choice.nflag = past[pp].nflag
                        track[past[pp].nflag].append(best_choice)
                        # 将组成最佳继承聚合体的原始nn标记为待从now中剔除
                        discard = []
                        for nn in range(len(now)):
                            for i in idx_best_clst:
                                for j in i:
                                    if (now[nn].loc['lat'] == now[idx_c2p[j]].loc['lat']) and (now[nn].loc['lon'] == now[idx_c2p[j]].loc['lon']):
                                        discard.append(now[nn])
                                        print('now to be discarded: %-3d (%6.3f %7.3f)' % (now[nn].nflag, now[nn].loc['lat'], now[nn].loc['lon']), file=debug)
                        now = list(set(now) - set(discard))  # 将组成最佳继承聚合体的原始nn从now中剔除
                        now.append(best_choice)  # 将最佳继承聚合体添加到now中
                        print('now the best choice: %-3d (%6.3f %7.3f)' % (best_choice.nflag, best_choice.loc['lat'], best_choice.loc['lon']), file=debug)
                        for _ in now:
                            print('now after changing:  %-3d (%6.3f %7.3f)' % (_.nflag, _.loc['lat'], _.loc['lon']), file=debug)
                # 将未编号的潜在低涡中心点簇按最近距离归一化
                now = now_the_closests_to_one(now)
                # 再将无匹配的本时次低涡赋予新编号
                ng = 0
                for nn in now:
                    if nn.nflag == -1:
                        track[num + ng] = []
                        track[num + ng].append(nn)
                        nn.nflag = num + ng
                        ng = ng + 1
                num = num + ng
                for _ in now:
                    print('now without flag after:  %-3d (%6.3f %7.3f)' % (_.nflag, _.loc['lat'], _.loc['lon']), file=debug)

        # 将所有编号的低涡归入下一时次的past中
        past = []
        for nn in now:
            if nn.nflag != -1:
                past.append(nn)
        print('----------------------------------------------------', file=debug)
        # 将下一时次的past根据pot数量和生命史进行排序，pot数量越多、生命史长度越长者获得优先判断权
        print('next past before:', [(_.nflag, len(track[_.nflag]) ** 2, len(_.pot)) for _ in past], file=debug)
        for i in range(len(past)):
            for j in range(i + 1, len(past)):
                if (len(track[past[i].nflag]) ** 2) * len(past[i].pot) < (len(track[past[j].nflag]) ** 2) * len(past[j].pot):
                    tmp = past[i]
                    past[i] = past[j]
                    past[j] = tmp
                elif (len(track[past[i].nflag]) ** 2) * len(past[i].pot) == (len(track[past[j].nflag]) ** 2) * len(past[j].pot):
                    if past[i].loc['lat'] < past[j].loc['lat']:
                        tmp = past[i]
                        past[i] = past[j]
                        past[j] = tmp
        print('next past after:', [(_.nflag, len(track[_.nflag]) ** 2, len(_.pot)) for _ in past], '\n', file=debug)

    # 存储低涡路径及物理量信息
    def write_file():
        f = open(r'C:\Users\maizh\OneDrive\meteo_data\Data of my Thesis\vortex database\original data\%s.txt' % year, 'w')
        nflag = 0
        for k in track.keys():
            # if len(track[k]) >= 1 and is_in_qxp(k=track[k][0]) == 1 and track[k][0].time.month in [5, 6, 7, 8, 9]:
            if len(track[k]) >= 12 and is_in_qxp(k=track[k][0]) == 1 and track[k][0].time.month in [5, 6, 7, 8, 9]:
                f.write(str(nflag))
                f.write('\n')
                for i in track[k]:
                    f.write(str(nflag))  # 存储低涡编号[0]
                    f.write(',')
                    f.write('%.6f' % (i.loc['lat']))  # 存储低涡中心点坐标纬度[1]
                    f.write(',')
                    f.write('%.6f' % (i.loc['lon']))  # 存储低涡中心点坐标经度[2]
                    f.write(',')
                    f.write(str(i.time))  # 存储低涡时间[3]
                    f.write(',')
                    #
                    _time = i.time - datetime.timedelta(hours=8)  # 为了读取nc文件临时转换回UTC+0
                    _month = _time.month
                    _day = _time.day
                    _hour = _time.hour
                    _file_dir = r'D:\meteo_data\ERA5\500\%s%02d%02d.nc' % (year, _month, _day)
                    _file = xr.open_dataset(_file_dir)
                    _z = _file.variables['z'].data[_hour, :, :] / 9.8
                    for j in i.pot:
                        _m = mt.trans(coor=j[0], lat=lat)
                        _n = mt.trans(coor=j[1], lon=lon)
                        _pot_z = _z[_m, _n]
                        f.write(str(_pot_z))  # 存储潜在低涡中心点涡度值[4]
                        f.write(' ')  # 读的时候不要读最后一个空格（.split()并不会去掉它）
                    f.write(',')
                    for j in i.pot:
                        f.write(str(j[0]))  # 存储潜在低涡中心点坐标纬度[5]
                        f.write('-')
                        f.write(str(j[1]))  # 存储潜在低涡中心点坐标经度[5]
                        f.write(' ')
                    f.write('\n')
                nflag = nflag + 1
        f.close()

    time = None
    u = None
    v = None
    z = None
    num = 0
    past = []
    track = {}

    date_begin = datetime.date(args[0], begin[0], begin[1])
    date_end = datetime.date(args[0], end[0], end[1])
    delta_day = (date_end - date_begin).days + 1

    # 根据多进程开启与否设置进度条显示效果
    if len(args) > 1:
        with args[1]:
            current = multi.current_process()._identity[0] - 1
            position = args[0] - min(list(year_range))
            pbar = tqdm(total=delta_day * 24,
                        desc='line: %02d  cpu: %02d  dealing with %s' % (position, current, args[0]),
                        ncols=100,
                        unit='hour',
                        position=position,
                        leave=False)
    else:
        position = args[0] - min(list(year_range))
        pbar = tqdm(total=delta_day * 24,
                    desc='dealing with %s' % args[0],
                    ncols=100,
                    unit='hour',
                    position=position,
                    leave=False)

    debug = open(r'C:\Users\maizh\OneDrive\meteo_data\Data of my Thesis\vortex database\original data\debug\%s.txt' % args[0], 'w')

    year = date_begin.year
    # 循环主体
    for _ in range(delta_day):
        date = date_begin + datetime.timedelta(days=_)
        month = date.month
        day = date.day
        if any(m in str(month) for m in monlist):
            for hour in range(0, 24, 1):
                read_data()
                trace_track()
                if len(args) > 1:
                    with args[1]:
                        pbar.update(1)
                else:
                    pbar.update(1)
        else:
            continue

    if len(args) > 1:
        with args[1]:
            pbar.close()

    write_file()


if __name__ == '__main__':
    time_start = tm.time()

    check_missing()

    if year_range.stop - year_range.start > 1:
        multi.freeze_support()
        lock = multi.Manager().Lock()  # 使用锁来确保一次只有一个进程打印到标准输出
        with multi.Pool(multi.cpu_count()) as pool:
            for yy in year_range:
                pool.apply_async(func=main, args=(yy, lock))
            pool.close()
            pool.join()
    else:
        for yy in year_range:
            main(yy)

    time_end = tm.time()
    print('totally cost = %.4fs' % (time_end - time_start))
