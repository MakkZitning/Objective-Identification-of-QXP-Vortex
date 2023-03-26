import numba as nb
import numpy as np


@nb.jit(nopython=True, cache=True)
def preprocess(u, v):
    u = np.where(u == 0, u - 0.00001, u)
    v = np.where(v == 0, v - 0.00001, v)
    wind = np.sqrt(u * u + v * v)
    return u, v, wind


@nb.jit(nopython=True, cache=True)
def wind_dir(u, v):
    wd = 0
    if u > 0 and v > 0:  # 东北象限(西南风)：0°~90°
        wd = np.arctan(abs(u) / abs(v)) * 180 / np.pi
    elif u > 0 and v < 0:  # 东南象限(西北风)：90°~180°
        wd = 180 - np.arctan(abs(u) / abs(v)) * 180 / np.pi
    elif u < 0 and v < 0:  # 西南象限(东北风)：180°~270°
        wd = 180 + np.arctan(abs(u) / abs(v)) * 180 / np.pi
    elif u < 0 and v > 0:  # 西北象限(东南风)：270°~360°
        wd = 360 - np.arctan(abs(u) / abs(v)) * 180 / np.pi
    return wd


@nb.jit(nopython=True, cache=True)
def locate(y, x, flag, lat, lon, u, v):
    llon = 0
    llat = 0
    if flag == 0:
        if v[y, x + 1] - v[y, x] == 0.0:
            v[y, x + 1] += 0.00001
        llon = lon[y, x] - v[y, x] * (lon[y, x + 1] - lon[y, x]) / (v[y, x + 1] - v[y, x])
        llat = lat[y, x]
    elif flag == 1:
        if u[y + 1, x] - u[y, x] == 0.0:
            u[y + 1, x] += 0.00001
        llon = lon[y, x]
        llat = lat[y, x] - u[y, x] * (lat[y + 1, x] - lat[y, x]) / (u[y + 1, x] - u[y, x])
    return llon, llat


@nb.jit(nopython=True, cache=True)
def traversal(mu, md, nl, nr, u, v, lat, lon, r, t_wind, t_angl):
    u, v, wind = preprocess(u, v)

    pot = []
    pot_type = []

    t_uv = 0  # 两侧风速达到此阈值（通常为+-0）则开始判断

    for m in range(mu, md):
        for n in range(nl, nr):
            u_ll = 0
            v_ll = 0
            u_rr = 0
            v_rr = 0
            u_uu = 0
            v_uu = 0
            u_dd = 0
            v_dd = 0
            wind_ll = 0
            wind_rr = 0
            wind_uu = 0
            wind_dd = 0
            # 横向网格点：左侧经向风<0（北风），右侧经向风>0（南风）
            if (v[m, n] < t_uv) and (v[m, n + 1] > -t_uv):
                for i in range(n, n - r - 1, -1):
                    u_ll = u_ll + u[m, i]
                    v_ll = v_ll + v[m, i]
                    wind_ll = wind_ll + wind[m, i]
                for i in range(n + 1, n + r + 2, 1):
                    u_rr = u_rr + u[m, i]
                    v_rr = v_rr + v[m, i]
                    wind_rr = wind_rr + wind[m, i]
                u_ll = u_ll / (r + 1)
                v_ll = v_ll / (r + 1)
                wind_ll = wind_ll / (r + 1)
                u_rr = u_rr / (r + 1)
                v_rr = v_rr / (r + 1)
                wind_rr = wind_rr / (r + 1)
                # 进一步判定切变点
                if v_ll < 0 and v_rr > 0:
                    if ((u_ll > 0 and u_rr < 0) or (u_ll < 0 and u_rr > 0)) and t_wind <= wind_ll + wind_rr:
                        llon, llat = locate(y=m, x=n, flag=0, lat=lat, lon=lon, u=u, v=v)
                        pot.append((llat, llon))
                        pot_type.append('lime')
                    elif (u_ll > 0 and u_rr > 0) or (u_ll < 0 and u_rr < 0):
                        wdd_ll = wind_dir(u=u_ll, v=v_ll)
                        wdd_rr = wind_dir(u=u_rr, v=v_rr)
                        if abs(wdd_ll - wdd_rr) >= t_angl and t_wind <= wind_ll + wind_rr:
                            llon, llat = locate(y=m, x=n, flag=0, lat=lat, lon=lon, u=u, v=v)
                            pot.append((llat, llon))
                            pot_type.append('lime')
            # 竖向网格点：上侧纬向风<0（东风），下侧纬向风>0（西风）
            if (u[m, n] < t_uv) and (u[m + 1, n] > -t_uv):
                for i in range(m, m - r - 1, -1):
                    u_uu = u_uu + (u[i, n])
                    v_uu = v_uu + (v[i, n])
                    wind_uu = wind_uu + wind[i, n]
                for i in range(m + 1, m + r + 2, 1):
                    u_dd = u_dd + u[i, n]
                    v_dd = v_dd + v[i, n]
                    wind_dd = wind_dd + wind[i, n]
                u_uu = u_uu / (r + 1)
                v_uu = v_uu / (r + 1)
                wind_uu = wind_uu / (r + 1)
                u_dd = u_dd / (r + 1)
                v_dd = v_dd / (r + 1)
                wind_dd = wind_dd / (r + 1)
                # 进一步判定切变点
                if u_uu < 0 and u_dd > 0:
                    if ((v_uu > 0 and v_dd < 0) or (v_uu < 0 and v_dd > 0)) and t_wind <= wind_uu + wind_dd:
                        llon, llat = locate(y=m, x=n, flag=1, lat=lat, lon=lon, u=u, v=v)
                        pot.append((llat, llon))
                        pot_type.append('blue')
                    elif v_uu < 0 and v_dd < 0:
                        wdd_uu = wind_dir(u=u_uu, v=v_uu)
                        wdd_dd = wind_dir(u=u_dd, v=v_dd)
                        if abs(wdd_uu - wdd_dd) >= t_angl and t_wind <= wind_uu + wind_dd:
                            llon, llat = locate(y=m, x=n, flag=1, lat=lat, lon=lon, u=u, v=v)
                            pot.append((llat, llon))
                            pot_type.append('blue')
                    elif v_uu > 0 and v_dd > 0:
                        wdd_uu = wind_dir(u=u_uu, v=v_uu)
                        wdd_dd = wind_dir(u=u_dd, v=v_dd)
                        wdd_uu = 360 - wdd_uu
                        if wdd_uu + wdd_dd >= t_angl and t_wind <= wind_uu + wind_dd:
                            llon, llat = locate(y=m, x=n, flag=1, lat=lat, lon=lon, u=u, v=v)
                            pot.append((llat, llon))
                            pot_type.append('blue')
    return pot, pot_type
