import numba as nb


@nb.jit(nopython=True, cache=True)
# 外伸悬臂法判断闭合气旋性低压中心 #
def arms(u, v, z, m, n, r_s, dtz_s, t1_s, t2_s, t3_s, r_x, dtz_x, t1_x, t2_x, t3_x):
    # 小尺度低涡
    r_s = r_s * 4  # 每条旋臂向外遍历r+1次
    outer_z_s = z[m, n] + dtz_s  # 要求达到的闭合等高线值
    k1_s = 0  # 达标旋臂数（高度值递增）
    k2_s = 0  # 达标旋臂数（闭合等高线）
    k3_s = 0  # 达标旋臂数（气旋式风场）
    # 大尺度低涡
    r_x = r_x * 4  # 每条旋臂向外遍历r+1次
    outer_z_x = z[m, n] + dtz_x  # 要求达到的闭合等高线值
    k1_x = 0  # 达标旋臂数（高度值递增）
    k2_x = 0  # 达标旋臂数（闭合等高线）
    k3_x = 0  # 达标旋臂数（气旋式风场）

    # 东向延申
    last = 0
    c_s = 0
    c_x = 0
    d_s = 0
    d_x = 0
    avgv_s = 0
    avgv_x = 0
    for i in range(m, m + 1):
        for j in range(n, n + r_x + 1):
            this = z[i, j]
            if j <= n + r_s:
                if last < this:
                    c_s += 1
                    c_x += 1
                if outer_z_s <= this:
                    d_s = 1
                if outer_z_x <= this:
                    d_x = 1
                avgv_s += v[i, j]
                avgv_x += v[i, j]
            elif n + r_s + 1 <= j <= n + r_x:
                if last < this:
                    c_x += 1
                if outer_z_x <= this:
                    d_x = 1
                avgv_x += v[i, j]
            last = this
    if c_s >= r_s * 0.55:
        k1_s += 1
    if c_x >= r_x * 0.55:
        k1_x += 1
    if d_s == 1:
        k2_s += 1
    if d_x == 1:
        k2_x += 1
    if avgv_s >= 0:
        k3_s += 1
    if avgv_x >= 0:
        k3_x += 1

    # 南向延申
    last = 0
    c_s = 0
    c_x = 0
    d_s = 0
    d_x = 0
    avgu_s = 0
    avgu_x = 0
    for i in range(m, m + r_x + 1):
        for j in range(n, n + 1):
            this = z[i, j]
            if i <= m + r_s:
                if last < this:
                    c_s += 1
                    c_x += 1
                if outer_z_s <= this:
                    d_s = 1
                if outer_z_x <= this:
                    d_x = 1
                avgu_s += u[i, j]
                avgu_x += u[i, j]
            elif m + r_s + 1 <= i <= m + r_x:
                if last < this:
                    c_x += 1
                if outer_z_x <= this:
                    d_x = 1
                avgu_x += u[i, j]
            last = this
    if c_s >= r_s * 0.55:
        k1_s += 1
    if c_x >= r_x * 0.55:
        k1_x += 1
    if d_s == 1:
        k2_s += 1
    if d_x == 1:
        k2_x += 1
    if avgu_s >= 0:
        k3_s += 1
    if avgu_x >= 0:
        k3_x += 1

    # 西向延申
    last = 0
    c_s = 0
    c_x = 0
    d_s = 0
    d_x = 0
    avgv_s = 0
    avgv_x = 0
    for i in range(m, m + 1):
        for j in range(n, n - r_x - 1, -1):
            this = z[i, j]
            if j >= n - r_s:
                if last < this:
                    c_s += 1
                    c_x += 1
                if outer_z_s <= this:
                    d_s = 1
                if outer_z_x <= this:
                    d_x = 1
                avgv_s += v[i, j]
                avgv_x += v[i, j]
            elif n - r_x <= j <= n - r_s - 1:
                if last < this:
                    c_x += 1
                if outer_z_x <= this:
                    d_x = 1
                avgv_x += v[i, j]
            last = this
    if c_s >= r_s * 0.55:
        k1_s += 1
    if c_x >= r_x * 0.55:
        k1_x += 1
    if d_s == 1:
        k2_s += 1
    if d_x == 1:
        k2_x += 1
    if avgv_s <= 0:
        k3_s += 1
    if avgv_x <= 0:
        k3_x += 1

    # 北向延申
    last = 0
    c_s = 0
    c_x = 0
    d_s = 0
    d_x = 0
    avgu_s = 0
    avgu_x = 0
    for i in range(m, m - r_x - 1, -1):
        for j in range(n, n + 1):
            this = z[i, j]
            if i >= m - r_s:
                if last < this:
                    c_s += 1
                    c_x += 1
                if outer_z_s <= this:
                    d_s = 1
                if outer_z_x <= this:
                    d_x = 1
                avgu_s += u[i, j]
                avgu_x += u[i, j]
            elif m - r_x <= i <= m - r_s - 1:
                if last < this:
                    c_x += 1
                if outer_z_x <= this:
                    d_x = 1
                avgu_x += u[i, j]
            last = this
    if c_s >= r_s * 0.55:
        k1_s += 1
    if c_x >= r_x * 0.55:
        k1_x += 1
    if d_s == 1:
        k2_s += 1
    if d_x == 1:
        k2_x += 1
    if avgu_s <= 0:
        k3_s += 1
    if avgu_x <= 0:
        k3_x += 1

    # 东北向延申
    last = 0
    c_s = 0
    c_x = 0
    d_s = 0
    d_x = 0
    avgu_s = 0
    avgu_x = 0
    avgv_s = 0
    avgv_x = 0
    for i in range(0, r_x + 1):
        this = z[m - i, n + i]
        if i <= r_s:
            if last < this:
                c_s += 1
                c_x += 1
            if outer_z_s <= this:
                d_s = 1
            if outer_z_x <= this:
                d_x = 1
            avgu_s += u[m - i, n + i]
            avgu_x += u[m - i, n + i]
            avgv_s += v[m - i, n + i]
            avgv_x += v[m - i, n + i]
        elif r_s + 1 <= i <= r_x:
            if last < this:
                c_x += 1
            if outer_z_x <= this:
                d_x = 1
            avgu_x += u[m - i, n + i]
            avgv_x += v[m - i, n + i]
        last = this
    if c_s >= r_s * 0.55:
        k1_s += 1
    if c_x >= r_x * 0.55:
        k1_x += 1
    if d_s == 1:
        k2_s += 1
    if d_x == 1:
        k2_x += 1
    if not (avgu_s >= 0 and avgv_s <= 0):
        k3_s += 1
    if not (avgu_x >= 0 and avgv_x <= 0):
        k3_x += 1

    # 东南向延申
    last = 0
    c_s = 0
    c_x = 0
    d_s = 0
    d_x = 0
    avgu_s = 0
    avgu_x = 0
    avgv_s = 0
    avgv_x = 0
    for i in range(0, r_x + 1):
        this = z[m + i, n + i]
        if i <= r_s:
            if last < this:
                c_s += 1
                c_x += 1
            if outer_z_s <= this:
                d_s = 1
            if outer_z_x <= this:
                d_x = 1
            avgu_s += u[m + i, n + i]
            avgu_x += u[m + i, n + i]
            avgv_s += v[m + i, n + i]
            avgv_x += v[m + i, n + i]
        elif r_s + 1 <= i <= r_x:
            if last < this:
                c_x += 1
            if outer_z_x <= this:
                d_x = 1
            avgu_x += u[m + i, n + i]
            avgv_x += v[m + i, n + i]
        last = this
    if c_s >= r_s * 0.55:
        k1_s += 1
    if c_x >= r_x * 0.55:
        k1_x += 1
    if d_s == 1:
        k2_s += 1
    if d_x == 1:
        k2_x += 1
    if not (avgu_s <= 0 and avgv_s <= 0):
        k3_s += 1
    if not (avgu_x <= 0 and avgv_x <= 0):
        k3_x += 1

    # 西南向延申
    last = 0
    c_s = 0
    c_x = 0
    d_s = 0
    d_x = 0
    avgu_s = 0
    avgu_x = 0
    avgv_s = 0
    avgv_x = 0
    for i in range(0, r_x + 1):
        this = z[m + i, n - i]
        if i <= r_s:
            if last < this:
                c_s += 1
                c_x += 1
            if outer_z_s <= this:
                d_s = 1
            if outer_z_x <= this:
                d_x = 1
            avgu_s += u[m + i, n - i]
            avgu_x += u[m + i, n - i]
            avgv_s += v[m + i, n - i]
            avgv_x += v[m + i, n - i]
        elif r_s + 1 <= i <= r_x:
            if last < this:
                c_x += 1
            if outer_z_x <= this:
                d_x = 1
            avgu_x += u[m + i, n - i]
            avgv_x += v[m + i, n - i]
        last = this
    if c_s >= r_s * 0.55:
        k1_s += 1
    if c_x >= r_x * 0.55:
        k1_x += 1
    if d_s == 1:
        k2_s += 1
    if d_x == 1:
        k2_x += 1
    if not (avgu_s <= 0 and avgv_s >= 0):
        k3_s += 1
    if not (avgu_x <= 0 and avgv_x >= 0):
        k3_x += 1

    # 西北向延申
    last = 0
    c_s = 0
    c_x = 0
    d_s = 0
    d_x = 0
    avgu_s = 0
    avgu_x = 0
    avgv_s = 0
    avgv_x = 0
    for i in range(0, r_x + 1):
        this = z[m - i, n - i]
        if i <= r_s:
            if last < this:
                c_s += 1
                c_x += 1
            if outer_z_s <= this:
                d_s = 1
            if outer_z_x <= this:
                d_x = 1
            avgu_s += u[m - i, n - i]
            avgu_x += u[m - i, n - i]
            avgv_s += v[m - i, n - i]
            avgv_x += v[m - i, n - i]
        elif r_s + 1 <= i <= r_x:
            if last < this:
                c_x += 1
            if outer_z_x <= this:
                d_x = 1
            avgu_x += u[m - i, n - i]
            avgv_x += v[m - i, n - i]
        last = this
    if c_s >= r_s * 0.55:
        k1_s += 1
    if c_x >= r_x * 0.55:
        k1_x += 1
    if d_s == 1:
        k2_s += 1
    if d_x == 1:
        k2_x += 1
    if not (avgu_s >= 0 and avgv_s >= 0):
        k3_s += 1
    if not (avgu_x >= 0 and avgv_x >= 0):
        k3_x += 1

    # 判断是否满足以上条件
    if (k1_s >= t1_s and k2_s >= t2_s and k3_s >= t3_s) or (k1_x >= t1_x and k2_x >= t2_x and k3_x >= t3_x):  # 设定双重标准r
    # if k1_x >= t1_x and k2_x >= t2_x and k3_x >= t3_x:  # 只设定一个r
        return 1
    else:
        return 0


# 遍历函数 #
def traversal(mu, md, nl, nr,
              u, v, z,
              lat, lon,
              r_s, dtz_s, t1_s, t2_s, t3_s,
              r_x, dtz_x, t1_x, t2_x, t3_x):
    # 识别潜在低涡中心点并返回其坐标
    # 判识要求：
    # 1.低涡中心强度（位势高度值不高于5870gpm）
    # 2.正涡度（没用上）
    # 3.存在闭合等高线
    # 4.存在气旋式风场
    # r为经纬度，1°=4格
    pot = []
    for m in range(mu, md + 1):
        for n in range(nl, nr + 1):
            if z[m, n] <= 5870:
                if arms(u=u, v=v, z=z,
                        m=m, n=n,
                        r_s=r_s, dtz_s=dtz_s, t1_s=t1_s, t2_s=t2_s, t3_s=t3_s,
                        r_x=r_x, dtz_x=dtz_x, t1_x=t1_x, t2_x=t2_x, t3_x=t3_x) == 1:
                    pot.append((lat[m][n], lon[m][n]))
    return pot
