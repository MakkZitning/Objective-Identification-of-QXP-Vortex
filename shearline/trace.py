import math
import numba as nb
from scipy.spatial.distance import cdist

from metools import metools as mt


# 返回各簇的端点坐标以及簇内元素距离矩阵
def find_endp(cluster):
    endp = [[] for _ in range(len(cluster))]
    dist = []
    dens = [[] for _ in range(len(cluster))]
    for i in range(len(cluster)):
        # 记录每个簇内元素间的距离
        dist.append(cdist(cluster[i], cluster[i]))
        # 记录每个簇内元素的局部密度（即与某点距离小于半径r的点的个数）
        dens[i] = [[] for _ in range(len(cluster[i]))]
        for j in range(len(cluster[i])):
            dens[i][j] = len([rr for rr in dist[i][j] if rr <= 0.5])
        # 记录端点
        endp_idx = [k for k in range(len(dens[i])) if dens[i][k] == 2]
        endp[i] = [cluster[i][k] for k in endp_idx]
    return endp, dist


@nb.jit(nopython=True, cache=True)
# 根据三点坐标计算所成角度
def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def trace(cluster):
    endp, dist = find_endp(cluster)

    # 开始追踪
    # 追踪要点
    ## 1.从端点开始
    ## 2.第二个追踪点须和起始点位于同一子簇内
    ## 3.第三个追踪点起需满足如下要素：
    ### 3.1.与前两个追踪点所成角度小于一定范围
    ### 3.2.与前一个追踪点的距离在一定范围内
    ## 4.当无满足点时，结束一次追踪
    ## 5.每个端点追踪出条线，取最长者

    wA = 0.6
    wB = 3.0
    r_next = 1
    r_angle = 75
    r_angle2 = 90

    shearline_idx = [[] for _ in range(len(cluster))]  # 总的切变线序列，每个簇一个
    length = []
    for i in range(len(cluster)):
        sequ = [[] for _ in range(len(endp[i]))]  # 对每个簇中的每个端点记录一次追踪序列
        for j in range(len(endp[i])):
            sequ[j].append(cluster[i].index(endp[i][j]))  # 存储端点序号为起始点
            # 开始追踪
            while True:
                eva = [0 for _ in range(len(cluster[i]))]  # 存储每个待追踪点的判定值
                for k in range(len(cluster[i])):  # 计算每个待追踪点与当前点的追踪系数（eva函数）
                    if len(sequ[j]) >= 2:  # 若序列长度达到2，则开始考虑角度
                        p_past = cluster[i][sequ[j][-2]]
                        p_this = cluster[i][sequ[j][-1]]
                        p_next = cluster[i][k]
                        past_this = p_this + p_past
                        this_next = p_next + p_this
                        angl = angle(past_this, this_next)
                        if angl < 1:
                            angl = 1
                        if k not in sequ[j] and dist[i][sequ[j][-1]][k] <= r_next and angl <= r_angle:
                            eva[k] = wA / (dist[i][sequ[j][-1]][k] ** 2) + wB / angl
                    else:  # 若刚开始追踪，只考虑距离
                        if k not in sequ[j] and dist[i][sequ[j][-1]][k] <= r_next:
                            eva[k] = wA / (dist[i][sequ[j][-1]][k] ** 2)
                if len(set(eva)) != 1:
                    idx_next = eva.index(max(eva))
                    sequ[j].append(idx_next)
                else:
                    break
        # 计算序列逐点距离之和
        ll = [0 for _ in range(len(sequ))]
        if len(ll) != 0:
            for k in range(len(sequ)):
                for kk in range(len(sequ[k]) - 1):
                    ll[k] = ll[k] + mt.dist(x1=cluster[i][sequ[k][kk]][0], x2=cluster[i][sequ[k][kk + 1]][0],
                                            y1=cluster[i][sequ[k][kk]][1], y2=cluster[i][sequ[k][kk + 1]][1])
            k = ll.index(max(ll))
            length.append(max(ll))
            shearline_idx[i] = sequ[k]  # 选取最长的序列作为该簇的切变线

    shearline = [[] for _ in range(len(shearline_idx))]
    for i in range(len(shearline)):
        for j in shearline_idx[i]:
            shearline[i].append((cluster[i][j][0], cluster[i][j][1]))

    # 按端点距离和整体走向连接不同簇的切变线
    # for i in range(len(shearline_idx)):
    #     for j in range(i + 1, len(shearline_idx)):
    #         endp_i = [cluster[i][shearline_idx[i][0]], cluster[i][shearline_idx[i][-1]]]
    #         endp_j = [cluster[j][shearline_idx[j][0]], cluster[j][shearline_idx[j][-1]]]
    #         # 第i个切变线与第j个切变线端点匹配
    #         for ii in range(len(endp_i)):
    #             mindist = 10000
    #             flag = 0
    #             for jj in range(len(endp_j)):
    #                 dist_endp = mt.dist(x1=endp_i[ii][0], x2=endp_j[jj][0],
    #                                     y1=endp_i[ii][1], y2=endp_j[jj][1])
    #                 if dist_endp <= 4 and dist_endp <= mindist:
    #                     mindist = dist_endp
    #                     flag = (ii, jj)
    #             if flag != 0:
    #                 endp_i1 = endp_i[flag[0]]
    #                 endp_j1 = endp_j[flag[1]]
    #                 # 判断匹配端点分别是第i和第j个切变线的头（编号0）还是尾（编号-1）
    #                 if flag[0] == 0:  # i头
    #                     endp_i2 = cluster[i][shearline_idx[i][1]]
    #                     if flag[1] == 0:  # j头
    #                         endp_j2 = cluster[j][shearline_idx[j][1]]
    #                         segm_i21 = endp_i1 + endp_i2
    #                         segm_j21 = endp_j1 + endp_j2
    #                         segm_i1j1 = endp_j1 + endp_i1
    #                         segm_j1i1 = endp_i1 + endp_j1
    #                         angl_i21j = 180 - angle(segm_i21, segm_i1j1)
    #                         angl_j21i = 180 - angle(segm_j21, segm_j1i1)
    #                         print(i, j, flag, endp_i1, endp_j1, endp_i2, endp_j2, mindist, angl_i21j, angl_j21i)
    #                         if angl_i21j >= r_angle2 and angl_j21i >= r_angle2:
    #                             shearline[j] = list(reversed(shearline[j])) + shearline[i]
    #                             shearline[i] = []
    #                     elif flag[1] == 1:  # j尾
    #                         endp_j2 = cluster[j][shearline_idx[j][-2]]
    #                         segm_i21 = endp_i1 + endp_i2
    #                         segm_j21 = endp_j1 + endp_j2
    #                         segm_i1j1 = endp_j1 + endp_i1
    #                         segm_j1i1 = endp_i1 + endp_j1
    #                         angl_i21j = 180 - angle(segm_i21, segm_i1j1)
    #                         angl_j21i = 180 - angle(segm_j21, segm_j1i1)
    #                         print(i, j, flag, endp_i1, endp_j1, endp_i2, endp_j2, mindist, angl_i21j, angl_j21i)
    #                         if angl_i21j >= r_angle2 and angl_j21i >= r_angle2:
    #                             shearline[j] = shearline[j] + shearline[i]
    #                             shearline[i] = []
    #                 elif flag[0] == 1:  # i尾
    #                     endp_i2 = cluster[i][shearline_idx[i][-2]]
    #                     if flag[1] == 0:  # j头
    #                         endp_j2 = cluster[j][shearline_idx[j][1]]
    #                         segm_i21 = endp_i1 + endp_i2
    #                         segm_j21 = endp_j1 + endp_j2
    #                         segm_i1j1 = endp_j1 + endp_i1
    #                         segm_j1i1 = endp_i1 + endp_j1
    #                         angl_i21j = 180 - angle(segm_i21, segm_i1j1)
    #                         angl_j21i = 180 - angle(segm_j21, segm_j1i1)
    #                         print(i, j, flag, endp_i1, endp_j1, endp_i2, endp_j2, mindist, angl_i21j, angl_j21i)
    #                         if angl_i21j >= r_angle2 and angl_j21i >= r_angle2:
    #                             shearline[j] = shearline[i] + shearline[j]
    #                             shearline[i] = []
    #                     elif flag[1] == 1:  # j尾
    #                         endp_j2 = cluster[j][shearline_idx[j][-2]]
    #                         segm_i21 = endp_i1 + endp_i2
    #                         segm_j21 = endp_j1 + endp_j2
    #                         segm_i1j1 = endp_j1 + endp_i1
    #                         segm_j1i1 = endp_i1 + endp_j1
    #                         angl_i21j = 180 - angle(segm_i21, segm_i1j1)
    #                         angl_j21i = 180 - angle(segm_j21, segm_j1i1)
    #                         print(i, j, flag, endp_i1, endp_j1, endp_i2, endp_j2, mindist, angl_i21j, angl_j21i)
    #                         if angl_i21j >= r_angle2 and angl_j21i >= r_angle2:
    #                             shearline[j] = shearline[i] + list(reversed(shearline[j]))
    #                             shearline[i] = []
    return endp, shearline
