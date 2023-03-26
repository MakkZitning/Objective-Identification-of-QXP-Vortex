import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


def clustering(data, eps, min_samples, t_less, t_cdist, t_disp):
    # 使用DBSCAN进行聚类
    clst = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto').fit_predict(data)
    cluster = [[] for _ in range(np.max(clst) + 1)]
    # 剔除未聚类的噪点
    for i in range(len(clst)):
        if clst[i] != -1:
            cluster[clst[i]].append(data[i])
    # 预先剔除元素过少的簇
    tmp = []
    for i in cluster:
        if len(i) >= t_less:
            tmp.append(i)
    cluster = tmp
    # 进入整理簇阶段
    idx_comb = []  # 构成聚合簇各子簇的标签
    clst_disp = []
    # 将较接近的簇对其标签进行分类
    # 离散簇(cluster dispersed)：与其他所有簇的最短距离大于t_cdist
    # 聚合簇(cluster combined)：与个别簇的最短距离小于t_cdist
    for i in range(len(cluster)):
        flag = 0
        for j in range(i + 1, len(cluster)):
            if np.min(cdist(cluster[i], cluster[j])) <= t_cdist:
                idx_comb.append([i, j])
                flag = 1
        if flag == 0:
            if True not in [i in k for k in idx_comb]:
                clst_disp.append(cluster[i])
    # 聚合簇标签的去重
    for i in range(len(idx_comb)):
        flag = 0
        for j in range(i + 1, len(idx_comb)):
            if True in [k in idx_comb[j] for k in idx_comb[i]]:
                idx_comb[j] = list(set(idx_comb[i] + idx_comb[j]))
                flag = 1
        if flag == 1:
            idx_comb[i] = []
    idx_comb = [i for i in idx_comb if i != []]
    # 按聚合簇标签对聚合簇进行整合
    clst_comb = [[] for _ in range(len(idx_comb))]  # 记录聚合簇
    for i in range(len(clst_comb)):
        for j in idx_comb[i]:
            clst_comb[i] = clst_comb[i] + cluster[j]
    # 剔除元素过少的离散簇
    tmp = []
    for i in clst_disp:
        if len(i) >= t_disp:
            tmp.append(i)
    clst_disp = tmp
    # 将聚合簇与离散簇整合为输出聚类结果
    cluster = clst_comb + clst_disp
    # 求各簇的质心
    center = []
    for i in range(len(cluster)):
        llat = 0
        llon = 0
        for j in range(len(cluster[i])):
            llat += cluster[i][j][0]
            llon += cluster[i][j][1]
        center.append((llat / len(cluster[i]), llon / len(cluster[i])))
    return cluster, center
