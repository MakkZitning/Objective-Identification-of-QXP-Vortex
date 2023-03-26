import numpy as np
from sklearn.cluster import DBSCAN


def clustering(data, eps, min_samples):
    clst = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto').fit_predict(data)
    cluster = [[] for _ in range(np.max(clst) + 1)]
    # 剔除未聚类的噪点
    for i in range(len(clst)):
        if clst[i] != -1:
            cluster[clst[i]].append(data[i])
    # 求低涡簇的质心
    center = [[] for _ in range(len(cluster))]
    for i in range(len(cluster)):
        llat = 0
        llon = 0
        for j in range(len(cluster[i])):
            llat += cluster[i][j][0]
            llon += cluster[i][j][1]
        center[i] = (llat / len(cluster[i]), llon / len(cluster[i]))
    return cluster, center
