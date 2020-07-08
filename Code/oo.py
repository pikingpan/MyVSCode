import numpy as np
import os

def euldist(points):
    len_pts=len(points)
    dist=np.zeros((len_pts,len_pts))
    #欧氏距离
    for i in range(len_pts):
        for j in range(len_pts):
            dist[i,j]=np.sqrt(np.sum(np.square(points[i]-points[j])))
    #截断距离。选取2%
    dc=np.sort(np.concatenate(dist))[int(np.ceil(len_pts+0.02*len_pts*(len_pts-1)))]
    return dist,dc

name_sets = 'C:\\Users\\Evil\\Desktop\\mycode\\DensityCluster-master\\D31.txt'
# path_sets = os.path.join(os.path.expanduser('~'), 'DataSets/Cluster/Shape-sets', name_sets)
path_sets=name_sets
data = np.loadtxt(path_sets)
points = data[:, 0:2]
print(points)
dist = euldist(points)
print(dist)