import numpy as np
import os
import matplotlib.pyplot as plt
import math
import random
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029
NP = 100
D = 1
F = 0.5
CR = 0.8
Generation = 30*D
Max = 1
Min = 0
#np.set_printoptions(threshold=1000)
def euldist(points):
    len_pts=len(points)
    dist=np.zeros((len_pts,len_pts))
    #欧氏距离
    for i in range(len_pts):
        for j in range(len_pts):
            dist[i,j]=np.sqrt(np.sum(np.square(points[i]-points[j])))
    #截断距离。选取2%
    dc=np.sort(np.concatenate(dist))[int(np.ceil(len_pts+0.01*len_pts*(len_pts-1)))]
    return dist,dc

def calrho(dist,dc):
    len_rho=len(dist)
    rho=np.zeros(len_rho)
    for i in range(len_rho):
        # rho[i]=np.sum(dist[i,:]<dc)
        # 高斯核中计算局部密度(都是等于截断距离dc内点i个数)
        rho[i]=np.sum(np.exp(-np.square(dist[i,:]/dc)))
    return rho

#从点i到具有更高局部密度的点的最小距离
def caldelta(rho,dist):
    len_delta=len(rho)
    delta=np.ones(len_delta)*np.inf
    q=np.arange(len_delta)
    for i in range(len_delta):
        for j in range(len_delta):
            #δi是从点i到具有更高局部密度的点的最小距离
            if (rho[j]>rho[i])&(dist[i,j]<delta[i]):
               delta[i]=dist[i,j]
               q[i]=j
    indexmax=np.argmax(delta)
    delta[indexmax]=dist[indexmax,:].max()
    return delta,q

def calcenters(gamma):
    x = np.flipud(np.argsort(gamma))
    y = np.flipud(np.sort(gamma))
    gamma_mean=gamma.mean()   #返回平均数
    centers=[x[0],x[1]]
    for i in range(2, len(y) - 1):
        # if y[i] - y[i + 1] < (y[i - 1] - y[i]) / 2.:
        #     break
        if y[i]-gamma_mean < y[i-1]-y[i]:
            break
        centers.append(x[i])
    return centers

def calclusters(q,rho,centers):
    clusters=np.array(centers).reshape(-1,1).tolist()
    qc=np.copy(q)
    for i in np.flipud(np.argsort(rho)):
        if i not in centers:
            if qc[i] not in centers:
                qc[i]=qc[qc[i]]
                clusters[centers.index(qc[i])].append(i)
    return qc,clusters

def plot(rho,delta,gamma,points,clusters):
    # plt.figure(figsize=(12,18))
    # plt.subplot(221)
    # plt.scatter(rho,delta,color='k')
    # plt.xlabel(r'$\rho$')
    # plt.ylabel(r'$\delta$')
    # plt.subplot(222)
    # plt.scatter(np.arange(len(gamma)),np.sort(gamma,),color='r')
    # plt.ylabel('r$\gamma$')
    # plt.subplot(223)
    for cluster in clusters:
        plt.scatter(points[cluster][:,0],points[cluster][:,1],color=np.random.rand(3))

def run(points,plotclusters=True):
    dist,dc=euldist(points)
    #print('dc',dc)
    #print('points length',len(dist))
    rho=calrho(dist,dc)
    #print('rho',rho)
    delta,q=caldelta(rho,dist)
    #print('detal q',delta,q)
    gamma=rho*delta
    #print('gamma',gamma)
    centers=calcenters(gamma)
    #print('center',(centers))
    qc,clusters=calclusters(q,rho,centers)
    #print('clusters',clusters)
    #print('some points',(points[clusters[0][0]]))
    #print('points',points)
    #if plotclusters:
        #plot(rho,delta,gamma,points,clusters)
        #plt.show()
    return clusters,centers

def initialtion():
    init_list = [[0]*D for i in range(NP)]
    for i in range(0, NP):
        for j in range(0, D):
            init_list[i][j] = random.uniform(Min, Max)
    return init_list

def F2(init_list):
    f=pow(math.sin(5*PI*init_list[0]),6)
    return f

def Cost(init_list):
    cost = [0]*NP
    f=1.0
    for i in range(0, NP):
        cost[i] = cost[i] + F2(init_list[i])
    return cost


#evolution
def Evolution(init_list, cost):
    trial = [0]*D
    U = [[0]*D for i in range(NP)]
    for i in range(0, NP):
        score = 0
        #mutate
        a = random.randint(0, NP-1)
        while a == i:
            a = random.randint(0, NP-1)
        b = random.randint(0, NP-1)
        while b == a | b == i:
            b = random.randint(0, NP-1)
        c = random.randint(0, NP-1)
        while c == a | c == b | c == i:
            c = random.randint(0, NP-1)
        j = random.randint(0, D-1)
        for k in range(1, D+1):
            if(random.random() <= CR) | (k == D):
                trial[j] = init_list[a][j] + F * \
                    (init_list[b][j] - init_list[c][j])
            else:
                trial[j] = init_list[i][j]
            j = (j+1) % D
        f = 1.0
        score = score + F2(trial)
        if(score <= cost[i]):
            for j in range(0, D):
                U[i][j] = trial[j]
            cost[i] = score
        else:
            for j in range(0, D):
                U[i][j] = init_list[i][j]
    for i in range(0, NP):
        for j in range(0, D):
            init_list[i][j] = U[i][j]
    return init_list, cost

if __name__=='__main__':
    #name_sets = 'C:\\Users\\Evil\\Desktop\\mycode\\DensityCluster-master\\D31.txt'
    # path_sets = os.path.join(os.path.expanduser('~'), 'DataSets/Cluster/Shape-sets', name_sets)
    #path_sets=name_sets
    #data = np.loadtxt(path_sets)
    #points = data[:, 0:2]
    init_list = np.array(initialtion())
    #print('point',points)
    #print(len(init_list[0]))
    y = []
    clusters,centers = run(init_list)
    print('ex caculate ',clusters)
    cost = Cost(init_list)
    y.append(min(cost))
    for g in range(Generation):
        init_list,cost=Evolution(init_list,cost)
        #clusters,centers = run(init_list)
        y.append(min(cost))
    clusters,centers = run(init_list)
    af_clu = init_list[clusters[0]]
    print('cluster.',clusters)
    print('cluster after',af_clu)
    #X_clu = [[0]*NP]*NP
    #for i in range(len(clusters)):
        #for j in range(len(clusters[i])):
            #X_clu[i][j] = clusters[i][j]
    #position = X_clu
    #fitness=Shere(init_list[X_clu[0][0]])
    #cost = Cost(init_list)
    #print('cost',cost)