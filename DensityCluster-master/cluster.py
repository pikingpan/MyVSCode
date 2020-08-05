from math import cos
#from DensityCluster-master.get_rho import how_many_goptima
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import random
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
import matlab
import matlab.engine
from numpy.lib.twodim_base import tri
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029
NP = 100
D = 1
F = 0.5
CR = 0.8
Generation = 30*D
Max = 1
Min = 0
eng=matlab.engine.start_matlab()
#np.set_printoptions(threshold=1000)
def euldist(points):
    len_pts=len(points)
    #print('out loop',np.shape(points))
    dist=np.zeros((len_pts,len_pts))
    #欧氏距离
    for i in range(len_pts):
        for j in range(len_pts):
            #print('in loop',np.shape(points))
            dist[i,j]=np.sqrt(np.sum(np.square(points[i]-points[j])))
    #截断距离。选取2%
    dc=np.sort(np.concatenate(dist))[int(np.ceil(len_pts+0.005*len_pts*(len_pts-1)))]
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
    #print('calclusters',clusters)
    qc=np.copy(q)
    #print("qc",qc)
    for i in np.flipud(np.argsort(rho)):
        if i not in centers:
            if qc[i] not in centers:
                qc[i]=qc[qc[i]]
                #print('',centers.index(qc[i]))
            clusters[centers.index(qc[i])].append(i)
    return qc,clusters

def plot(rho,delta,gamma,points,clusters):
    for cluster in clusters:
        plt.scatter(points[cluster][:,0],points[cluster][:,1],color=np.random.rand(3))

def run(points,plotclusters=True):
    dist,dc=euldist(points)
    rho=calrho(dist,dc)
    delta,q=caldelta(rho,dist)
    gamma=rho*delta
    centers=calcenters(gamma)
    qc,clusters=calclusters(q,rho,centers)

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
    cost = eng.niching_func(matlab.double(init_list.tolist()),2)
    return cost


#evolution
def Evolution(init_list,cost):
    trial = [0]*D
    U = [[0]*D for i in range(NP)]
    #print("ex init_list",init_list)
    for i in range(0, NP):
        score = 0
        #mutate
        a = random.randint(0, len(init_list)-1)
        while a == i:
            a = random.randint(0, len(init_list)-1)
        b = random.randint(0, len(init_list)-1)
        while b == a | b == i:
            b = random.randint(0, len(init_list)-1)
        c = random.randint(0, len(init_list)-1)
        while c == a | c == b | c == i:
            c = random.randint(0, len(init_list)-1)
        j = random.randint(0, D-1)
        #print(a,b,c)
        for k in range(1, D+1):
            if((random.random() <= CR) or (k == D)):
                trial[j] = init_list[a][j] + F * \
                    (init_list[b][j] - init_list[c][j])
                if (trial[j]<=0):
                    trial[j] = 0
                elif(trial[j]>=1):
                    trial[j]=1
            else:
                trial[j] = init_list[i][j]
            j = (j+1) % D
        score = score + F2(trial)
        if(score >= cost[i]):
            for j in range(0, D):
                U[i][j] = trial[j]
            cost[i] = score
        else:
            for j in range(0, D):
                U[i][j] = init_list[i][j]
    for i in range(0, NP):
        for j in range(0, D):
            #print(i,j)
            init_list[i][j] = U[i][j]
    #print('after init_list',init_list)
    return init_list, cost


def delete_zero(clusters):
    X_clu = np.array([[0]*NP]*NP)
    x = []
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            X_clu[i][j] = clusters[i][j]
        x.append(np.trim_zeros(X_clu[i]))
    return np.array(x) 

def add_element(init_list,clusters):
    X_index = [[0]*NP]*NP
    x = []
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            X_index[i][j] = init_list[clusters[i][j]]
        x.extend(X_index[i][0:len(clusters[i])])
    return np.array(x)


if __name__=='__main__':
    init_list = np.array(initialtion())
    y = []
    t = [0]*NP
    t = Cost(init_list)
    cost = np.array(t)
    #print(t)
    y.append(max(cost))
    for g in range(Generation):
        clusters,centers = run(init_list)
        print('the cluster after ',clusters,' and generation',g)
        init_list = add_element(init_list,clusters)
        init_list,cost=Evolution(init_list,cost)
        y.append(max(cost))

    temp = matlab.double(init_list.tolist())
    #no=eng.get_fgoptima(2)
    #print("the NP",temp)
    count = eng.count_goptima(temp,2,0.1,nargout=2)
    print('goptimation ',count)
    print('%.3f' % max(y))
    #print('cost',len(y))