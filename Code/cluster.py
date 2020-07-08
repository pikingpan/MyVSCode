import numpy as np
import os
import matplotlib.pyplot as plt
import math
import random
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029
NP = 100
D = 30
F = 0.5
CR = 0.8
Generation = 30*D
Max = 100
Min = 0
def caldist(populations):
    NP_num = len(populations)
    D_num = len(populations[0])
    dist = np.zeros(NP_num)
    for i in range(NP_num):
        dist_1=0
        for j in range(NP_num):
            dist_1=dist_1+pow((populations[i][j]-populations[i+1][j]),2)
        dist[i] = dist_1
    return dist

def initialtion():
    init_list = [[0]*D for i in range(NP)]
    for i in range(0, NP):
        for j in range(0, D):
            init_list[i][j] = random.uniform(Min, Max)
    return init_list

initlist=initialtion()
dist = caldist(initlist)
print(dist)
