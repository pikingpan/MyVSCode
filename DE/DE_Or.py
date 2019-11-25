import numpy as np
import matplotlib.pyplot as plt
import math
import random
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
NP = 100
D = 100
F = 0.5
CR = 0.8
Generation = 30*D
Max = 5.12
Min = 0


# Object function
def Sphere(x):
    fx = 0
    for i in range(0,D):
        fx = fx + x[i]**2
    return fx


def f7(x):
    fx = 0
    for i in range(0,D):
        fx = fx + i*x[i]**4 
    fx = fx + random.random()
    return fx 


def f2(x):
    fx_1 = 0
    fx_2 = 1
    for i in range(0,D):
        fx_1 = fx_1 + abs(x[i])
        fx_2 = fx_2 * x[i]
    return (fx_1 + fx_2)

def f3(x):
    fx_1 = 0
    fx_2 = 0
    for i in range(0,D):
        for j in range(0,i):
            fx_1 = fx_1 + x[j]
        fx_2 = fx_2 + fx_1**2
    return fx_2

#Rosenbrock functin
def f5(x):
    fx = 0
    for i in range(0,D):
        fx = fx + 100*((x[i]**2)-x[i+1])**2 + (x[i]-1)**2
    return fx

def f4(x):
    fx = 0
    t = [0]*D
    for i in range(0,D):
        t[i] = abs(x[i])
    fx = max(t)
    return fx
def f6(x):
    fx = 0
    for i in range(0,D):
        fx = fx + math.floor(x[i]+0.5)
    return fx
def f8(x):
    fx = 0
    for i in range(0,D):
        fx = fx + (-x[i])*math.sin(math.sqrt(abs(x[i])))
    return fx
def Rastrigin(x):
    fx = 0
    for i in range(0,D):
        fx = fx + ( x[i]**2 - 10*math.cos(2*math.pi*x[i]) + 10 )
    return fx
def f10(x):
    fx_1 = 0
    fx_2 = 0
    fx_3 = 0
    for i in range(0,D):
        fx_2 = fx_2 + x[i]**2
        fx_3 = fx_3 + math.cos(2*math.pi*x[i])
    fx_1 = -20 * math.exp((-0.2)*math.sqrt((1/D)*fx_2)) - math.exp((1/D)*fx_3) + 20 + math.exp(1)
    return fx_1

# Initial function
def initialtion():
    init_list = [[0]*D for i in range(NP)]
    for i in range(0, NP):
        for j in range(0, D):
            init_list[i][j] = random.uniform(Min, Max)
    return init_list


def Cost(init_list):
    cost = [0]*NP
    for i in range(0, NP):
        cost[i] = cost[i] + Rastrigin(init_list[i])
    return cost


#evolution
def Evolution(init_list,cost):
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
                trial[j] = init_list[a][j] + F * (init_list[b][j] - init_list[c][j])
            else:
                trial[j] = init_list[i][j]
            j = (j+1) % D

        score = score + Rastrigin(trial)
        if(score<=cost[i]):
            for j in range(0,D):
                U[i][j] = trial[j]
            cost[i]=score
        else:
            for j in range(0,D):
                U[i][j] = init_list[i][j]
    for i in range(0, NP):
        for j in range(0, D):
            init_list[i][j] = U[i][j]
    return init_list , cost

#main function
print("start run")
y=[]
init_list = initialtion()
cost = Cost(init_list)
y.append(min(cost))
for g in range(0, Generation):
    init_list , cost = Evolution(init_list,cost)
    y.append(min(cost))
x = init_list[cost.index(min(cost))]
print("When get best fx ,The x is ")
print(x)
print('%.7f' %min(cost))

#print(len(y))
x_label = np.arange(0,Generation+1,1)
#print(x_label)
plt.plot(x_label,y)
plt.title("DE Fig")
plt.xlabel('Generation')
plt.ylabel('y')
plt.savefig('./De_Ras.png')
'''
figure = plt.figure()
axes = Axes3D(figure)
X,Y = np.meshgrid(x_label,x_label)
plt.show()
'''
