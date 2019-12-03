import test_funpy as test
import random
NP = 100
D = 100
F = 0.5
CR = 0.8
Generation = 30*D
Max = 5.12
Min = 0

def initialtion():
    init_list = [[0]*D for i in range(NP)]
    for i in range(0, NP):
        for j in range(0, D):
            init_list[i][j] = random.uniform(Min, Max)
    return init_list
init_list = initialtion()
F = [0]*D
f = test.test_func(init_list,F,D,1,1)
print(f)