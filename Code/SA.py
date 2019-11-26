import matplotlib.pyplot as plt
import math
import random

"""
求解的目标表达式为：
y = 10 * math.sin(5 * x) + 7 * math.cos(4 * x)  x belongs to (0,10)
"""


def main():
    plot_obj_func()
    T_init = 100  # 初始最大温度
    alpha = 0.90  # 降温系数
    T_min = 1e-3  # 最小温度，即退出循环条件
    T = T_init
    x = random.random() * 10  # 初始化x，在0和10之间
    y = 10 * math.sin(5 * x) + 7 * math.cos(4 * x)
    results = []  # 存x，y
    while T > T_min:
        x_best = x
        # y_best = float('-inf')  # 设置这个有可能会陷入局部最优，不一定全局最优
        y_best = y  # 设置成这个收敛太快
        flag = 0  # 用来标识该温度下是否有新值被接受
        # 每个温度迭代50次，找最优解
        for i in range(50):
            delta_x = random.random() - 0.5  # 自变量进行波动
            # 自变量变化后仍要求在[0,10]之间
            if 0 < (x + delta_x) < 10:
                x_new = x + delta_x
            else:
                x_new = x - delta_x
            y_new = 10 * math.sin(5 * x_new) + 7 * math.cos(4 * x_new)
            # 要接受这个y_new为当前温度下的理想值，要满足
            # 1y_new>y_old
            # 2math.exp(-(y_old-y_new)/T)>random.random()
            # 以上为找最大值，要找最小值就把>号变成<
            if (y_new > y or math.exp(-(y - y_new) / T) > random.random()):
                flag = 1  # 有新值被接受
                x = x_new
                y = y_new
                if y > y_best:
                    x_best = x
                    y_best = y
        if flag:
            x = x_best
            y = y_best
        results.append((x, y))
        T *= alpha

    print('最优解 x:%f,y:%f' % results[-1])

    plot_final_result(results)
    plot_iter_curve(results)



# 看看我们要处理的目标函数
def plot_obj_func():
    """y = 10 * math.sin(5 * x) + 7 * math.cos(4 * x)"""
    X1 = [i / float(10) for i in range(0, 100, 1)]
    Y1 = [10 * math.sin(5 * x) + 7 * math.cos(4 * x) for x in X1]
    plt.plot(X1, Y1)
    plt.show()


# 看看最终的迭代变化曲线
def plot_iter_curve(results):
    X = [i for i in range(len(results))]
    Y = [results[i][1] for i in range(len(results))]
    plt.plot(X, Y)
    plt.show()

def plot_final_result(results):
    X1 = [i / float(10) for i in range(0, 100, 1)]
    Y1 = [10 * math.sin(5 * x) + 7 * math.cos(4 * x) for x in X1]
    plt.plot(X1, Y1)
    plt.scatter(results[-1][0], results[-1][1], c='r', s=10)
    plt.show()

if __name__ == '__main__':
    # for i in range(100):
    main()