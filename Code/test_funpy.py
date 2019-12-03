
xx = [1,2,3,4]
D=4
def Test(x,xshift):
    for i in range(0,D):
        xshift[i] = x[i] - 1


def tt():
    fit = [0]*len(xx)
    Test(xx,fit)
    print(fit)
tt()