
import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
 
def f1(x):
    return 4.0*x-2.0
def f2(x):
    return 0.5*x+2.0
def f3(x):
    return -0.3*x+7.0
 
 
xr = np.linspace(0.0,10.0,100)
y1r = f1(xr)
y2r = f2(xr)
y3r = f3(xr)
 
plt.plot(xr,y1r,label=r'$y=4.0*x-2.0$')
plt.plot(xr,y2r,label=r'$y=0.5*x=2.0$')
plt.plot(xr,y3r,label=r'$y=-0.3*x+7.0$')
 
x = Symbol('x')
x1, =  solve(f1(x)-f2(x))
x2, =  solve(f1(x)-f3(x))
x3, =  solve(f2(x)-f3(x))
 
 
y1 = f1(x1)
y2 = f3(x2)
y3 = f2(x3)
 
 
plt.fill([x1,x2,x3,x1],[y1,y2,y3,y1],'grey',alpha=0.5)
 
plt.xlim((0.0, 10.000))
plt.ylim((0.0, 10.000))
 
plt.grid(True, linestyle='-.')
 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
 
plt.xlabel(r'$x$',fontsize=13)
plt.ylabel(r'$y$',fontsize=13)
 
plt.show()
