from sympy import symbols
from sympy.plotting import plot3d
from sympy.functions import exp

x, y = symbols('x y')
plot3d(x*exp(-x**2-y**2), (x, -3, 3), (y, -2, 2))
