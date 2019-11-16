import numpy as np
Hi = np.ones((2,2))
Hb = np.ones((2,2))

	
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def AF(p,kind):   # 激励函数
	t = []
	if kind == 1:   # sigmoid
	    return sigmoid(p)
	elif kind == 2:   # tanh
		pass
	elif kind == 3:    # ReLU
		return np.where(p<0,0,p)
	else:
		pass
 
		
def dAF(p,kind):   # 激励函数导数
	t = []
	if kind == 1:   # sigmoid
		return (sigmoid(p)*(1 - sigmoid(p)))
	elif kind == 2:   # tanh
		pass
	elif kind == 3:    # ReLU
		
		return np.where(p<0,0,1) 
	else:
		pass

