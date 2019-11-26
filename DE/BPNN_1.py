import numpy as np
import math
import matplotlib.pyplot as plt 
 
######## 数据集 ########
 
p_s = [[1,1,2],[1,2,3],[2,1,6],[5,2,5],[8,3,4],[7,7,4],[7,7,7],[13,8,3],[6,10,11],[13,0,17],[14,7,12]]              # 用来训练的数据集 x
t_s = [[2],[6],[12],[50],[96],[196],[343],[312],[660],[0],[1176]]   # 用来训练的数据集 y
 
p_t = [[6,9,1017],[2,3,4],[5,9,10]]      # 用来测试的数据集 x_test    
t_t = [[54918],[24],[450]]               # 用来测试的数据集 对应的实际结果 y_test                                                                        
 
######## 超参数设定 ########
 
n_epoch = 20000           # 训练次数
 
HNum = 2;                   # 各层隐藏层节点数
 
HCNum = 2;                  # 隐藏层层数
 
AFKind = 3;                 # 激励函数种类
emax = 0.1;                # 最大允许均方差根
LearnRate = 0.01;           # 学习率
 
######## 中间变量设定 ########
TNum = 7;                   # 特征层节点数 (特征数)
 
SNum = len(p_s);            # 样本数
 
INum = len(p_s[0]);         # 输入层节点数（每组数据的维度）
ONum = len(t_s[0]);         # 输出层节点数（结果的维度）
StudyTime = 0;              # 学习次数
KtoOne = 0.0;               # 归一化系数
e = 0.0;                    # 均方差根
######################################################### 主要矩阵设定 ######################################################
 
I = np.zeros(INum);         #输入层矩阵
 
Ti = np.zeros(TNum);		
To = np.zeros(TNum);
 
Hi = np.zeros((HCNum,HNum));
Ho = np.zeros((HCNum,HNum));
 
Oi = np.zeros(ONum);
Oo = np.zeros(ONum);
 
Teacher = np.zeros(ONum);
 
u = 0.2*np.ones((TNum,HNum))                  # 初始化 权值矩阵u
w = 0.2*np.ones(((HCNum-1,HNum,HNum)))        # 初始化 权值矩阵w
v = 0.2*np.ones((HNum,ONum))                  # 初始化 权值矩阵v
 
dw = np.zeros((HCNum-1,HNum,HNum))
 
Hb = np.zeros((HCNum,HNum));
Ob = np.zeros(ONum);
 
He = np.zeros((HCNum,HNum));
Oe = np.zeros(ONum);
 
p_s = np.array(p_s)
t_s = np.array(t_s)
p_t = np.array(p_t)
 

######################################################### 方法 #######################################################
 
def Calcu_KtoOne(p,t):                         # 确定归一化系数
	p_max = p.max();
	t_max = t.max();
	return max(p_max,t_max);
	
def trait(p):                                  # 特征化
	t = np.zeros((p.shape[0],TNum));
	for i in range(0,p.shape[0],1):
		t[i,0] = p[i,0]*p[i,1]*p[i,2]
		t[i,1] = p[i,0]*p[i,1]
		t[i,2] = p[i,0]*p[i,2]
		t[i,3] = p[i,1]*p[i,2]
		t[i,4] = p[i,0]
		t[i,5] = p[i,1]
		t[i,6] = p[i,2]
	
	return t
	
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def d_sigmoid(x):
	return sigmoid(x)*(1 - sigmoid(x))


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
		return d_sigmoid(p)
	elif kind == 2:   # tanh
		pass
	elif kind == 3:    # ReLU
		
		return np.where(p<0,0,1) 
	else:
		pass
 
		
		
def nnff(p,t):
	pass
	
def nnbp(p,t):
	pass
	
 
def train(p,t):                                # 训练
	
	global e	#方差
	global v   	#输出层权值
	global w	#隐层1的权值
	global dw	#隐层2的权值
	global u	#输出层权值
	global I 	#输入
	global Ti 
	global To 
	global Hi 
	global Ho 
	global Oi 
	global Oo 
	global Teacher 
	global Hb 
	global Ob 
	global He 
	global Oe
	global StudyTime
	global KtoOne

	e = 0.0
	p = trait(p)	
	KtoOne = Calcu_KtoOne(p,t)
		
	for isamp in range(0,SNum,1):
		To = p[isamp]/KtoOne
		Teacher = t[isamp]/KtoOne
		
		
		################ 前向 nnff #############################
		######## 计算各层隐藏层输入输出 Hi Ho ########
		
		for k in range(0,HCNum,1):
			if k == 0:
				Hi[k] = np.dot(To,u)
				Ho[k] = AF(np.add(Hi[k],Hb[k]),AFKind)
			else:
				Hi[k] = np.dot(Ho[k-1],w[k-1])
				Ho[k] = AF(np.add(Hi[k],Hb[k]),AFKind)
		
		########   计算输出层输入输出 Oi Oo    ########
		Oi = np.dot(Ho[HCNum-1],v)
		Oo = AF(np.add(Oi,Ob),AFKind)
		
				
		################ 反向 nnbp #############################
		
		######## 反向更新 v ############
		
		Oe = np.subtract(Teacher,Oo)
		Oe = np.multiply(Oe,dAF(np.add(Oi,Ob),AFKind))
						
		e += sum(np.multiply(Oe,Oe))
		
		
		
		#### v 梯度 ####		
		
		dv = np.dot(np.array([Oe]),np.array([Ho[HCNum-1]])).transpose()			  # v 的梯度
 
		v = np.add(v,dv*LearnRate)    # 更新 v
		
		######## 反向更新 w #############
		He = np.zeros((HCNum,HNum))
	
		for c in range(HCNum-2,-1,-1):
			if c == HCNum-2:
				He[c+1] = np.dot(v,Oe)
				He[c+1] = np.multiply(He[c+1],dAF(np.add(Hi[c+1],Hb[c+1]),AFKind))
				
				
				#dw[c] = dot(array([He[c+1]]),array([Ho[c]]).transpose())
				dw[c] = np.dot(np.array([Ho[c]]).transpose(),np.array([He[c+1]]))
				#dw[c] = dw[c].transpose()  #@@@@@@ 若结果不理想，可尝试用此条语句
				
				w[c] = np.add(w[c],LearnRate*dw[c])
				
		
				
			else:
				He[c+1] = np.dot(w[c+1],He[c+2])
				He[c+1] = np.multiply(He[c+1],dAF(np.add(Hi[c+1],Hb[c+1]),AFKind))
				
				dw[c] = np.dot(np.array([Ho[c]]).transpose(),np.array([He[c+1]]))	
				
				w[c] = np.add(w[c],LearnRate*dw[c])

		######## 反向更新 u #############
		
		He[0] = np.dot(w[0],He[1])
		He[0] = np.multiply(He[0],dAF(np.add(Hi[0],Hb[0]),AFKind))
				
				
		du = np.dot(np.array([To]).transpose(),np.array([He[0]]))
				
		u = np.add(u,du)
		
		
		######### 更新阈值 b ############
		
		Ob = Ob + Oe*LearnRate
				
		Hb = Hb + He*LearnRate
		
	
	e = np.sqrt(e)
 
	
def predict(p):
				
	p = trait(p)
	p = p/KtoOne
	p_result = np.zeros((p.shape[0],1))
 
	for isamp in range(0,p.shape[0],1):
		for k in range(0,HCNum,1):
			if k == 0:
				Hi[k] = np.dot(p[isamp],u)
				Ho[k] = AF(np.add(Hi[k],Hb[k]),AFKind)
			else:
				Hi[k] = np.dot(Ho[k-1],w[k-1])
				Ho[k] = AF(np.add(Hi[k],Hb[k]),AFKind)
			
			
		########   计算输出层输入输出 Oi Oo    ########
		Oi = np.dot(Ho[HCNum-1],v)
		Oo = AF(np.add(Oi,Ob),AFKind)
		Oo = Oo*KtoOne
		p_result[isamp] = Oo
	return p_result
 
# 可视化多项式曲线拟合结果
def draw_fit_curve(origin_xs, origin_ys, prediction_ys, step_arr, loss_arr):
	pass
	
step , e_t = [],[]
for i in range(1,n_epoch,1):
	if i%1000 == 0:
		print('已训练 %d 千次 ,误差均方差 %f'%((i/1000),e))
		step.append(i)
		e_t.append(e)
	train(p_s,t_s)
print('训练完成，共训练 %d 次，误差均方差 %f'%(i,e))

print()
		
result = predict(p_t)
 
print('模型预测结果 : ')
for i in result:
	print('%.2f'%i)
		
print('\n实际结果 : ')	
for i in t_t:
	print(i)
#误差下降曲线
plt.title("BPNN ERRO CUR")
plt.plot(step,e_t) 
plt.xlabel('X:Generation')
plt.ylabel('Y:erro')
plt.savefig('C:\\Users\\Evil\\Desktop\\mycode\\Pic\\BpNN.png')
plt.show()
		