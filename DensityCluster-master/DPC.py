from __future__ import division
import numpy as np
from sklearn import preprocessing
from scipy.special import comb
from sklearn.metrics import adjusted_rand_score,f1_score,accuracy_score,precision_score,precision_recall_fscore_support
from sklearn.datasets import *
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv
import random
from time import time
from matplotlib.font_manager import FontProperties
defaultencoding = 'utf-8'

def calculate(data,percent):
	#计算距离
	distlist = []
	dist = np.zeros((len(data), len(data)));
	for i in range(len(data)-1):
		for j in range(i+1,len(data)):
			distance = np.sqrt(np.sum(np.square(data[i] - data[j])))
			dist[i, j] = distance
			dist[j, i] = distance
			if(i != j):
				distlist.append(distance)
	# dc
	# for i in range(len(dist) - 1):
	# 	for j in range(len(dist) - 1):
	# 		if(dist[i,j] != 0):
	sortdist = sorted(distlist)
	position = round(len(distlist) * percent / 100)
	dc = sortdist[position]
	print('dc',dc)
	#计算局部密度
	# 计算局部密度 rho (利用 Gaussian 核)
	rho = np.zeros(len(dist))
	# Gaussian kernel
	for i in range(len(dist) - 1):
		for j in range(i + 1, len(dist)):
			rho[i] = rho[i] + np.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
			rho[j] = rho[j] + np.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
	rho = [item / max(rho) for item in rho]
	# 生成 delta 和 nneigh 数组
	# delta 距离数组
	# nneigh 比本身密度大的最近的点
	# 记录 rho 值更大的数据点中与 ordrho(ii) 距离最近的点的编号 ordrho(jj)
	# 将 rho 按降序排列，ordrho 保持序
	ordrho = np.flipud(np.argsort(rho))
	delta = np.zeros(len(dist))
	nneigh = np.zeros(len(dist),dtype=int)
	delta[ordrho[0]] = -1.
	nneigh[ordrho[0]] = 0
	# 求最大距离
	maxd = max(dist.flatten())
	for ii in range(1,len(dist)):
		delta[ordrho[ii]] = maxd
		for jj in range(ii):
			if dist[ordrho[ii], ordrho[jj]] < delta[ordrho[ii]]:
				delta[ordrho[ii]] = dist[ordrho[ii], ordrho[jj]]
				nneigh[ordrho[ii]] = ordrho[jj]
		delta[ordrho[0]] = max(delta)
		# delta = [item/max(delta) for item in delta]
	gamma = [rho[i] * delta[i] for i in range(len(dist))]
	plt.figure(1);
	for index in range(len(dist)):
		plt.plot(rho[index], delta[index], 'ko');
	pos = plt.ginput(1)
	rhomin = pos[0][0]
	deltamin = pos[0][1]
	cl = -1*np.ones(len(dist),dtype=int)
	icl = []
	NCLUST = 1
	for i in range(len(dist)):
		if rho[i]>rhomin and delta[i]>deltamin:
			cl[i] = NCLUST #第 i 号数据点属于第 NCLUST 个 cluster
			icl.append(i) # 逆映射,第 NCLUST 个 cluster 的中心为第 i 号数据点
			NCLUST+=1
	for i in range(len(dist)):
		if cl[ordrho[i]] == -1:
			cl[ordrho[i]] = cl[nneigh[ordrho[i]]]
	print(cl)
	color = ['bo', 'co', 'go', 'ko', 'mo', 'ro', 'wo', 'yo']
	s = random.sample(color, NCLUST-1)
	plt.close()
	for i in range(len(data)):
		plt.plot(data[i][0],data[i][1],s[cl[i]-1])
	plt.show();
	
def loadData():
	dataSet = []
	# fileIn = open('C:/data/jain.txt')
	fileIn = open('C:/data/jain.dat')
	for line in fileIn.readlines():
		lineArr = line.strip().split('\t')
		dataSet.append([float(lineArr[0]), float(lineArr[1])])
	data = np.array(dataSet);
	return data;

#data,label = load_breast_cancer(True)
#min_max_scaler = preprocessing.MinMaxScaler()
#data = min_max_scaler.fit_transform(data)
def test():
	data = loadData()
	dataSet = np.array(data);
	dist = calculate(dataSet,2)
	# local_density(dist,2)
if __name__ == '__main__': 
	test();
# len = len(data)
# # start = time()
# fsdp = FSDP(len,7.8605,data)
# fsdp.clustering(0.4703,0.7108)
# print (time() - start)/60.0
# print fsdp.cl
# print label
# print "f1_score"
# print f1_score(fsdp.cl,label,average="weighted")
# print "accary"
# print accuracy_score(fsdp.cl,label)
# print adjusted_rand_score(fsdp.cl,label)
# gamma = np.flipud(np.sort(fsdp.gamma))
# plt.figure("标准CS")
# plt.subplot(311)
# x = [t for t in range(0, len(gamma))]
# plt.plot(x, gamma, 'r*')
# plt.subplot(312)
# plt.plot(fsdp.rho, fsdp.delta, 'r*')
# plt.legend(loc='upper right', prop=font)
#
# plt.subplot(313)
# color = []
# for i in fsdp.cl:
#     color.append(colors.cnames.values()[i])
# plt.scatter(data[:,0], data[:,1],marker=".",c=color)
# plt.legend(loc='upper right', prop=font)
# plt.show()
