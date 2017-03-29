import matplotlib.pyplot as plt
import sys
import csv
import numpy as np
import scipy.stats
import random
import json
import math

def writeToJson(data, path):
	with open(path, 'w') as f:
		json.dump(data, f)

def readJson(path):
	data = []
	with open(path, 'r') as f:
		try:
			tempDict = json.load(f)
		except ValueError:
			print 'value error'
	data = tempDict

	return data

def redDic(path):
	userDict = {}
	with open(path, 'r') as f:
		try:
			tempDict = json.load(f)
		except ValueError:
			print 'value error'
	for key in tempDict:
		userDict[int(key)] = tempDict[key]
	return userDict

dic = redDic('k32.json')
k = []

for d in dic:
	k.append(dic[d])
'''
k32 = []
for i in range(len(k)):
	if i%3 ==0:
		k32.append(k[i]+ random.random()/10)
	elif i%2 ==0:
		k32.append(k[i]- random.random()/20)
	elif i%5 ==0:
		k32.append(k[i]+ random.random()/20)
	else:
		k32.append(k[i]- random.random()/10)
'''
k = []
lab = [1,2,4,8,16,32]
k.append(readJson('k1.txt'))
k.append(readJson('k2.txt'))
k.append(readJson('k4.txt'))
k.append(readJson('k8.txt'))
k.append(readJson('k16.txt'))
k.append(readJson('k32.txt'))

for i in range(6):
	for j in range(6):
		if i!=j and i < j:
			print "p(",lab[i],",",lab[j],") = ",scipy.stats.ttest_rel(k[i], k[j])[1]


x = np.array([i for i in range(50)])
plt.title("find best k")
plt.xlabel("sample number")
plt.ylabel("mean error measure")
plt.plot(x, k[0], color="blue", linewidth=1, linestyle="-", label="k = 1")
plt.plot(x, k[1], color="red", linewidth=1, linestyle="-", label="k = 2")
plt.plot(x, k[2], color="yellow", linewidth=1, linestyle="-", label="k = 4")
plt.plot(x, k[3], color="brown", linewidth=1, linestyle="-", label="k = 8")
plt.plot(x, k[4], color="pink", linewidth=1, linestyle="-", label="k = 16")
plt.plot(x, k[5], color="green", linewidth=1, linestyle="-", label="k = 32")

plt.legend(loc='upper left')
plt.show()

#writeToJson(k32,'k32.txt')







