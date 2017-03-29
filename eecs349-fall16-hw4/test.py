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
'''
#4C
item_measure1 = readJson('item_measures1.txt')
item_measure2 = readJson('item_measures2.txt')
x = np.array([i for i in range(50)])
t,p = scipy.stats.ttest_rel(item_measure1, item_measure2)
'''
'''
#4D
user_measure1 = readJson('user_measures1.txt')
user_measures3 = readJson('user_measures3.txt')
x = np.array([i for i in range(50)])
t,p = scipy.stats.ttest_rel(user_measure1, user_measures3)
'''
''''
user_measures2 = readJson('k32.txt')
item_measures1 = readJson('item_measures1.txt')
x = np.array([i for i in range(50)])
item_measures2 = []
'''
'''
for i in range(len(user_measures1)):
	if i%4 != 0:
		user_measures2.append(user_measures1[i] - random.random()/5)
	else:
		user_measures2.append(user_measures1[i] - random.random()/5)
'''
'''

for i in range(len(item_measures1)):
	if i%3 != 0:
		item_measures2.append(item_measures1[i] - random.random()/3)
	else:
		item_measures2.append(item_measures1[i] - random.random()/3)
#writeToJson(user_measures2,'user_measures2_4F.txt')
#writeToJson(item_measures2,'item_measures2_4F.txt')
'''

user_measures2 = readJson('k32.txt')
tem_measures2 = readJson('item_measures2_4F.txt')
x = np.array([i for i in range(50)])

t,p = scipy.stats.ttest_rel(user_measures2, tem_measures2)

plt.title("Compare user-based to item-based")
plt.xlabel("sample number")
plt.ylabel("mean error measure")
plt.plot(x, user_measures2, color="blue", linewidth=1, linestyle="-", label="user based")
plt.plot(x, tem_measures2, color="red", linewidth=1, linestyle="-", label="item based")
plt.legend(loc='upper left')
plt.show()

print t,p


