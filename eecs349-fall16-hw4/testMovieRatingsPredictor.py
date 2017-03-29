import sys
import numpy as np
import csv
import item_cf
import user_cf
import random
import scipy.stats
import json


# 0 is user_cf, 1 is item_cf
def getSamples(datafile):
	samples = []

	t = 0
	while t < 50:
		random_list = random.sample(range(0,100000),100)
		
		sample = []
		csvfile = open(datafile, 'rb')
		dat = csv.reader(csvfile, delimiter='\t')
		
		k = 0
		for i, row in enumerate(dat):
			if k == 100:
				break
			if i in random_list:
				sample.append([int(row[0]),int(row[1]),int(row[2])])
				k += 1

		t += 1
		samples.append(sample)

	return samples


def measure_error(val1,val2):

	return (val1-val2)**2.0

def hw4_4B(samples,datafile = 'u.data',distance =0, k = 5, iFlag = 0, numOfUsers = 943, numOfItems= 1682):
	user_measures= []
	item_measures = []
    
	for i in range(50):
		user_measure = 0.0
    	item_measure = 0.0
    	for j in range(100):
			trueRating1,predictedRating1 = user_cf.user_based_cf(datafile, samples[i][j][0], samples[i][j][1], distance , k, iFlag, numOfUsers, numOfItems)
			user_measure += measure_error(trueRating1,predictedRating1)
    		
			trueRating2,predictedRating2 = item_cf.item_based_cf(datafile, samples[i][j][0], samples[i][j][1], distance , k, iFlag, numOfUsers, numOfItems)
			item_measure += measure_error(trueRating2,predictedRating2)
    	
    	user_measures.append(user_measure/100.0)
    	item_measures.append(item_measure/100.0)

    	return user_measures,item_measures



def hw4_4C(samples,datafile = 'u.data',distance =0, k = 5, iFlag = 0, numOfUsers = 943, numOfItems= 1682):
	item_measures = []
	
	for i in range(50):
		item_measure = 0.0
		for j in range(100):
			trueRating,predictedRating = item_cf.item_based_cf(datafile, samples[i][j][0], samples[i][j][1], distance , k, iFlag, numOfUsers, numOfItems)
			item_measure += measure_error(trueRating,predictedRating)
		item_measures.append(item_measure/100.0)
		print i,item_measures

	return item_measures

def hw4_4D(samples,datafile = 'u.data',distance =0, k = 5, iFlag = 0, numOfUsers = 943, numOfItems= 1682):
	user_measures = []
	
	for i in range(50):
		user_measure = 0.0
		for j in range(100):
			trueRating,predictedRating = user_cf.user_based_cf(datafile, samples[i][j][0], samples[i][j][1], distance , k, iFlag, numOfUsers, numOfItems)
			user_measure += measure_error(trueRating,predictedRating)
		user_measures.append(user_measure/100.0)
		print i,user_measures

	return user_measures

def hw4_4E(samples,datafile = 'u.data',distance =0, k = 5, iFlag = 0, numOfUsers = 943, numOfItems= 1682):
	user_measures = []
	
	for i in range(50):
		user_measure = 0.0
		for j in range(100):
			trueRating,predictedRating = user_cf.user_based_cf(datafile, samples[i][j][0], samples[i][j][1], distance , k, iFlag, numOfUsers, numOfItems)
			user_measure += measure_error(trueRating,predictedRating)
		user_measures.append(user_measure/100.0)
		print i,user_measures

	return user_measures

def hw4_4F(samples,datafile = 'u.data',distance =0, k = 5, iFlag = 0, numOfUsers = 943, numOfItems= 1682):
	user_measures= []
	item_measures = []
    
	for i in range(50):
		user_measure = 0.0
    	item_measure = 0.0
    	for j in range(100):
    		print j,samples[i][j],user_measure,item_measure
    		trueRating1,predictedRating1 = user_cf.user_based_cf(datafile, samples[i][j][0], samples[i][j][1], distance , k, iFlag, numOfUsers, numOfItems)
    		user_measure += measure_error(trueRating1,predictedRating1)
    		
    		trueRating2,predictedRating2 = item_cf.item_based_cf(datafile, samples[i][j][0], samples[i][j][1], distance , k, iFlag, numOfUsers, numOfItems)
    		item_measure += measure_error(trueRating2,predictedRating2)
    	
    	user_measures.append(user_measure/100.0)
    	item_measures.append(item_measure/100.0)
    	print i,user_measures,item_measures

    	return user_measures,item_measures


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

def main(datafile,numOfUsers,numOfItems):

	
	#samples = getSamples('u.data')
	#writeToJson(samples,'samples.txt')
	samples = readJson('samples.txt')

	#user_measures1,item_measures1 = hw4_4B(samples,datafile, 0, 5, 0,numOfUsers,numOfItems)
	#t1,p1 = scipy.stats.ttest_ind(user_measures, item_measures, equal_var=False)
	#print t,p

	'''
	item_measures2 = hw4_4C(samples,datafile, 1, 5, 0,numOfUsers,numOfItems)
	print '2',item_measures2
	writeToJson(item_measures2,'item_measures2.txt')
	
	user_measures3 = hw4_4D(samples,datafile,0,5,1,numOfUsers,numOfItems)
	print '3',user_measures3
	writeToJson(user_measures3,'user_measures3.txt')
	'''

	ks = [1,2,4,8,16,32]
	user_measures4 = []
	for k in ks:
		print k
		user_measures4.append(hw4_4E(samples,datafile,0,k,0,numOfUsers,numOfItems))
	print '4',user_measures4
	writeToJson(user_measures4,'user_measures4.txt')

	user_measures5,item_measures5 = hw4_4F(samples,datafile, 0, 5, 0,numOfUsers,numOfItems)
	data = [user_measures5,item_measures5]
	print '5'
	writeToJson(data,'measures5.txt')

main('u.data',943,1682)
#hw4_4C()




