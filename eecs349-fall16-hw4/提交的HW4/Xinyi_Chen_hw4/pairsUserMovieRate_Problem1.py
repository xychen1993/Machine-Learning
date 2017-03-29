import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import json

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

def pairOfUsers(users_items):
	count = np.zeros((1682), dtype = int)
	length = len(users_items)
	n_pairs = 0;

	for i in range(length):
		items1 = users_items[i+1]

		j = i + 2
		while j <= length:
			items2 = users_items[j]
			pair_count = np.zeros((2000), dtype = int)

			for item1 in items1:
				pair_count[int(item1)] = 1

			k = 0
			for item2 in items2:
				if(pair_count[int(item2)] != 0):
					k += 1

			j += 1
			count[k] += 1
			n_pairs += 1

	
	n = 0
	n_pre = 0
	mean = 0.0
	median = 0
	count3 = []

	for i in range(len(count)):
		n_pre = n
		n += count[i]
		mean += (i)*count[i]
		count3.append(count[i])

		if n_pre <=n_pairs/2  and n >= n_pairs/2:
			median = i
	
	mean /= n
	
	print "mean: ",mean
	print "median:",median
	#writeToJson(count3,'count.txt')

	x = np.arange(0, 1682, 1)
	#count = readJson('count.txt')

	rects = plt.bar(x, count, 1)
	plt.title("movies reviewed in common")
	plt.xlabel('number of movies reviewed in common')
	plt.ylabel(' number of user pairs who have reviewed that many movies in common')
	plt.show()

def movieReviews(reviewNumber):
	reviewNumber= sorted(reviewNumber.iteritems(), key=lambda d:d[1], reverse = True)
	xticklabels = []
	y = []
	c = []
	x = []
	i = 0
	for movie in reviewNumber:
		if i%100 == 0:
			xticklabels.append(movie[0])
		i += 1

		y.append(movie[1])
		c.append(math.log(movie[1],10))
		x.append(math.log(i,10))

	print "Most reviews: No.",reviewNumber[0][0],"		Review number:",reviewNumber[0][1]
	print "Fewest reviews: No.",reviewNumber[-1][0],"		Review number:",reviewNumber[-1][1]
	print reviewNumber
	writeToJson(y,'y.txt')
	xy = range(1,1683,1)
	fig,ax = plt.subplots()
	plt.title("number of movie reviews")
	plt.xlabel("rank")
	plt.ylabel("number of reviews")
	plt.plot(xy,y,'-')
	plt.show()
	
	plt.title("loglog scale")
	plt.loglog(xy,y,'-')
	plt.show()

def main():
	file = "u.data"


	csvfile = open(file, 'rb')
	dat = csv.reader(csvfile, delimiter='	')

	user_items = {}
	reviewNumber = {}


	for i, row in enumerate(dat):
		if i >= 0:
			user = int(row[0])
			item  = str(row[1])

			if user_items.has_key(user) == True:
				user_items[user].append(item)
			else:
				user_items[user] = [item]

			if reviewNumber.has_key(item) == True:
				reviewNumber[item] += 1
			else:
				reviewNumber[item] = 1
	
	pairOfUsers(user_items)
	movieReviews(reviewNumber)
if __name__ == "__main__":
	main()