#	Starter code for linear regression problem
#	Below are all the modules that you'll need to have working to complete this problem
#	Some helpful functions: np.polyfit, scipy.polyval, zip, np.random.shuffle, np.argmin, np.sum, plt.boxplot, plt.subplot, plt.figure, plt.title
import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
import random

def nfoldpolyfit(X, Y, maxK, n, verbose = 1):
	amount = len(X)/n
	x_random = np.array([i for i in range(len(X))])
	random.shuffle(x_random)
	lists_x = []
	lists_y = []
	
	i = 0
	while i < len(X):
		
		training_sample = []
		training_sample_y = []
		for j in range(amount):
			training_sample.append(X[x_random[i+j]])
			training_sample_y.append(Y[x_random[i+j]])
		i += amount
		lists_x.append(training_sample)
		lists_y.append(training_sample_y)

	lists_test_x = []
	lists_test_y = []
	for i in range(len(x_random)-amount*(n-1)):
		lists_test_x.append(X[x_random[len(x_random)-i-1]])
		lists_test_y.append(Y[x_random[len(x_random)-i-1]])

	lists_x.append(lists_test_x)
	lists_y.append(lists_test_y)


	min_mean_square_errors = []
	best = [sys.float_info.max,0.0,0]
	best_folds_mean_square_error = sys.maxint
	
	

	for degree_j in range(maxK+1):
		
		#min_mean_square_error = 0
		square_error = 0.0

		for n_i in range(n):
			training_x = []
			training_y = []
			testing_x = []
			testing_y = []

			for l in range(len(lists_x)):
				if l == n_i:
					testing_x = lists_x[l]
					testing_y = lists_y[l]
				else:
					training_x += lists_x[l]
					training_y += lists_y[l]
		 	
			
			P = np.polyfit(training_x, training_y, degree_j, rcond=None, full=False, w=None, cov=False)
			for i in range(len(testing_x)):
				square_error += (np.polyval(P,testing_x[i]) - testing_y[i])**2
			mean_square_error = square_error/len(testing_x)
		min_mean_square_errors.append(mean_square_error/n)
		
		if best[0] > mean_square_error/n:
			best[0] = mean_square_error/n
			best[2] = degree_j
	
	degrees = np.array([i for i in range(maxK+1)])

	best[1] = np.polyfit(X, Y, best[2], rcond=None, full=False, w=None, cov=False)
	
	re = []
	X_s = sorted(X)
	xp = np.linspace(X_s[0], X_s[len(X_s)-1], 200)	
	for i in range(len(xp)):
		y = np.polyval(best[1],xp[i])
		re.append(y)

	
	fig = plt.figure()
	mse = fig.add_subplot(211)
	mse.set_title("MSE of Each K (K = %d is the best)"%best[2])
	mse.set_xlabel('k')
	mse.set_ylabel('MSE')
	mse.plot(degrees, min_mean_square_errors, '-')

	best_function = fig.add_subplot(212)
	best_function.set_title("Best Function (K = %d)"%best[2])
	best_function.set_xlabel('x')
	best_function.set_ylabel('y')
	best_function.plot(X, Y, '.', xp,re , '-')
	
	if verbose == 1:
		plt.show()


def main():
	# read in system arguments, first the csv file, max degree fit, number of folds, verbose
	rfile = sys.argv[1]
	maxK = int(sys.argv[2])
	nFolds = int(sys.argv[3])
	verbose = int(sys.argv[4])

	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []
	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)
	nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
	main()
