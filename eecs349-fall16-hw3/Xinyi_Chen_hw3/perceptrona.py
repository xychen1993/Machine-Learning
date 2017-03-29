import sys
import csv
import numpy as np
import scipy

def perceptrona(w_init, X, Y):
	k = 0

	while 1:
		k += 1
		classified = True
		
		for i in range(len(X)):
			
			m = np.polyval(w_init,X[i])
			if (Y[i] == 1 and m > 0) or (Y[i] == -1 and m < 0) :
				continue
			else:
				xy = np.array([X[i],1])*Y[i]
				w_init += xy
				classified = False
		
		if classified:
			break

	
	return (w_init, k)

def main():
	rfile = sys.argv[1]
	
	#read in csv file into np.arrays X1, X2, Y1, Y2
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X1 = []
	Y1 = []
	X2 = []
	Y2 = []
	for i, row in enumerate(dat):
		if i > 0:
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			Y1.append(float(row[2]))
			Y2.append(float(row[3]))
	X1 = np.array(X1)
	X2 = np.array(X2)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)

	w_init = np.array([0.0,0.0])
	print perceptrona(w_init, X1, Y1)
	#print perceptrona(w_init, X2, Y2)

if __name__ == "__main__":
	main()
