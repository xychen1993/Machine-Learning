#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from gmm_est import gmm_est
import scipy.stats
import math

np.set_printoptions(threshold=np.nan)

def main():

    file_path = sys.argv[1]
    X1, X2 = read_gmm_file(file_path)
    

    trainPath = 'gmm_train.csv'
    t1, t2 = read_gmm_file(trainPath)
    wt1 = [0.5, 0.5]
    mu1 = [10.0, 30.0]
    sigmasq1 = [1.0, 1.0]
    
    p1 = len(X1) / float(len(X1) + len(X2))
    mu2 = [-25.0, -5.0, 50.0]
    sigmasq2 = [1.0, 1.0, 1.0]
    wt2 = [0.2, 0.3, 0.5]
    mu1, sigma1, w1, L1 = gmm_est(t1, mu1, sigmasq1, wt1, 20)
    mu2, sigma2, w2, L2 = gmm_est(t2, mu2, sigmasq2, wt2, 20)
    result1 = gmm_classify(X1, mu1, sigma1, w1,  mu2, sigma2, w2, p1)
    result2 = gmm_classify(X2, mu1, sigma1, w1,  mu2, sigma2, w2, p1)
    
    class1_data = []
    class2_data = []
    for i in range(len(X1)):
        if result1[i] == 1:
            class1_data.append(X1[i])
        else:
            class2_data.append(X1[i])
    for i in range(len(X2)):
        if result2[i] == 1:
            class1_data.append(X2[i])
        else:
            class2_data.append(X2[i]) 

    class1_data = np.array(class1_data)
    class2_data = np.array(class2_data)


    print 'Class 1'
    print class1_data

    print '\nClass 2'
    print class2_data

def normpdf(x, mu, sigma):
    u = (x - mu) / abs(sigma)
    y = (1 / (math.sqrt(2 * math.pi) * abs(sigma))) * math.exp(-u * u / 2)
    return y

def gmm_classify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1):
    
    class_pred = []

    for i in range(len(X)):
        p11 = 0.0
        p22 = 0.0
        
        for k in range(len(mu1)):
            p11 += wt1[k]*normpdf(X[i],mu1[k], math.sqrt(sigmasq1[k]))
        for k in range(len(mu2)):
            p22 += wt2[k]*normpdf(X[i],mu2[k], math.sqrt(sigmasq2[k]))
        p11 *= p1
        p22 *= 1 - p1
        
        if p11 >= p22:
            class_pred.insert(i,1)
        else:
            class_pred.insert(i,2)
    
    return class_pred


def read_gmm_file(path_to_file):
    """
    Reads either gmm_test.csv or gmm_train.csv
    :param path_to_file: path to .csv file
    :return: two numpy arrays for data with label 1 (X1) and data with label 2 (X2)
    """
    X1 = []
    X2 = []

    data = open(path_to_file).readlines()[1:] # we don't need the first line
    for d in data:
        d = d.split(',')

        # We know the data is either class 1 or class 2
        if int(d[1]) == 1:
            X1.append(float(d[0]))
        else:
            X2.append(float(d[0]))

    X1 = np.array(X1)
    X2 = np.array(X2)

    return X1, X2

if __name__ == '__main__':
    main()
