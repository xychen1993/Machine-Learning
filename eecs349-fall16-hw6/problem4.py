#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from gmm_classify import gmm_classify
from gmm_est import gmm_est
import scipy.stats
import math


def main():
    train = 'gmm_train.csv'
    test = 'gmm_test.csv'
    t1, t2 = read_gmm_file(train)
    X1, X2 = read_gmm_file(test)
    # YOUR CODE FOR PROBLEM 3 GOES HERE
    # initial value
    wt1 = [0.5, 0.5]
    mu1 = [10.0, 30.0]
    sigmasq1 = [1.0, 1.0]

    p1 = len(t1) / float(len(t1) + len(t2))
    mu2 = [-25.0, -5.0, 50.0]
    sigmasq2 = [1.0, 1.0, 1.0]
    wt2 = [0.2, 0.3, 0.5]

    mu1, sigma1, w1, L1 = gmm_est(t1, mu1, sigmasq1, wt1, 20)
    mu2, sigma2, w2, L2 = gmm_est(t2, mu2, sigmasq2, wt2, 20)
    # get class data
    result1 = gmm_classify(X1, mu1, sigma1, w1,  mu2, sigma2, w2, p1)
    result2 = gmm_classify(X2, mu1, sigma1, w1,  mu2, sigma2, w2, p1)
    
    if p1 >= 0.5:
        prior_error_rate = 1 - p1
    else:
        prior_error_rate = p1
    
    count = 0
    for c in result1:
        if c == 2:
            count += 1
    for c2 in result2:
        if c2 == 1:
            count += 1
    GMM_error_rate = count * 1.0 / (len(result1) + len(result2)) * 1.0

    print 'prior error rate: %.2f%%, GMM error rate: %.2f%%' %(prior_error_rate*100, GMM_error_rate*100)
    
    class1x1_data, class2x1_data = separate_class(X1, result1)
    class1_x1 = np.zeros(len(class1x1_data))
    class2_x1 = np.zeros(len(class2x1_data))
    class1x2_data, class2x2_data = separate_class(X2, result2)
    class1_x2 = np.zeros(len(class1x2_data))
    class2_x2 = np.zeros(len(class2x2_data))

    plt.subplot(2, 1, 1)
    plt.title('class 1')
    plt.xlabel('data point')
    plt.ylabel('amount')
    plt.hist(X1, len(X1), color='green')
    plt.plot(class1x1_data, class1_x1, 'go')
    plt.plot(class2x1_data, class2_x1, 'ro')

    plt.subplot(2, 1, 2)
    plt.title('class 2')
    plt.xlabel('data point')
    plt.ylabel('amount')
    plt.hist(X2, len(X2), color='black')
    plt.plot(class2x2_data, class2_x2, 'go')
    plt.plot(class1x2_data, class1_x2, 'ro')
    plt.show()

def normpdf(x, mu, sigma):
    u = (x - mu) / abs(sigma)
    y = (1 / (math.sqrt(2 * math.pi) * abs(sigma))) * math.exp(-u * u / 2)
    return y

def separate_class(x, result):
    class1_data = []
    class2_data = []
    for i in range(len(x)):
        if result[i] == 1:
            class1_data.append(x[i])
        else:
            class2_data.append(x[i])
    return np.array(class1_data), np.array(class2_data)


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
