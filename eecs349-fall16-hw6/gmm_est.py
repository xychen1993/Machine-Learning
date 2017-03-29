import sys
import time
import numpy as np
import scipy.stats 
import csv
import math
import matplotlib.pyplot as plt

def main():
    """
    This function runs your code for problem 2.

    You can also use this to test your code for problem 1,
    but make sure that you do not leave anything in here that will interfere
    with problem 2. Especially make sure that gmm_est does not output anything
    extraneous, as problem 2 has a very specific expected output.
    """
    file_path = sys.argv[1]
    x1, x2 = read_gmm_file(file_path)

    wt_init1 = [0.5, 0.5]
    sigmasq_init1 = [1.0, 1.0]
    mu_init1 = [10.0, 30.0]

    mu_init2 = [-25.0, -.0, 50.0]
    sigmasq_init2 = [1.0, 1.0, 1.0]
    wt_init2 = [.2, .3, .5]

    mu_results1, sigma2_results1, w_results1, L1 = gmm_est(x1, mu_init1, sigmasq_init1, wt_init1, 20)
    mu_results2, sigma2_results2, w_results2, L2 = gmm_est(x2, mu_init2, sigmasq_init2, wt_init2, 20)


    x = np.array([i for i in range(20)])


    plt.plot(x,L1,color="blue", linewidth=1, linestyle="-", label="class 1")
    plt.plot(x,L2,color="red", linewidth=1, linestyle="-", label="class 2")
    plt.xlabel("iteration number")
    plt.ylabel("likelyhood")
    plt.legend(loc='upper right')
    plt.savefig("likelihood_problem2.png")

    
    # YOUR CODE FOR PROBLEM 2 GOES HERE

    # mu_results1, sigma2_results1, w_results1 are all numpy arrays
    # with learned parameters from Class 1
    print 'Class 1'
    print 'mu =', mu_results1, '\nsigma^2 =', sigma2_results1, '\nw =', w_results1

    # mu_results2, sigma2_results2, w_results2 are all numpy arrays
    # with learned parameters from Class 2
    print '\nClass 2'
    print 'mu =', mu_results2, '\nsigma^2 =', sigma2_results2, '\nw =', w_results2

def normpdf(x, mu, sigma):
    u = (x - mu) / abs(sigma)
    y = (1 / (math.sqrt(2 * math.pi) * abs(sigma))) * math.exp(-u * u / 2)
    return y

def gmm_est(X,mu_init,sigmasq_init,wt_init,its):

    L = []
    N = len(X)
    for i in range(its):

        ll = 0.0
        for x in X:
            ll_temp = 0
            for k_l in range(len(mu_init)):
                ll_temp += wt_init[k_l] * normpdf(x,mu_init[k_l], math.sqrt(sigmasq_init[k_l]))
            ll += np.log(ll_temp)
        L.append(ll)

        for j in range (len(mu_init)):
            r = []
            big_r = 0.0

            for xn in X:
                numerator = wt_init[j]*normpdf(xn,mu_init[j], math.sqrt(sigmasq_init[j]))
                denominator = 0.0

                for k in range(len(mu_init)):
                    denominator += wt_init[k] * normpdf(xn,mu_init[k], math.sqrt(sigmasq_init[k]))

                if denominator == 0:
                    r.append(0.0)
                else:
                    r.append(numerator/denominator)
                    big_r += numerator/denominator


            tmp2 = []
            for x in X:
                tmp2.append((x-mu_init[j])**2)

            sigmasq_init[j] = np.dot(r, tmp2) / big_r

            wt_init[j] = big_r/N

            mu_init[j] = np.dot(r, X)/big_r

    return mu_init, sigmasq_init, wt_init, L

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
