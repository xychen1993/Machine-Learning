#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Prem Seetharaman (replace your name here)

import sys
import numpy as np
import os
import shutil
import math
import random
import json
import matplotlib.pyplot as plt
import scipy.stats



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


def parse(text_file):
	#This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Don't edit this function. It writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)
		

def makedictionary(spam_directory, ham_directory, dictionary_filename):
	#Making the dictionary. 
	spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))]
	ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))]
	
	random1 = np.array([i for i in range(len(spam))])
	random2 = np.array([i for i in range(len(ham))])
	random.shuffle(random1)
	random.shuffle(random2)

	spam_random = []
	ham_random = []
	data = []
	for i in range(len(random1)):
		spam_random.append(spam[random1[i]])

	for i in range(len(random2)):
		ham_random.append(ham[random2[i]])

	flag_spam = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]
	#flag_ham = [0, 255, 510, 765, 1020, 1275, 1530, 1785, 2040, 2295, 2551]
	flag_ham = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 501]

	
	for i in range(11):
		if i == 10:
			break
		spam_train = spam_random[:flag_spam[i]] + spam_random[flag_spam[i + 1]:]
		spam_test = spam_random[flag_spam[i]:flag_spam[i + 1]]
		ham_train = ham_random[:flag_ham[i]] + ham_random[flag_ham[i + 1]:flag_ham[len(flag_ham) - 1]] 
		ham_test = ham_random[flag_ham[i]:flag_ham[i + 1]]
		print i,len(spam_train),len(spam_test),len(ham_train),len(ham_test)
		print "len:",len(spam_train),len(spam_test),len(ham_train),len(ham_test)

		spam_prior_probability = len(spam_train)/float((len(spam_train) + len(ham_train)))
		
		words = {}

		#These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
		for s in spam_train:
			counted = []
			for word in parse(open(spam_directory + s)):
				if word not in words:
					counted.append(word)
					words[word] = {'spam': 2.0, 'ham': 1.0}
				elif word not in counted:
					counted.append(word)
					words[word]['spam'] += 1.0

		for h in ham_train:
			counted = []
			for word in parse(open(ham_directory + h)):
				if word not in words:
					counted.append(word)
					words[word] = {'spam': 1.0, 'ham': 2.0}
				elif word not in counted:
					counted.append(word)
					words[word]['ham'] += 1.0

		spam_length = len(spam_train) + 1
		ham_length = len(ham_train) + 1

		for word in words:
			words[word]['spam'] /= spam_length
			words[word]['ham'] /= ham_length

			#Write it to a dictionary output file.
		writedictionary(words, dictionary_filename + str(i) + ".dict")
		s_filter,s_prior = spamsort(spam_test, ham_test, 'sorted_spam', 'sorted_ham', words, spam_prior_probability) 
		data.append([s_filter,s_prior])
		print s_filter,s_prior
	writeToJson(data,"data.txt")

def is_spam(content, dictionary, spam_prior_probability):
	#TODO: Update this function. Right now, all it does is checks whether the spam_prior_probability is more than half the data. If it is, it says spam for everything. Else, it says ham for everything. You need to update it to make it use the dictionary and the content of the mail. Here is where your naive Bayes classifier goes.
	v_spam = math.log(spam_prior_probability)
	v_ham = math.log(1 - spam_prior_probability)
	for word in content:
		if dictionary.has_key(word):
			v_spam += math.log(dictionary[word]['spam'])
			v_ham += math.log(dictionary[word]['ham'])

	if v_spam > v_ham:
		return True
	else:
		return False

def spamsort(spam_test, ham_test, spam_directory, ham_directory, dictionary, spam_prior_probability):
	n1 = 0.0
	n2 = 0.0
	s_filter_spam = 0.0
	s_filter_ham = 0.0

	s_prior = 0.0
	spam_directory = "spam/"
	ham_directory = "easy_ham/"

	for m in spam_test:
		n1 += 1
		content = parse(open(spam_directory + m))
		spam = is_spam(content, dictionary, spam_prior_probability)
		if spam == False:
			s_filter_spam += 1
	s_filter_spam /= n1
	
	for m in ham_test:
		n2 += 1
		content = parse(open(ham_directory + m))
		spam = is_spam(content, dictionary, spam_prior_probability)
		if spam == True:
			s_filter_ham += 1

	s_filter_ham /= n2

	s_filter = (s_filter_ham + s_filter_spam)/2
	if spam_prior_probability < 0.5:
		s_prior = len(spam_test)*1.0/(n1+n2)
	else:
		s_prior = len(ham_test)*1.0/(n1+n2)
			
	return s_filter,s_prior

def compare(path = "data.txt"):
	data = readJson(path)
	s_filter = []
	s_prior = []
	for d in data:
		s_filter.append(d[0])
		s_prior.append(d[1])
	t,p = scipy.stats.ttest_rel(s_filter, s_prior)
	print "p:",p
	x = np.array([i for i in range(10)])
	k,p = scipy.stats.mstats.normaltest(s_filter)
	print "xxxxx:",p
	print "ss:",np.var(s_filter)
	k,p = scipy.stats.mstats.normaltest(s_prior)
	print "xxxxx:",p
	print "ss:",np.var(s_prior)


	plt.title("compare spam filter to prior probability")
	plt.xlabel("cross validation number")
	plt.ylabel("error rate")
	plt.ylim(0,0.6)
	plt.plot(x,s_filter, color="blue", linewidth=1, linestyle="-", label="spam filter")
	plt.plot(x,s_prior, color="red", linewidth=1, linestyle="-", label="prior probability")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	#Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
	training_spam_directory = "spam/" #sys.argv[1]
	training_ham_directory = "easy_ham/"#sys.argv[2]
	
	test_mail_directory = "easy_ham/"#"spam/"#sys.argv[3]
	test_spam_directory = 'sorted_spam'
	test_ham_directory = 'sorted_ham'
	
	if not os.path.exists(test_spam_directory):
		os.mkdir(test_spam_directory)
	if not os.path.exists(test_ham_directory):
		os.mkdir(test_ham_directory)
	
	dictionary_filename = "dictionary"
	
	#makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)
	compare()
