#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Prem Seetharaman (replace your name here)

import sys
import numpy as np
import os
import shutil
import math

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
	
	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	
	words = {}
	

	#These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
	for s in spam:
		counted = []
		for word in parse(open(spam_directory + s)):
			if word not in words:
				counted.append(word)
				words[word] = {'spam': 2.0, 'ham': 1.0}
			elif word not in counted:
				counted.append(word)
				words[word]['spam'] += 1.0

	for h in ham:
		counted = []
		for word in parse(open(ham_directory + h)):
			if word not in words:
				counted.append(word)
				words[word] = {'spam': 1.0, 'ham': 2.0}
			elif word not in counted:
				counted.append(word)
				words[word]['ham'] += 1.0

	spam_length = len(spam) + 1
	ham_length = len(ham) + 1
	
	for word in words:
		words[word]['spam'] /= spam_length
		words[word]['ham'] /= ham_length
		#Write it to a dictionary output file.
	writedictionary(words, dictionary_filename)
	return words, spam_prior_probability

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



def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
	mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f))]
	n = 0.0
	s = 0.0
	for m in mail:
		n += 1
		content = parse(open(mail_directory + m))
		spam = is_spam(content, dictionary, spam_prior_probability)
		if spam:
			s += 1
			shutil.copy(mail_directory + m, spam_directory)
		else:
			shutil.copy(mail_directory + m, ham_directory)


if __name__ == "__main__":
	#Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
	training_spam_directory = sys.argv[1]
	training_ham_directory = sys.argv[2]
	
	test_mail_directory = sys.argv[3]
	test_spam_directory = 'sorted_spam'
	test_ham_directory = 'sorted_ham'
	
	if not os.path.exists(test_spam_directory):
		os.mkdir(test_spam_directory)
	if not os.path.exists(test_ham_directory):
		os.mkdir(test_ham_directory)
	
	dictionary_filename = "dictionary.dict"
	
	#create the dictionary to be used
	dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)
	#sort the mail
	spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability) 
