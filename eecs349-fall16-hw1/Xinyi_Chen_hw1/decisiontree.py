import csv
import random
import tree
import calculation
import sys

#Change values here
inputFileName = sys.argv[1]
trainingSetSize = int(sys.argv[2])
numberOfTrials = int(sys.argv[3])
verbose = int(sys.argv[4])

def open_file(file_path=inputFileName):
	with open(file_path) as csvfile:
		reader  = csv.DictReader(csvfile, delimiter='	')
		mydic = []
		attributes = []

		for line in reader:
			mydic.append(line)

		for key in mydic[0]:
			attributes.append(key)
	
	return attributes,mydic


def run(attributes,data,trainingSetSize,numberOfTrials,verbose):
	i = 0
	mean_tree = 0.0
	mean_prior = 0.0

	if len(data) < trainingSetSize:
		print 'Error: Training set size must less than amount of examples(%d) '%(len(data))
		return 0

	while i < numberOfTrials:
		training_set = random.sample(data,trainingSetSize)
		testing_set = []
		
		for n in data:
			if n not in training_set:
				testing_set.append(n)

		for key in training_set[0]:
			if key not in attributes:
				attributes.append(key)

		decision_tree =tree.tree(attributes,training_set)
		print 'TRIAL NUMBER: ',i
		print '--------------------'
		print 'DECISION TREE STRUCTURE:'
		
		decision_tree.print_tree(decision_tree.root)

		trues = 0.0
		total = 0

		for sample in testing_set:
			total += 1
			if sample['CLASS'] == decision_tree.test_tree(sample,decision_tree.root):
				trues += 1
		if total != 0:
			rate = trues/total
		else:
			rate = 0.0

		mean_tree += rate
		print '	Percent of test cases correctly classified by a decision tree built with ID3 = %.2f%%'  %(rate*100)
		rate = calculation.prior_probability(training_set,testing_set)
		mean_prior += rate
		print '	Percent of test cases correctly classified by using prior probabilities from the training set =  %.2f%%'  %(rate*100)

		i += 1
		if verbose == 1:
			print '\ntraning set: ',training_set
			print '\ntesting set: ', testing_set

		print '\n'

	print 'example file used = ',inputFileName
	print 'number of trials = ',numberOfTrials
	print 'training set size for each trial = ',len(training_set)
	print 'testing set size for each trial = ',len(data) - len(training_set)
	print 'mean performance of decision tree over all trials = %.2f%% correct classification' %(mean_tree/numberOfTrials*100)
	print 'mean performance of using prior probability derived from the training set = %.2f%% correct classification' %(mean_prior/numberOfTrials*100)

attributes = open_file()[0]
raw_data = open_file()[1]
run(attributes,raw_data,trainingSetSize,numberOfTrials,verbose)
