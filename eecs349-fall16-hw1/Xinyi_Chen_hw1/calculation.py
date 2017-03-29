import math

#Default attribute is CLASS to calculate Entropy(S)
def entropy(data,attribute="CLASS"):
	trues = [0.0,0.0];
	falses = [0.0,0.0];
	
	for row in data:
		if row[attribute]=='true':
			if row["CLASS"]=='false':
				trues[0] += 1
			else:
				trues[1] += 1
		elif row[attribute]=='false':
			if row["CLASS"]=='false':
				falses[0] += 1
			else:
				falses[1] += 1

	total_true = trues[0]+trues[1]
	total_false = falses[0]+falses[1]
	total = total_true+total_false;

	if attribute=="CLASS":
		total_true = total
		total_false = total

	entropy_true = 0.0;
	entropy_false = 0.0;
	
	for n in trues:
		if n!=0:
			entropy_true += -(n/total_true)*math.log(n/total_true,2)

	for n in falses:
		if n!=0:
			entropy_false += -(n/total_false)*math.log(n/total_false,2)

	if attribute=="CLASS":
		entropy_true += entropy_false
	
	#return trues[0],trues[1],total_true,falses[0],falses[1],total_false,entropy_true,entropy_false
	return total_true,total_false,entropy_true,entropy_false

def information_gain(entropy_s,entropy_list):
	total = entropy_list[0]+entropy_list[1]

	return entropy_s - entropy_list[0]/total*entropy_list[2] - entropy_list[1]/total*entropy_list[3]

def mode(training_set):
		trues = 0;
		falses = 0;
		
		for n in training_set:
			if n['CLASS'] == 'true':
				trues += 1
			else:
				falses += 1

		if trues > falses:
			return 'true'
		else:
			return 'false'

def prior_probability(training_set,testing_set):
	tures = 0
	falses = 0
	for row in training_set:
		if row['CLASS'] == 'true':
			tures += 1
		else:
			falses += 1

	if tures > falses:
		prior_probability = 'true'
	else:
		prior_probability = 'false'
	
	value = 0.0
	for n in testing_set:
		if n['CLASS'] == prior_probability:
			value += 1
	if len(testing_set) !=0:
		return value/len(testing_set)
	else:
		return 0.0


		