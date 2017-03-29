import spellcheck
import random
#delete # at the bottom of this file to run functions

#hw2_problem_2B run by terminal: python spellcheck.py <ToBeSpellCheckedFileName> 3esl.txt

def hw2_problem_2A_find_closest_word():
	# takes about 50 seconds to run
	spellcheck.determine_distance = 0
	data  = spellcheck.open_file("wikipediatypoclean.txt","3esl.txt")
	random_words = random.sample(data[0],10)
	# randomly choose 10 words from to be corrected words
	for word in random_words:
		print word,"->",spellcheck.find_closest_word(word,data[3])

def hw2_problem_2A_levenshtein_distance():
	print "levenshtein distance between hello and aeloo is:",spellcheck.levenshtein_distance("hello","aeloo",1,1,1)


def hw2_problem_2C():
	#takes about 80 seconds to run
	spellcheck.determine_distance = 0
	data  = spellcheck.open_file("wikipediatypoclean.txt","3esl.txt","false",1)
	print '%.2f%%'%(spellcheck.measure_error(data[0],data[1],data[2])*100)

def hw2_problem_3C():
	#takes about 1000 seconds to run
	spellcheck.determine_distance = 0
	data  = spellcheck.open_file("wikipediatypoclean.txt","3esl.txt","false",1)
	spellcheck.hw2_problem3(data)

def hw2_problem_4A():
	print "qwerty levenshtein distance between hello and aeloo is:",spellcheck.qwerty_levenshtein_distance("frog","log",1,1)

def hw2_problem_4B():
	#takes about 400 seconds to run
	spellcheck.determine_distance = 1
	data  = spellcheck.open_file("wikipediatypoclean.txt","3esl.txt","false",1)
	spellcheck.hw2_problem4(data)

#hw2_problem_2A_find_closest_word()
#hw2_problem_2A_levenshtein_distance()
#hw2_problem_2C()
#hw2_problem_3C()
hw2_problem_4A()
#hw2_problem_4B()