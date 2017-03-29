import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

if len(sys.argv) > 1:
	to_be_corrected = sys.argv[1]
	dictionary_path = sys.argv[2]
else:
	to_be_corrected = ""
	dictionary_path = ""

deletion_cost = 1
insertion_cost = 1
substitution_cost = 1
determine_distance = 0
#determine_distance = 0 use levenshtein_distance else use qwerty_levenshtein_distance todetermine distance

#Build BK_Tree to speed up search
class node(object):
	def __init__(self, parent = "hello", word = "",  children = None):
		self.word = word
		self.parent = parent
		self.children = children


class bk_tree(object):
	def __init__(self,word):
		self.root = node("hello",word,{})

	def print_tree(self,root=None):
		if root == None:
			return 
		else:
			print 'parent: ',root.parent,'	word:',root.word, '	Children: ',root.children
			for key in root.children:
				self.print_tree(root.children[key])

		
	def build_bk_tree(self,root,word,parent='root'):
		if root == None:
			return

		global determine_distance
		if determine_distance == 0:
			distance = levenshtein_distance(root.word,word,deletion_cost,insertion_cost,substitution_cost)
		else:
			distance = qwerty_levenshtein_distance(root.word,word,deletion_cost,insertion_cost,substitution_cost)


		if distance == 0:
			return
		for key in root.children:
			if key == distance:
				self.build_bk_tree(root.children[key],word,root.children[key].word)
				return

		root.children[distance] = node(parent,word,{})
		return

	def search_tree(self,root,word,parent_distance):
		if root == None:
			return parent_distance,root.parent
		
		global determine_distance
		if determine_distance == 0:
			distance = levenshtein_distance(root.word,word,deletion_cost,insertion_cost,substitution_cost)
		else:
			distance = qwerty_levenshtein_distance(root.word,word,deletion_cost,insertion_cost,substitution_cost)

		if distance == 0:
			return 0,root.word

		if  len(root.children) == 0:
			if distance > parent_distance:
				return parent_distance,root.parent
			else:
				return distance,root.word


		close1 = -1
		close2 = 0xFFF
		close_middle = 0xFFF
		min_distance = [distance,root.word]

		for key in root.children:
			if key == distance:
				 close_middle = self.search_tree(root.children[key],word,distance)
				 if close_middle[0] < min_distance[0]:
				 	min_distance = close_middle

			if key < distance:
				close1 = key
			if key > distance and key < close2:
				close2 = key
		
		close_left = 0xFFF
		close_right = 0xFFF
		
		if close1 != -1:
			close_left = self.search_tree(root.children[close1],word,distance)
			if close_left[0] < min_distance[0]:
				min_distance = close_left
		if close2 != 0xFFF:
			close_right = self.search_tree(root.children[close2],word,distance)
			if close_right[0] < min_distance[0]:
				min_distance = close_right

		return min_distance


#opeb files and build bk_tree
def open_file(to_be_corrected = "testcor.txt",dictionary_path="testdic.txt", build_bktree = "true",subset = 0):
	with open(to_be_corrected) as csvfile:
		reader  = csv.reader(csvfile, delimiter='	')
		words = []
		ture_words = []

		for row in reader:
			words.append(row[0])
			ture_words.append(row[1])

	
	with open(dictionary_path) as csvfile:
		reader  = csv.reader(csvfile, delimiter='	')
		dictionary = []

		bk_tree_instance = bk_tree("hello")
		
		for row in reader:
			if subset == 1 and len(row[0]) == 1 and ord(row[0]) == 68:
				break
			if len(row[0]) == 1 and ord(row[0]) > 64 and ord(row[0]) < 91:
				continue
			else:
				if build_bktree == "true":
					bk_tree_instance.build_bk_tree(bk_tree_instance.root,row[0],bk_tree_instance.root.word)
				else:
					dictionary.append(row[0])

	return words,ture_words,dictionary,bk_tree_instance

def levenshtein_distance(string1 = "abc",string2 = "acce",deletion_cost = 1,insertion_cost = 1,substitution_cost = 1):
	string1 = string1.replace(',',"").replace('|',"").replace('#',"").replace(' ',"").replace('.',"").replace('\'',"").replace('-',"").replace('/',"")
	string2 = string2.replace(',',"").replace('|',"").replace('#',"").replace(' ',"").replace('.',"").replace('\'',"").replace('-',"").replace('/',"")

	len1 = len(string1)
	len2 = len(string2)
	matrix = np.zeros((len1+1,len2+1),dtype = int)

	i = len2
	while i >= 0:
		matrix[0][i] = deletion_cost * i
		i -= 1

	i = 0
	for row in matrix:
		row[0] = insertion_cost * i

		if i == 0:
			i += 1
			continue

		j = 1
		while j <= len2:

			if string1[i-1].lower() == string2[j-1].lower():
				matrix[i][j] = matrix[i-1][j-1]
			else:
				matrix[i][j] = min(matrix[i-1][j-1]+substitution_cost,matrix[i][j-1]+insertion_cost,matrix[i-1][j]+deletion_cost)
			j += 1

		i += 1
	return matrix[len1][len2]

#Manhattan distance
def qwerty_levenshtein_distance(string1, string2, deletion_cost = 1, insertion_cost = 1,nothing = 1):
	keyboard = {'Q':[0,0],'W':[0,1],'E':[0,2],'R':[0,3],'T':[0,4],'Y':[0,5],'U':[0,6],'I':[0,7],'O':[0,8],'P':[0,9],
				'A':[1,0],'S':[1,1],'D':[1,2],'F':[1,3],'G':[1,4],'H':[1,5],'J':[1,6],'K':[1,7],'L':[1,8],
				'Z':[2,0],'X':[2,1],'C':[2,2],'V':[2,3],'B':[2,4],'N':[2,5],'M':[2,6]}
	
	string1 = string1.replace(',',"").replace('|',"").replace('#',"").replace(' ',"").replace('.',"").replace('\'',"").replace('-',"").replace('/',"")
	string2 = string2.replace(',',"").replace('|',"").replace('#',"").replace(' ',"").replace('.',"").replace('\'',"").replace('-',"").replace('/',"")

	len1 = len(string1)
	len2 = len(string2)
	matrix = np.zeros((len1+1,len2+1),dtype = int)

	i = len2
	while i >= 0:
		matrix[0][i] = deletion_cost * i
		i -= 1

	i = 0
	for row in matrix:
		row[0] = insertion_cost * i

		if i == 0:
			i += 1
			continue

		j = 1
		while j <= len2:
			curr1 = string1[i-1]
			curr2 = string2[j-1]
			if (64<ord(curr1)<91 or 96<ord(curr1)<123) and (64<ord(curr2)<91 or 96<ord(curr2)<123):
				if curr1.upper() == curr2.upper():
					matrix[i][j] = matrix[i-1][j-1]
				else:
					substitution_cost = abs(keyboard[curr1.upper()][0] - keyboard[curr2.upper()][0])+abs(keyboard[curr1.upper()][1] - keyboard[curr2.upper()][1])
					matrix[i][j] = min(matrix[i-1][j-1]+substitution_cost,matrix[i][j-1]+insertion_cost,matrix[i-1][j]+deletion_cost)
			j += 1
			

		i += 1
	return matrix[len1][len2]

def find_closest_word(string1,bk_tree_instance):

	return bk_tree_instance.search_tree(bk_tree_instance.root,string1,0xFFF)[1]


def correct(to_be_corrected,dictionary_path):

	f=file(to_be_corrected,"r")
	data  = f.read()
	corrected = ""

	with open(dictionary_path) as csvfile:
		reader  = csv.reader(csvfile, delimiter='	')
		dictionary = []

		bk_tree_instance = bk_tree("hello")
		
		for row in reader:
			bk_tree_instance.build_bk_tree(bk_tree_instance.root,row[0],bk_tree_instance.root.word)
				

	word = ""
	for char in data:
		if 64 < ord(char) < 91 or 96 < ord(char) < 123:
			word += char
		else:
			if len(word) > 0:
				right = bk_tree_instance.search_tree(bk_tree_instance.root,word,0xFFF)[1]
				corrected += right
			word = ""
			corrected += char

	corrected += word

	f=file('corrected.txt',"w+")
	f.writelines(corrected)

   	return

def measure_error(typos, truewords, dictionarywords):

	bk_tree_instance = bk_tree("hello")
		
	for word in dictionarywords:
		if len(word) == 1 and ord(word) > 64 and ord(word) < 91:
			continue
		else:
			bk_tree_instance.build_bk_tree(bk_tree_instance.root,word,bk_tree_instance.root.word)
	
	total = 0
	trues = 0.0
	for word in typos:
		closest = bk_tree_instance.search_tree(bk_tree_instance.root,word,0xFFF)[1]
		truewords_array = truewords[total].split(',' or '|' or '#' or ' ' )
		
		for true_word in truewords_array:
			true_word = true_word.replace(',',"").replace('|',"").replace('#',"").replace(' ',"").replace('.',"").replace('\'',"").replace('-',"").replace('/',"")
			if true_word == closest:
				trues += 1
				break

		total +=1
	
	return 1 - trues/total

#find best performance of combinations of three values
def hw2_problem3(data):	
	global determine_distance
	determine_distance = 0

	print "No.\t","deletion_cost\t","insertion_cost\t","substitution_cost\t","error_rate\t","runtime\t"
	global deletion_cost
	global insertion_cost
	global substitution_cost
	rates = []
	times = []
	x = []

	value = [0,1,2,4]
	i = 0
	No = 0
	while  i < 4:
		deletion_cost = value[i]
		j = 0
		while j < 4:
			insertion_cost = value[j]
			h = 0
			while h < 4:
				substitution_cost = value[h]
				start = time.time()
				error_rate = measure_error(data[0],data[1],data[2])
				time_takes = time.time() - start
				print No,"\t",deletion_cost,"\t",insertion_cost,"\t",substitution_cost,"\t",'%.2f%%\t'%(error_rate*100),'%.2fs'%(time_takes)
	 			times.append(time_takes)
	 			rates.append(error_rate*100)
	 			x.append(No)
	 			h += 1
	 			No += 1
			j+=1
		i += 1
	plt.title("hw2_problem_3C")
	plt.plot(x,rates,label="error rate(%)",color="red",linewidth=1)
	plt.plot(x,times,label="time(seconds)",color="blue",linewidth=1)
	plt.legend()		
	plt.show()
	return 

#use qwerty_levenshtein_distance instead of levenshtein_distance and find best performance
def hw2_problem4(data):
	global determine_distance
	determine_distance = 1

	print "No.\t","deletion_cost\t","insertion_cost\t","error_rate\t","runtime\t"
	global deletion_cost
	global insertion_cost
	rates = []
	times = []
	x = []

	value = [1,2,3,4]
	i = 0
	No = 0
	while  i < 4:
		deletion_cost = value[i]
		j = 0
		while j < 4:
			insertion_cost = value[j]
			h = 0
			start = time.time()
			error_rate = measure_error(data[0],data[1],data[2])
			time_takes = time.time() - start
			print No,"\t",deletion_cost,"\t",insertion_cost,"\t",'%.2f%%\t'%(error_rate*100),'%.2fs'%(time_takes)
	 		rates.append(error_rate*100)
	 		times.append(time_takes)
	 		x.append(No)
	 		No += 1
			j+=1
		i += 1
	plt.title("hw2_problem4B")
	plt.plot(x,rates,label="error rate(%)",color="red",linewidth=1)
	plt.plot(x,times,label="time(seconds)",color="blue",linewidth=1)
	plt.legend()	
	plt.show()
	return 

if len(to_be_corrected) > 0:
	determine_distance = 0
	correct(to_be_corrected,dictionary_path)
