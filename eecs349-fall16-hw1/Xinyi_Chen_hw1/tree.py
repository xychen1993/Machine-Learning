import calculation

class node(object):
	def __init__(self, parent = 'root', attribute = -1, trueChild=None, falseChild=None):
		self.attribute = attribute
		self.parent = parent
		self.trueChild = trueChild
		self.falseChild = falseChild


class tree(object):
	def __init__(self,attributes,training_set):
		self.root = self.build_tree(attributes,training_set)

	def print_tree(self,root=None):

		if root != None:
			if root.attribute == 'true' or root.attribute == 'false':
				print 'parent: ',root.parent,' - '
			else:
				trueChild = root.trueChild.attribute
				falseChild = root.falseChild.attribute
				if trueChild  == 'true' or trueChild  == 'false':
					trueChild = 'leaf'
				if falseChild  == 'true' or falseChild  == 'false':
					falseChild = 'leaf'
				
				print 'parent: ',root.parent,'	attribute:',root.attribute, '	trueChild: ',trueChild,'	falseChild: ',falseChild
			
			self.print_tree(root.trueChild)
			self.print_tree(root.falseChild)
		
	def build_tree(self,attributes,training_set,parent='root',default='false'):

		if len(training_set) == 0:
			return node(parent,default)
		
		flag = training_set[0]['CLASS']
		for n in training_set:
			if n['CLASS'] != flag:
				flag = -1
				break
		if flag != -1:
			return node(parent,flag)

		if len(attributes)==1:
			return node(parent,calculation.mode(training_set))	

		information_gain = {}
		entropy_s = calculation.entropy(training_set)[2]
	
		info_gain_biggest = ['',-1]
		for attribute in attributes:
			if attribute != 'CLASS':
				temp = calculation.information_gain(entropy_s,calculation.entropy(training_set,attribute))
				information_gain[attribute] = temp
				if temp > info_gain_biggest[1]:
					info_gain_biggest = [attribute,temp]

		training_set_true = []
		training_set_false = []

		for n in training_set:
			if n[info_gain_biggest[0]] == 'true':
				training_set_true.append(n)
			else:
				training_set_false.append(n)

		
		attributes.remove(info_gain_biggest[0])

		return node(parent,info_gain_biggest[0],self.build_tree(attributes,training_set_true,info_gain_biggest[0],calculation.mode(training_set_true)),self.build_tree(attributes,training_set_false,info_gain_biggest[0],calculation.mode(training_set_false)))

	def test_tree(self,sample,root=None):
		if root.attribute == 'true' or root.attribute == 'false':
			return root.attribute
		else:
			if sample[root.attribute] == 'true':
				return self.test_tree(sample,root.trueChild)
			else:
				return self.test_tree(sample,root.falseChild)
				
		
				



				

	
		
	



