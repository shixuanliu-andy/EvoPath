import random
import json
rootpath = './data/yagosmall/'
class node:
	def __init__(self,x,y=None,pre=None,next=None,level=1):
		self.x=x
		self.y=y
		self.pre=pre
		self.next=next
		self.level=level
	def back(self):
		M=[self.x]
		t = self.pre
		while t!=None:
			M.append(t.x)
			t=t.pre
		return M[::-1]
class typetree:
	def __init__(self):
		self.node_father = {}
		self.node_child = {}
		self.ontology = []
		with open(rootpath + 'yagoTaxonomy.tsv', 'r') as fin:
			for i in fin.readlines():
				line = i.strip().split('\t')
				if len(line) == 4:
					c1, r, c2 = line[1], line[2], line[3]
					self.ontology.append([c1, c2])
					self.node_father[c1] = []
					self.node_father[c2] = []
					self.node_child[c1] = []
					self.node_child[c2] = []
		for i in self.ontology:
			c1, c2 = i[0], i[1]
			self.node_father[c1].append(c2)
			self.node_child[c2].append(c1)
	def find_LCA(self,ty):
		stack=[]
		stack.append(node(ty))
		res=[]
		while len(stack)>0:
			t=stack.pop()
			if t.x in self.node_father:
				if len(self.node_father[t.x])==0:
					res.append(t.back())
				else:
					for next in self.node_father[t.x]:
						next_node=node(next,level=t.level+1)
						next_node.pre=t
						stack.append(next_node)
		return res
	def filter_typelist(self,typelist):
		root_type=[self.find_LCA(i) for i in typelist]
		res=set([])
		for i in root_type:
			for j in i:
				M=[]
				for leaf in j:
					if leaf in typelist:
						M.append(leaf)

				M=tuple(M)
				res.add(M)

		return res


if __name__=='__main__':
	tree=typetree()
	with open(rootpath+'yagoFacts_ent2type.json','r') as fin:
		ent2type=json.load(fin)
	ent=list(ent2type.keys())
	for i in range(20):
		t=random.choice(ent)
		typelist=ent2type[t]
		print('typelist:',typelist)
		res=tree.filter_typelist(typelist)
		print('filter :',res)

