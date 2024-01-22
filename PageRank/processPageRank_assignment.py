#!/opt/local/bin/python3
# -*- coding: utf-8 -*-

import argparse
import sys
import re
from scipy.sparse import coo_matrix
import numpy
import time


def pagerank(graph, beta=0.2, epsilon=1.0e-8):

	#Fill the initializations
	inlink_map=[]

	for j in range(graph.shape[0]):
		print("Making in-link map of %d\r"%(j), end=' ', file=sys.stderr)
		inlink_map.append(graph.getcol(j).nonzero()[0])

	out_degree=numpy.array(graph.sum(axis=1))

	print("\nLink-map done!", file=sys.stderr)
	ranks=numpy.ones(graph.shape[0])/graph.shape[0]

	new_ranks = {}

	delta = 1.0
	n_iterations = 0

	# Pràctica 1 -  PageRank
	# Mineria de Dades, Curs 2021-22
	# @author Oscar Julian Ponte(oscar.julian)
	# 17 d'Octubre 2021

	import sklearn.model_selection as model_selection
	import sklearn.decomposition._pca
	import sklearn.neighbors
	import sklearn.metrics

	# import matlplotlib.pyplot as plt

	matriuA = []

	# [1] README
	# El que farem primer serà omplir els valors que podem carregar prèviament.
	# ((1-beta) * (1 / graph.shape[0])) == ((1-beta)/graph.shape[0])
	matriuA = numpy.ones((graph.shape[0], graph.shape[0])) * ((1 - beta) / graph.shape[0])

	# Primerament el codi no utilitzavem aquesta mini 'optimització' per a pre-omplir la matriu A amb els 1 i aquestes operacions.
	# Sense omplir-la podia arribar a tardar gairebé 5min en crear la matriu 'A' abans de començar les iteracions.

	# [2] README
	# Recorrem els valors de dins de cada casella de inlink_map que son tambe amb els que volem treballar:
	for i in range(len(inlink_map)):
		for j in range(len(inlink_map[i])):
			matriuA[i][inlink_map[i][j]] += (beta / out_degree[inlink_map[i][j]])

	while delta > epsilon:
		new_ranks = numpy.zeros(graph.shape[0])

		# [3] README
		# Degut a que ja tenim la matriuA construida, podem resoldre l'equació directament multiplicant:
		new_ranks = matriuA @ ranks

		delta=numpy.sqrt(numpy.sum(numpy.power(ranks-new_ranks,2)))
		ranks,new_ranks=new_ranks,ranks
		print("\nIteration %d has been computed with an delta of %e (epsilon=%e)"%(n_iterations,delta,epsilon), file=sys.stderr)
		n_iterations += 1

	print()
	rranks={}
	for i in range(ranks.shape[0]):
		rranks[i]=ranks[i]
	return rranks, n_iterations

def processInput(filename):

	webs={}
	rows=numpy.array([],dtype='int8')
	cols=numpy.array([],dtype='int8')
	data=numpy.array([],dtype='float32')
	for line in open(filename,'r'):
		line=line.rstrip()
		
		m=re.match(r'^n\s([0-9]+)\s(.*)',line)
		if m:
			webs[int(m.groups()[0])]=m.groups()[1]
			continue
		m=re.match(r'^e\s([0-9]+)\s([0-9]+)',line)
		if m:
			rows=numpy.append(rows,int(m.groups()[0]))
			cols=numpy.append(cols,int(m.groups()[1]))
			data=numpy.append(data,1)
			
	graph=coo_matrix((data,(rows,cols)),dtype='float32',shape=(max(webs.keys())+1,max(webs.keys())+1))
	return (webs,graph)

if __name__ == "__main__":
		parser = argparse.ArgumentParser(description="Analyze web data and output PageRank")
		parser.add_argument("file", type=str, help="file to be processed")
		parser.add_argument("--beta", type=float, help="β value to be considered",default=0.85)
		args = parser.parse_args()

		webs,graph=processInput(args.file)
		start = time.time()
		ranks,n_iterations=pagerank(graph,args.beta)
		end = time.time()
		print("It took %f seconds to converge"%(end - start), file=sys.stderr)
		keys=[list(ranks.keys())[x] for x in numpy.argsort(list(ranks.values()))[-1::-1]]
		values=[list(ranks.values())[x] for x in numpy.argsort(list(ranks.values()))[-1::-1]]
		for p,(k,v) in enumerate(zip(keys,values)): 
			print("[%d] %s:\t%e"%(p,webs[k],v))
