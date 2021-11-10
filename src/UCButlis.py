import numpy as np
import sys
import pandas as pd

"""bandits算法用的"""

def isInvertible(S):
	return np.linalg.cond(S) < 1 / sys.float_info.epsilon

def edge_probability(n):
	return 3 * np.log(n) / n

def is_power2(n):
	return n > 0 and ((n & (n - 1)) == 0)

def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d-1))   #正态分布
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis = 1), np.ones(np.shape(x)[1])))/np.sqrt(2), np.ones((num_items, 1))/np.sqrt(2)), axis = 1)
    return x


if __name__== "__main__":
	# rnames = ['userId', 'movieId', 'rating', 'timestamp']
	# rating = pd.read_table('D:/PyCharm Community Edition 2019.1.3/OnlineClusteringOfBandits/Movielens/ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')
	# rating.to_csv('ratings.csv')
	# mnames=['movieId','title','genres']
	# movies = pd.read_table(
	# 	'D:/PyCharm Community Edition 2019.1.3/OnlineClusteringOfBandits/Movielens/ml-1m/movies.dat', sep='::',
	# 	header=None, names=mnames, engine='python')
	# movies.to_csv('movies.csv')
	# unames=['userId','gender','age','occupation','zip']
	# users = pd.read_table(
	# 	'D:/PyCharm Community Edition 2019.1.3/OnlineClusteringOfBandits/Movielens/ml-1m/users.dat', sep='::',
	# 	header=None, names=unames, engine='python')
	# users.to_csv('users.csv')

	import read_movielens

