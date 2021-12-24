import numpy as np
from src.TSutil import generate_items

def get_best_reward(items, theta):
	return np.max(np.dot(items, theta))

class Environment:
	# p: frequency vector of users
	def __init__(self, L, d, m, num_users, theta,negatives,items_em,train):
		self.L = L
		self.d = d
		# self.p = p # probability distribution over users
		self.negatives = negatives
		self.items_em = items_em
		self.theta = theta
		self.can_train = {}    #储存user:组为key的items的向量
		self.train_rating = train

	def get_items(self):
		self.items = generate_items(num_items = self.L, d = self.d)
		return self.items

	def feedback(self, i, k):
		x = self.items_em[k, :]
		thetaone = self.theta[i, :]
		r = np.dot(thetaone, x)
		y = self.train_rating[i, k]
		if y == -1:
			return -2, -2
		# y = np.random.binomial(1, r)   #一次伯努利试验，p=r
#		br = get_best_reward(self.items, self.theta[i])
		return y, r

	def generate_users(self):
		X = np.random.multinomial(1, self.p)
		I = np.nonzero(X)[0]
		return I
