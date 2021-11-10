import numpy as np
from src.UCButlis import generate_items

def get_best_reward(items, theta):
	return np.max(np.dot(items, theta))

class Environment:
	# p: frequency vector of users
	def __init__(self, L, d, m, num_users, p, theta):
		self.L = L
		self.d = d
		self.p = p # probability distribution over users

		self.items_em = np.load("./item_em.npy")
		self.theta = theta

	def get_items(self):
		self.items = generate_items(num_items = self.L, d = self.d)
		return self.items

	def feedback(self, i, k):
		x = self.items[k, :]
		r = np.dot(self.theta[i], x)
		y = np.random.binomial(1, r)   #一次伯努利试验，p=r
		br = get_best_reward(self.items, self.theta[i])
		return y, r, br

	def generate_users(self):
		X = np.random.multinomial(1, self.p)
		I = np.nonzero(X)[0]
		return I
