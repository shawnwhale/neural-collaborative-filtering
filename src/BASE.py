import numpy as np
import random


"""bandits算法用的"""
class Thompson:
    def __init__(self):
        self.para_u = np.zeros(shape=(6040, 18))
        self.para_k = np.zeros(shape=(6040, 18))

    def select(self,u):
        choice = np.argmax(np.random.normal(self.para_u[u,:], 1/(self.para_k[u,:]+1)))
        return choice

    def update(self,u, c ,y, r):
        if y == 1:
            self.para_u[u,c] =  (self.para_u[u,c]* self.para_k[u,c] + r)/ (self.para_k[u,c]+2)
        else:
            self.para_u[u, c] = (self.para_u[u, c] * self.para_k[u, c] - 0.1) / (self.para_k[u, c] + 2)
        self.para_k[u,c] = self.para_k[u,c]+1
        return

class Base:
    # Base agent for online clustering of bandits
    def __init__(self, d, T):
        self.d = d
        self.T = T
        self.beta = np.sqrt(self.d * np.log(self.T / self.d)) # parameter for select item
        self.rewards = np.zeros(self.T)
        self.best_rewards = np.zeros(self.T)
        self.thompson = Thompson()

    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        result = np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1)

        return np.argmax(result)

    def recommend(self, i, items, t):
        # items is of type np.array (L, d)
        # select one index from items to user i
        return

    def store_info(self, i, x, y, t, r, br):
        return

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta

    def update(self, t):
        return

    def run(self, envir):
        return

# class LinUCB(Base):
#     def __init__(self, d, T):
#         super(LinUCB, self).__init__(d, T)
#         self.S = np.eye(d)
#         self.b = np.zeros(d)
#         self.Sinv = np.eye(d)
#         self.theta = np.zeros(d)
#
#     def recommend(self, i, items, t):
#         return self._select_item_ucb(self.S, self.Sinv, self.theta, items, t, t)
#
#     def store_info(self, i, x, y, t, r, br):
#         self.rewards[t] += r
#         self.best_rewards[t] += br
#         # x是user_features
#         self.S += np.outer(x, x)
#         self.b += y * x
#
#         self.Sinv, self.theta = self._update_inverse(self.S, self.b, self.Sinv, x, t)

class LinUCB_IND(Base):
    # each user is an independent LinUCB
    def __init__(self, nu, d, T):
        super(LinUCB_IND, self).__init__(d, T)
        self.S = {i:np.eye(d) for i in range(nu)}
        self.b = {i:np.zeros(d) for i in range(nu)}
        self.Sinv = {i:np.eye(d) for i in range(nu)}
        self.theta = {i:np.zeros(d) for i in range(nu)}
        self.item_train = {i: np.zeros(d) for i in range(nu)}
        self.N = np.zeros(nu)

    def result_print(self, envir,i,k):
        items = envir.items_em
        t=1000
        for num in range(k):
            kk = self.recommend(i=i, items=items, t=t)
            x = items[kk]
            print(np.dot(x, self.theta[i]))
            print(envir.train_rating[i, kk])

    def recommend(self, i, items, t):
        return self._select_item_ucb(self.S[i], self.Sinv[i], self.theta[i], items, self.N[i], t)

    def store_info(self, i, x, y, t, r):
        if y == -2 and r == -2:
            pass
        else:
            self.rewards[t] += r


            self.S[i] += np.outer(x, x)
            self.b[i] += y * x
            self.N[i] += 1

            self.Sinv[i], self.theta[i] = self._update_inverse(self.S[i], self.b[i], self.Sinv[i], x, self.N[i])

    def run(self, envir):
        dim_np =  np.load('./item_dim.npy')
        item_embedding_stan = envir.items_em
        for t in range(1,self.T):
            if t % 5000 == 0:
                print(t // 5000, end = ' ')
            # self.I = envir.generate_users()
            # for i in self.I:
            i = 0
            items = envir.items_em
            if t == 1:
                # 都摇一遍
                for item_seq in range(3706):
                    x = items[item_seq]
                    y, r = envir.feedback(i=i, k=item_seq)
                    self.store_info(i=i, x=x, y=y, t=t, r=r)
            choice = self.thompson.select(i)
            candidate = dim_np[:, choice]
            candidate = np.nonzero(candidate)[0]
            index = []
            for j in candidate:
                index.append(j)
            index = set(index)
            # 每个user有不同的item池
            key = str(i) + ":" + str(choice)
            if key in envir.can_train:
                items = envir.can_train.get(key)
            else:
                nega_sam = set(envir.negatives.ix[i,'negative_samples'])
                for j in nega_sam:
                    index.discard(j)
                this_items_em = np.zeros(shape=(3706, 8))
                for j in index:
                    this_items_em[j, :] = item_embedding_stan[j, :]
                envir.can_train[key] = this_items_em
                items = this_items_em

            kk = self.recommend(i=i, items=items, t=t)
            x = items[kk]
            y, r = envir.feedback(i=i, k=kk)
            self.store_info(i=i, x=x, y=y, t=t, r=r)
            self.thompson.update(i,choice,y,r)
           # self.update(t)

        self.result_print(envir,i,10)



if __name__ == '__main__':
    t = Thompson()
    sum = 0
    for i in range(100):
        c = t.select()
        sum =sum + c
        t.update(c,1)
    print(sum)