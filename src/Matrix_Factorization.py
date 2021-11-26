#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Feiniu

import numpy as np
import pandas as pd
from src.metrics import MetronAtK
import torch

class Matrix_Factorization(object):

    def __init__(self, K=10, alpha=0.01, beta=0.02, epoch=1000, regularization=True, random_state=0):

        self.R = None
        self.K = K
        self.P = None
        self.Q = None
        self.r_index = None
        self.r = None
        self.length = None
        self.alpha = alpha
        self.beta = beta
        self.epoch = epoch
        self.regularization = regularization
        self.random_state = random_state


    def fit(self, R):

        np.random.seed(self.random_state)
        self.R = R.values
        M, N = self.R.shape
        self.P = np.random.rand(M, self.K)
        self.Q = np.random.rand(N, self.K)

        self.r_index = self.R.nonzero()
        self.r = self.R[self.r_index[0], self.r_index[1]]
        self.length = len(self.r)


    def _comp_descent(self, index):

        r_i = self.r_index[0][index]
        r_j = self.r_index[1][index]

        p_i = self.P[r_i]
        q_j = self.Q[r_j]

        r_ij_hat = p_i.dot(q_j)
        e_ij = self.R[r_i, r_j] - r_ij_hat

        if self.regularization == True:
            descent_p_i = -2 * e_ij * q_j + self.beta * p_i
            descent_q_j = -2 * e_ij * p_i + self.beta * q_j
        else:
            descent_p_i = -2 * e_ij * q_j
            descent_q_j = -2 * e_ij * p_i

        return r_i, r_j, p_i, q_j, descent_p_i, descent_q_j


    def _update(self, p_i, q_j, descent_p_i, descent_q_j):

        p_i_new = p_i - self.alpha * descent_p_i
        q_j_new = q_j - self.alpha * descent_q_j

        return p_i_new, q_j_new


    def _estimate_r_hat(self):

        r_hat = self.P.dot(self.Q.T)[self.r_index[0], self.r_index[1]]

        return r_hat


    def start(self):

        epoch_num = 1
        while epoch_num <= self.epoch:
            for index in range(0, self.length):

                r_i, r_j, p_i, q_j, descent_p_i, descent_q_j = self._comp_descent(index)
                p_i_new, q_j_new = self._update(p_i, q_j, descent_p_i, descent_q_j)

                self.P[r_i] = p_i_new
                self.Q[r_j] = q_j_new

            r_hat = self._estimate_r_hat()
            e = r_hat - self.r
            error = e.dot(e)
            print('The error is %s=================Epoch:%s' %(error, epoch_num))
            epoch_num += 1

        R_hat = self.P.dot(self.Q.T)
        return R_hat

    def evaluate(self,R_hat,evaluate_data):
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        test_scores = []
        negative_scores = []
        for i in range(test_users.shape[0]):
            test_score = R_hat[test_users[i].item(), test_items[i].item()]
            test_scores.append(test_score)
        test_scores = torch.from_numpy(np.array(test_scores))
        for i in range(negative_users.shape[0]):
            negative_score = R_hat[negative_users[i].item(),negative_items[i].item()]
            negative_scores.append(negative_score)
        negative_scores = torch.from_numpy(np.array(negative_scores))

        metron = MetronAtK(top_k=20)

        metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        entropy = metron.cal_entropy()
        kendall = metron.cal_kendall()
        hit_ratio, ndcg = metron.cal_hit_ratio(), metron.cal_ndcg()
        ils = metron.cal_ils()

        return hit_ratio, ndcg, ils, kendall,entropy

if __name__ == '__main__':

    user_rating = pd.read_csv('../data/Moivelens/ml-latest-small/user-rating.csv', index_col=0)

    aa = Matrix_Factorization(K = 5)
    aa.fit(user_rating)
    aa.start()