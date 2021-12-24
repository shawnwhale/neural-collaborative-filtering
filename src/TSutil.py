import numpy as np
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt

"""bandits算法用的"""
def init_negasample(user,user_choice_pool,user_train_positive_pool):
    result_pool = {}
    for group in range(18):

        key = str(user) + ":" + str(group)

        if len(user_choice_pool[key]) > 50:
            one_result = random.sample(user_choice_pool[key], 50)
        else:
            one_result = random.sample(user_choice_pool[key], len(user_choice_pool[key]))

        positive_set = user_train_positive_pool[key]
        one_result = set(one_result)
        for temp in positive_set:
            if temp not in one_result:
                one_result.add(temp)
        result_pool[key] = list(one_result)
    return result_pool

if __name__ == '__main__':
    items_popu_dic = np.load("./items_popu_dic.npy",allow_pickle=True).tolist()
    list = []
    ind=[]
    index = 1
    for item in items_popu_dic:
        list.append(items_popu_dic[item])
        ind.append(index)
        index = index+1
    list.sort(reverse=True)
    plt.xlabel("index")

    plt.ylabel("popularity")
    plt.title("Items’ popularity in MovieLens-1M")
    plt.plot(ind, list, 'ko',markersize=2)

    plt.show()