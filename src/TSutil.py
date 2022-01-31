import numpy as np
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
import math

GROUP_NUM = 26

maxpopu = 3428
# perpopu = 269.6691
# stdpopu = 383.9960

perpopu = 29.4552
stdpopu = 55.5683
midpopu = 242 #中位数
"""bandits算法用的"""
def init_negasample(user,user_choice_pool,user_train_positive_pool):
    result_pool = {}
    popu_sum = 0.0
    popu_av = 0.0
    popu_ind = 0
    items_popu_dic = np.load("./items_popu_dic.npy",allow_pickle=True).tolist()
    for group in range(GROUP_NUM):

        key = str(user) + ":" + str(group)

        if len(user_choice_pool[key]) > 50:
            one_result = random.sample(user_choice_pool[key], 50)
        else:
            one_result = random.sample(user_choice_pool[key], len(user_choice_pool[key]))

        positive_set = user_train_positive_pool[key]
        one_result = set(one_result)
        for temp in positive_set:

            popu = items_popu_dic[temp]
            popu_sum = popu_sum + popu
            popu_ind = popu_ind + 1

            if temp not in one_result:
                one_result.add(temp)
        result_pool[key] = list(one_result)
    if popu_ind > 0:
        popu_av = popu_sum / popu_ind   #user 93 = null?
    else:
        popu_av = perpopu
    # print(popu_av)
    return result_pool,popu_av




if __name__ == '__main__':

    # books

    # items_book_popu_dic = np.load("./items_book_popu_dic.npy",allow_pickle=True).tolist()
    # list = []
    # ind=[]
    # index = 1
    # for item in items_book_popu_dic:
    #     list.append(items_book_popu_dic[item])
    #     ind.append(index)
    #     index = index+1
    # list.sort(reverse=True)
    # plt.xlabel("index")
    #
    # plt.ylabel("popularity")
    # plt.title("Items’ popularity in goodbooks-10k")
    # plt.plot(ind, list, 'ko',markersize=1)
    # plt.show()
    #
    # sum = 0
    # for key in list:
    #     sum = sum + int(key)
    # print(sum/len(list))
    #
    # items_book_popu_dic = np.load("./items_book_popu_dic.npy", allow_pickle=True).tolist()
    # # items_book_popu_tran_dic = {}
    # list_temp = []
    # for item in items_book_popu_dic:
    #     list_temp.append(items_book_popu_dic[item])
    # std = np.std(list_temp)
    # print(std)   # std：55.5683   max:880 per:29.4552

    items_popu_dic = np.load("./items_book_popu_dic.npy", allow_pickle=True).tolist()
    items_book_popu_tran_dic = {}
    for item in items_popu_dic:
        stdv = abs(items_popu_dic[item] - perpopu) / stdpopu + 1
        tran = math.log(1 + 1 / stdv)
        tran = 1 + tran * 0.5
        # tran = 1.0 / items_popu_dic[item]   #Ziwei_WSDM_2021
        items_book_popu_tran_dic[item] = tran
        print(tran)
    np.save("./items_book_popu_tran_dic.npy", items_book_popu_tran_dic)

    # movieLens

    # items_popu_dic = np.load("./items_popu_dic.npy",allow_pickle=True).tolist()
    # list = []
    # ind=[]
    # index = 1
    # for item in items_popu_dic:
    #     list.append(items_popu_dic[item])
    #     ind.append(index)
    #     index = index+1
    # list.sort(reverse=True)
    # plt.xlabel("index")
    #
    # plt.ylabel("popularity")
    # plt.title("Items’ popularity in MovieLens-1M")
    # plt.plot(ind, list, 'ko',markersize=2)
    #
    # plt.show()

    # items_popu_dic = np.load("./items_popu_dic.npy", allow_pickle=True).tolist()
    # items_popu_tran_dic = {}
    # list_temp = []
    # for item in items_popu_dic:
    #     list_temp.append(items_popu_dic[item])
    # std = np.std(list_temp)
    # print(std)

    # items_popu_dic = np.load("./items_popu_dic.npy", allow_pickle=True).tolist()
    # items_popu_tran_dic = {}
    # for item in items_popu_dic:
    #     # stdv = abs(items_popu_dic[item] - perpopu) / stdpopu + 1
    #     # tran = math.log(1 + 1 / stdv)
    #     # tran = 1 + tran * 0.5
    #     tran = 1.0 / items_popu_dic[item]   #Ziwei_WSDM_2021
    #     items_popu_tran_dic[item] = tran
    #     print(tran)
    # np.save("./items_popu_tran_dic.npy", items_popu_tran_dic)

    # items_popu_dic = np.load("./items_popu_dic.npy", allow_pickle=True).tolist()
    # location_list = []
    # for item in items_popu_dic:
    #     location_list.append(items_popu_dic[item])
    # sorted(location_list)
    # len_location = len(location_list)
    # print(len_location * 0.8)
    # print(len_location * 0.5)
    # print(location_list[3704 - 2964])
    # print(location_list[1852])