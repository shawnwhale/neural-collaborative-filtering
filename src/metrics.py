import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import time

class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
        # the golden set
        test = pd.DataFrame({'user': test_users,
                             'test_item': test_items,
                             'test_score': test_scores})
        # the full set
        full = pd.DataFrame({'user': neg_users + test_users,
                            'item': neg_items + test_items,
                            'score': neg_scores + test_scores})
        full = pd.merge(full, test, on=['user'], how='left')
        # rank the items according to the scores for each user
        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full

    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]  # golden items hit in the top_K items

        return len(test_in_top_k) * 1.0 / full['user'].nunique()

    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        test_in_top_k =top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x)) # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()


    def cal_entropy(self, user=6040):
        #熵多样性计算
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        item_entropy_dic = {}
        sumrow = 0
        for index, row in top_k.iterrows():
            item = int(row['item'])
            sumrow = sumrow + 1
            if item in item_entropy_dic:
                item_entropy_dic[item] = item_entropy_dic[item]+1
            else:
                item_entropy_dic[item] = 1

        # if user > 1200:
        #     name = str(user)+str(time.time())
        #     np.save("./entropy_dic"+name +".npy", item_entropy_dic)

        entropy = 0
        for i in range(3706):
            if i in item_entropy_dic:
                temp = item_entropy_dic[i]/sumrow
                entropy = entropy + temp * math.log(temp)
            else:
                continue
        return -entropy

    def cal_ils(self):
        dim_np= np.load('./item_dim.npy')
        num_k=self._top_k
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank']<=top_k]
        user_itemsS = top_k.groupby('user')
        ils_sum = 0
        for user_items in user_itemsS:
            items_seq = user_items[1]['item']
            items_vec = dim_np[items_seq, :]
            similarity_matrix = cosine_similarity(items_vec)
            similarity_matrix_df = pd.DataFrame(similarity_matrix)
            similarity_matrix_df = (similarity_matrix_df + 1) * 0.5    #对余弦相似度归一化处理
            L = np.tril(similarity_matrix_df, -1)
            one_ils= 2*L.sum()/(num_k*(num_k-1))
            ils_sum=ils_sum+one_ils
        ils = ils_sum / full['user'].nunique()
        return (1.0-ils) * 2.0   #return ild

    def cal_kendall(self):
        kendall_np = np.load('./user_dim.npy')
        dim_np = np.load('./item_dim.npy')

        num_k = self._top_k
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        user_itemsS = top_k.groupby('user')
        kendall_sum = 0
        up_num = 0 #符合要求的用户数
     #   temp_up_num = 0
        for user_items in user_itemsS:
            items_seq = user_items[1]['item']
            user_id = int(user_items[0])
            items_vec = dim_np[items_seq, :]   #获得推荐项目的向量
            items_vec_sum = np.zeros(shape=(1,18))
            for vec in items_vec:
                items_vec_sum += vec
            temp = kendall_np[user_id, :]
            # for t in temp:
            #     temp_up_num  =temp_up_num + int(t)
    #        if temp_up_num > 200:
            up_num = up_num +1
            data = np.vstack((items_vec_sum, temp))
            data = pd.DataFrame(data)
            data = pd.DataFrame(data.values.T)
            data = data.corr('kendall')
            kendall = data.iloc[0, 1]
            kendall_sum += kendall
      #      temp_up_num = 0
        return kendall_sum/up_num

    def cal_popular(self):
        items_popu_dic = np.load("./items_popu_dic.npy", allow_pickle=True).tolist()
        num_k = self._top_k
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        user_itemsS = top_k.groupby('user')
        popu_per = 0

        for user_items in user_itemsS:
            items_seq = user_items[1]['item']
            for seq in items_seq:
                popu = items_popu_dic[seq]
                popu_per = popu_per +popu

        return popu_per / (full['user'].nunique() * num_k)