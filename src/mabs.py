import numpy as np
import random
import gc
import torch
from sklearn.metrics.pairwise import cosine_similarity
from src.metrics import MetronAtK
import traceback

USER_BEGIN = 0
USER_NUM = 101
VEC_DIM = 16
items_popu_tran_dic = np.load("./items_popu_tran_dic.npy",allow_pickle=True).tolist()
train_np = np.load("./UCBtrain.npy")
dim_np = np.load('./item_dim.npy')
"""bandits算法用的"""
class Thompson:
    def __init__(self):
        self.para_u = np.zeros(shape=(6040, 18))
        # self.para_u = np.full(shape=(6040, 18),fill_value = 0.1)
        self.para_k = np.zeros(shape=(6040, 18))

    def select(self,u , repeat = 0 ):
        temp = np.random.normal(self.para_u[u, :], 1 / (self.para_k[u, :] + 1))
        argsort = np.argsort(-temp)
        choice = np.argmax(temp)
        if repeat == 0:
            return choice
        else:
            choices = []
            choices.append(choice)
            for re in range(1,repeat):
                temp = np.random.normal(self.para_u[u, :], 1 / (self.para_k[u, :] + 1))
                choice = np.argmax(temp)
                choices.append(choice)
            return choices

    def update(self,u, c ,y,item_earnings = 1):
        # if y == 1:
        self.para_u[u,c] =  (self.para_u[u,c]* self.para_k[u,c] + y)/ (self.para_k[u,c]+2.0)

        self.para_k[u,c] = self.para_k[u,c]+1
        return

class Evaluate_util:
    def __init__(self):
        self.test_users = []
        self.test_items = []
        self.negative_users = []
        self.negative_items = []
        self.test_scores = []
        self.negative_scores = []

class Explorer:
    def __init__(self,item_embedding,user_embedding, negatives, train_ratings, test_ratings, engine):
        self.thompson = Thompson()
        self.nega_item = {}
        self.posi_item = {}
        self.item_embedding = item_embedding
        self.user_embedding = []  #没使用
        self.negatives = negatives
        self.train_ratings = train_ratings
        self.test_ratings = test_ratings
        self.engine = engine
        # 记录某用户在某一组可以选的candidate,记录某用户在某一组可以选的正样本
        self.user_choice_pool ={}
        self.user_train_positive_pool = {}
        self.evaluate_data = Evaluate_util()


    def add_choice_pool(self,userbegin,userend):
        user_choice_pool = {}
        user_train_positive_pool ={}   #清除之前的
        item_dim = np.load('./item_dim.npy')
        # for index, line in self.test_ratings.iterrows():
        #     user = int(line['userId'])
        #     item = int(line['itemId'])
        #     test_item[user] = item
        # np.save("./test_item.npy",test_item)
        test_item = np.load("./test_item.npy",allow_pickle=True).tolist()
        # for index, line in self.train_ratings.iterrows():
        #     user = int(line['userId'])
        #     item = int(line['itemId'])
        #     if user in train_positive:
        #         train_positive[user].append(item)
        #     else:
        #         train_positive[user] = []
        #         train_positive[user].append(item)
        # np.save("./train_positive.npy", train_positive)
        train_positive = np.load("./train_positive.npy",allow_pickle=True).tolist()
        for group in range(18):
            candidate = item_dim[:, group]
            candidate = np.nonzero(candidate)[0]
            index = []
            for j in candidate:
                index.append(j)
            index = set(index)
            for user in range(userbegin,userend):
                # 每个user有不同的item池
                key = str(user) + ":" + str(group)
                nega_sam = set(self.negatives.ix[user, 'negative_samples'])
                index.discard(test_item[user])   #去除测试集正样本
                for j in nega_sam:
                    index.discard(j)
                user_choice_pool[key] = index

                positive_pool = train_positive[user].copy()
                positive_pool = set(positive_pool)
                temp_dele = []
                for j in positive_pool:
                    if j not in index:
                        temp_dele.append(j)
                for j in temp_dele:
                    positive_pool.discard(j)
                user_train_positive_pool[key] = positive_pool
        return user_choice_pool, user_train_positive_pool

    def rewards(self, user, maxarg,rating_np = train_np):
        reward = rating_np[user,maxarg]
        if reward == 1:
            return 1
        else:
            return 0

    def get_echoes(self,pool):
        group_num = {}  # 记录摇到了多少个
        for group in pool:
            if group in group_num:
                group_num[group] = group_num[group] + 1
            else:
                group_num[group] = 1
        return  group_num

    def calculate(self,user,posi_vec,k=10):
        choice_pool = self.thompson.select(user, repeat=k)
        result,flag = self.test_recommend(user,choice_pool,posi_vec, k=k)
        return  flag


    def gmf_recommend(self,user, k=10):
        # 对比实验，gmf推荐前k个准确率
        ans_item = -1
        result =set()
        test_list = list(self.negatives.ix[user, 'negative_samples'])
        for index, line in self.test_ratings.iterrows():
            userid = int(line['userId'])
            itemid = int(line['itemId'])
            if userid == user:
                test_list.append(itemid)
                ans_item = itemid
                break

        self.engine.model.eval()
        one_result = np.array(list(test_list))
        result_ten = torch.LongTensor(one_result).cuda()
        user_seq = np.full(len(one_result), user, dtype=int)
        user_seq = torch.LongTensor(user_seq).cuda()
        pre_scores = self.engine.model(user_seq, result_ten)
        temp_np = pre_scores.cpu().detach().numpy()
        sorted_scores = np.argsort(- np.transpose(temp_np))
        for i in range(10):
            item_num = one_result[sorted_scores[0,i]]
            result.add(item_num)
        if ans_item in result:
            flag = 1
        else:
            flag = 0
        return result, flag

    def test_recommend(self, user, choice_pool,posi_vec, k=10):
        result = []
        result_score = []
        result_set = set()
        ans_item = -1
        test_set = set(self.negatives.ix[user, 'negative_samples'])
        for index, line in self.test_ratings.iterrows():
            userid = int(line['userId'])
            itemid = int(line['itemId'])
            if userid == user:
                test_set.add(itemid)
                ans_item = itemid
                break

        # 分配test_set到18个group
        item_group = {}
        for item in test_set:
            group_non = np.nonzero(dim_np[item])
            for seq in group_non[0]:
                if seq in item_group:
                    items = item_group[seq]
                    items.append(item)
                else:
                    item_group[seq] = []
                    item_group[seq].append(item)
        group_non = np.nonzero(dim_np[ans_item])
        for seq in group_non[0]:
            if seq in item_group:
                items = item_group[seq]
                items.append(item)
            else:
                item_group[seq] = []
                item_group[seq].append(item)

        self.engine.model.eval()
     #   result.append(ans_item)
        result, result_set,result_score = self.deal_choice(user, choice_pool, result, result_set, result_score, item_group, posi_vec)
        secu = 0
        while len(result) <  2 * k:
            secu = secu + 1
            if secu > 5:
                # 防止卡死在某一个group
                next_pool = []
                num_pool = random.randint(0, 17)
                next_pool.append(num_pool)
            else:
                next_pool = self.thompson.select(user, repeat=1)
            result, result_set,result_score = self.deal_choice(user, next_pool, result, result_set, result_score, item_group, posi_vec)
        temp_np = np.array(result_score)
        sorted_scores = np.argsort(- temp_np)
        ans_table = []
        for num in range(k):
            seq = int(sorted_scores[num])
            ans_table.append(result[seq])
        ans_set = set(ans_table)
        ans_loc = -1
        if ans_item in ans_set:
            flag = 1
            for ite in range(len(result)):
                if ans_item == result[ite]:
                    ans_loc = ite
                    break
        else:
            flag = 0

        #放进计算包里面
        self.evaluate_data.test_users.append(user)
        self.evaluate_data.test_items.append(ans_item)
        if flag == 1:
            self.evaluate_data.test_scores.append(result_score[ans_loc])
            for ite in range(len(result)):
                if ite == ans_loc:
                    continue
                else:
                    self.evaluate_data.negative_users.append(user)
                    self.evaluate_data.negative_items.append(result[ite])
                    self.evaluate_data.negative_scores.append(result_score[ite])
        else:
            self.evaluate_data.test_scores.append(-1)
            for ite in range(len(result)):
                self.evaluate_data.negative_users.append(user)
                self.evaluate_data.negative_items.append(result[ite])
                self.evaluate_data.negative_scores.append(result_score[ite])

        return ans_table, flag

    def deal_choice(self,user,choice_pool,result,result_set,result_score, item_group, posi_vec):
        item_embedding = self.item_embedding
        group_num = self.get_echoes(choice_pool) # 记录摇到了多少个
        choice_pool = set(choice_pool)
        for group in choice_pool:
            num = group_num[group] * 2

            if group not in item_group:
                continue
            one_result = item_group[group]
            result_ten = torch.LongTensor(one_result).cuda()
            user_seq = np.full(len(one_result), user, dtype=int)
            user_seq = torch.LongTensor(user_seq).cuda()
            pre_scores = self.engine.model(user_seq, result_ten)
            temp_np = pre_scores.cpu().detach().numpy()

            m = 0  # 序号
            for temp in one_result:
                arfa = items_popu_tran_dic[temp]  # 阿尔法是流行度调整参数
                temp_np[m, :] = temp_np[m, :] * arfa
                m=m+1

            similarity_part = []
            for temp in one_result:
                items_vec = np.zeros(shape=(2, VEC_DIM))
                items_vec[0, :] = posi_vec
                items_vec[1, :] = item_embedding[temp, :]
                similarity_posi = cosine_similarity(items_vec)
                similarity_part.append(similarity_posi[0,1])
            similarity_part = np.array(similarity_part)

            # temp_np = np.transpose(temp_np)
            temp_np = np.transpose(temp_np) -  0.1 * similarity_part
            sorted_scores = np.argsort(- temp_np)
            range_jud = sorted_scores.shape[1] - num    #大于0，说明可选内容多。反之，权重大。
            if range_jud >= 0:
                range_num = num
            else:
                range_num = sorted_scores.shape[1]
            Ie = 0
            su = 0
            while su < range_num:
                if Ie >= sorted_scores.shape[1]:
                    break
                seq_scores = int(sorted_scores[0,Ie])
                res_item = int(one_result[seq_scores])
                if res_item in result_set:
                    Ie = Ie + 1
                else:
                    result.append(res_item)
                    result_score.append(temp_np[0,seq_scores])
                    result_set.add(res_item)
                    Ie = Ie + 1
                    su = su + 1
        return result, result_set, result_score


    def generate_recommend(self, user, choice_pool,posi_vec,nega_set):
        result = []
        result_scores = []
        result_earnings = []
        result_set = set()
        positive_temp = -1
        item_embedding = self.item_embedding
        self.engine.model.eval()
        group_num = self.get_echoes(choice_pool)  # 记录摇到了多少个
        choice_pool = set(choice_pool)
        result_choice_pool = []
        for group in choice_pool:

            num = group_num[group]

            key = str(user) + ":" + str(group)
            if positive_temp == -1 and len(self.user_train_positive_pool[key]) > 0 :
                positive_temp = random.sample(self.user_train_positive_pool[key], 1)
                for item in positive_temp:
                    positive_item = int(item)

                no_tensor = []
                no_tensor.append(positive_item)
                result_ten = torch.LongTensor(no_tensor).cuda()
                no_tensor = []
                no_tensor.append(user)
                user_seq = torch.LongTensor(no_tensor).cuda()
                pre_scores = self.engine.model(user_seq, result_ten)
                items_vec = np.zeros(shape=(2, VEC_DIM))
                items_vec[0, :] = posi_vec
                items_vec[1, :] = item_embedding[positive_item, :]
                similarity_posi = cosine_similarity(items_vec)
                pre_scores = pre_scores.cpu().detach().numpy()
                arfa = items_popu_tran_dic[positive_item]    #阿尔法是流行度调整参数
                positive_scores = pre_scores[0,0] * arfa - similarity_posi[0, 1]
                result.append(positive_item)
                result_set.add(positive_item)
                result_earnings.append(pre_scores[0,0])
                result_scores.append(positive_scores)
                result_choice_pool.append(group)
            else:
                if len(self.user_choice_pool[key]) > 100:
                    one_result = random.sample(self.user_choice_pool[key], 100)
                else:
                    one_result = random.sample(self.user_choice_pool[key], len(self.user_choice_pool[key]))
                if len(one_result) > 0:
                    result_ten = torch.LongTensor(one_result).cuda()
                    user_seq = np.full(len(one_result), user, dtype=int)
                    user_seq = torch.LongTensor(user_seq).cuda()
                    pre_scores = self.engine.model(user_seq, result_ten)
                    pre_np = pre_scores.cpu().detach().numpy()

                    similarity_part = []
                    m = 0   #序号
                    for temp in one_result:
                        arfa = items_popu_tran_dic[temp]  # 阿尔法是流行度调整参数
                        pre_np[m, :] = pre_np[m, :] * arfa
                        items_vec = np.zeros(shape=(2, VEC_DIM))
                        items_vec[0, :] = posi_vec
                        items_vec[1, :] = item_embedding[temp, :]
                        similarity_posi = cosine_similarity(items_vec)
                        similarity_part.append(similarity_posi[0,1])
                        m = m + 1
                    similarity_part = np.array(similarity_part)

                    scores = pre_np.T - similarity_part

                    sorted_scores = np.argsort(- scores)
                    range_jud = sorted_scores.shape[1] - num  # 大于0，说明可选内容多。反之，权重大。
                    if range_jud >= 0:
                        range_num = num
                    else:
                        range_num = sorted_scores.shape[1]
                    Ie = 0
                    su = 0
                    while su < range_num:
                        if Ie >= sorted_scores.shape[1]:
                            break
                        seq_scores = int(sorted_scores[0, Ie])
                        res_item = int(one_result[seq_scores])
                        if res_item in result_set:
                            Ie = Ie + 1
                        elif res_item in nega_set:
                            Ie = Ie + 1
                        else:
                            result.append(res_item)
                            result_earnings.append(pre_np.T[0, seq_scores])
                            result_scores.append(scores[0, seq_scores])
                            result_choice_pool.append(group)
                            result_set.add(res_item)
                            Ie = Ie + 1
                            su = su + 1
        return  result, result_scores,result_earnings,result_choice_pool   #返回item编号，及计算得分

    def run(self,T,k=10):
        succ = 0

        item_embedding = self.item_embedding
        self.engine.model.eval()
        self.user_choice_pool,self.user_train_positive_pool = self.add_choice_pool(USER_BEGIN,USER_BEGIN + 100)
        create_num = USER_BEGIN + 100
        for user in range(USER_BEGIN , USER_NUM):
            succ_part = 0
            posi_vec = np.zeros(shape=(1, VEC_DIM))
            posi_num = 0
            nega_set = set()

            if user >= create_num:
                self.user_choice_pool, self.user_train_positive_pool = self.add_choice_pool(create_num,create_num + 100)
                create_num = create_num + 100
            for turn in range(T):
                try:
                    choice_pool=[]
                    #摇repeat次进行一次学习
                    choice_pool = self.thompson.select(user,repeat= 10)
                    result,result_scores,result_earnings,result_choice_pool = self.generate_recommend(user, choice_pool, posi_vec, nega_set)
                    item_temp = np.array(result_scores)
                    resarg = np.argmax(item_temp)   #ValueError: attempt to get argmax of an empty sequence
                    item_num = result[resarg]
                    item_earnings = result_earnings[resarg]
                    reward = self.rewards(user, item_num)
                    group_num = result_choice_pool[resarg]
                    new_vec = item_embedding[item_num, :]
                    if reward == 1:
                        succ_part = succ_part+1
                        posi_vec = (posi_vec + posi_num)/2
                        posi_num = posi_num + 1
                    else:
                        # nega_set.add(item_num)

                        posi_vec = (posi_vec * posi_num + new_vec) / (posi_num + 1.0)
                        posi_num = posi_num + 1
                    self.thompson.update(user, group_num ,reward,item_earnings)
                except:
                    traceback.print_exc()
                    continue
            reward = self.calculate(user,posi_vec,k=k)
     #       result, reward=self.gmf_recommend(user)
            succ = succ +reward
            print("完成user计算：" + str(user))
            hit = succ / (float(user) - float(USER_BEGIN) +1.0 )
            print(hit)
            gc.collect()
            if user > 0 and user % 100 == 0:
                hit_ratio, ndcg, ils, kendall, entropy=self.evaluate(self.evaluate_data,k=k)
                print(str(hit_ratio) +"!"+ str(ndcg )+"!"+ str(ils) +"!"+str(kendall)+ "!" + str(entropy))
        f = open("newK=20.txt", "w")
        hit_ratio, ndcg, ils, kendall, entropy= self.evaluate(self.evaluate_data,k=k)
        f.write("hit_ratio:ndcg:ils:kendall:entropy:{}:{}:{}:{}:{}\n".format(hit_ratio, ndcg, ils, kendall, entropy))
        f.close()


    def evaluate(self,evaluate_data,k=10):

        test_users = evaluate_data.test_users
        test_items = evaluate_data.test_items
        negative_users = evaluate_data.negative_users
        negative_items = evaluate_data.negative_items
        test_scores = evaluate_data.test_scores
        negative_scores = evaluate_data.negative_scores

        metron = MetronAtK(top_k=k)

        metron.subjects = [test_users,
                                 test_items,
                                 test_scores,
                                 negative_users,
                                 negative_items,
                                 negative_scores]
        kendall = metron.cal_kendall()
        entropy= metron.cal_entropy(len(test_users))
        hit_ratio, ndcg = metron.cal_hit_ratio(), metron.cal_ndcg()
        ils = metron.cal_ils()

        return hit_ratio, ndcg, ils, kendall,entropy