import pandas as pd
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

if __name__ == '__main__':
       # Load Data
       ml1m_dir = 'data/ml-1m/movies.csv'
       ml1m_movies = pd.read_csv(ml1m_dir,header=0, names=['oid', 'id', 'title', 'genre'])
       genre_dic = {}
       genre_dicT = {}
       genre_num=0
       genre = ml1m_movies['genre']
       for i in range(genre.size):

              genres = genre[i].split('|')
              for one in genres:
                     if genre_dic.get(one):
                            pass
                     else:
                            genre_num+=1
                            genre_dic[one] = genre_num
                            genre_dicT[genre_num] = one

       # 存np
       # genre_np=np.array([],dtype= str )
       # for j in range(genre_num):
       #        genre_np =np.append(genre_np,genre_dicT[j+1])
       # np.save('./output_genre',genre_np)
       # np_out = np.load('./output_genre.npy')
       # print(np_out)

       #生成多样性矩阵
       np_out = np.load('./output_genre.npy')
       reitem_dic = np.load("./reitem_dic.npy", allow_pickle=True).tolist()

       dim_np =np.zeros(shape=(3706,18))
       for i in range(ml1m_movies.shape[0]):
              movie = ml1m_movies.loc[i,:]
              num = movie['oid']
              num = reitem_dic[num]  #old to key
              genre = movie['genre']
              genres = genre.split('|')
              for one in genres:
                     index = genre_dic[one]
                     dim_np[num,index-1] = 1
       np.save('./item_dim', dim_np)