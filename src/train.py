import pandas as pd
import numpy as np
import time
import gc
from src.gmf import GMFEngine
from src.mlp import MLPEngine
from src.neumf import NeuMFEngine
from src.data import SampleGenerator



gmf_config = {'alias': 'gmf_factor16_from120_books-implict',
              'num_epoch': 120,
              'batch_size': 4096,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 5432,
              'num_items': 9701,
              'latent_dim': 16,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format(
              'gmf_factor16_from120_books-implict_Epoch4_HR0.5462_NDCG0.3018 ILS0.0000  Kendall0.0000.model'),

              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f} ILS{:.4f}  Kendall{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_060000001',
              'num_epoch': 60,
              'batch_size':4096,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 5432,
              'num_items': 9701,
              'latent_dim': 16,
              'num_negative': 4,
              'layers': [32,128,64,32,16],
              # 'layers': [16,64,32,16,8], # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format(
              'gmf_factor16_from120_books-implict_Epoch14_HR0.7211_NDCG0.3440 ILS0.8044  Kendall0.4060.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f} ILS{:.4f} Kendall{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor16_books',
                'num_epoch': 100,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 5432,
                'num_items': 9701,
                'latent_dim_mf': 16,
                'latent_dim_mlp': 16,
                'num_negative': 4,
                'layers': [32,128,64,32,16],  # layers[0] is the concat of latent user vector & latent item vector
                # 'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'use_cuda': True,
                'device_id': 0,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor16_from120_books-implict_Epoch14_HR0.7211_NDCG0.3440 ILS0.8044  Kendall0.4060.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_bz256_166432168_pretrain_reg_060000001_Epoch7_HR0.7268_NDCG0.3506 ILS0.8027 Kendall0.4059.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f} ILS{:.4f} Kendall{:.4f}.model'
                }

#
# # Load Data
# ml1m_dir = 'data/ml-1m/ratings.dat'
# ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
# # Reindex
# user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
# user_id['userId'] = np.arange(len(user_id))
# ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
# item_id = ml1m_rating[['mid']].drop_duplicates()
# item_id['itemId'] = np.arange(len(item_id))
# ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
# ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
# print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
# print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
#
#
#
# # DataLoader for training
# sample_generator = SampleGenerator(ratings=ml1m_rating)
# evaluate_data = sample_generator.evaluate_data
# # Specify the exact model
# config = gmf_config
# engine = GMFEngine(config)
# # config = mlp_config
# # engine = MLPEngine(config)
# # config = neumf_config
# # engine = NeuMFEngine(config)
#
# f=open("{}HRandNDCG-{}.txt".format(int(time.time()),config.get('alias')),"w")
#
# for epoch in range(config['num_epoch']):
#     print('Epoch {} starts !'.format(epoch))
#     print('-' * 80)
#     train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
#     engine.train_an_epoch(train_loader, epoch_id=epoch)
#     hit_ratio, ndcg, ils, kendall= engine.evaluate(evaluate_data, epoch_id=epoch)
#     engine.save(config['alias'], epoch, hit_ratio, ndcg, ils, kendall)
#
#     f.write("hit_ratio:ndcg:ils:kendall:{}:{}:{}:{}\n".format(hit_ratio, ndcg, ils, kendall))
#     gc.collect()
# f.close()
def load_books():
    # Load Data
    book_dir = 'data/to_read_changed.csv'
    book_rating = pd.read_csv(book_dir, sep=',', header=0, names=['uid', 'mid', 'rating', 'timestamp'],
                              engine='python')
    # Reindex
    user_id = book_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    book_rating = pd.merge(book_rating, user_id, on=['uid'], how='left')
    item_id = book_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    book_rating = pd.merge(book_rating, item_id, on=['mid'], how='left')
    book_rating = book_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(book_rating.userId.min(), book_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(book_rating.itemId.min(), book_rating.itemId.max()))
    return book_rating

if __name__ == '__main__':
    # # Load Data
    # ml1m_dir = 'data/ml-1m/ratings.dat'
    # ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    # # Reindex
    # user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    # user_id['userId'] = np.arange(len(user_id))
    # ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    # item_id = ml1m_rating[['mid']].drop_duplicates()
    # item_id['itemId'] = np.arange(len(item_id))
    # ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    # ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    # print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    # print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))

    book_rating = load_books()

    # 处理goodbooks的流行度
    # items_book_popu_dic = {}
    # for index, row in book_rating.iterrows():
    #     itemId = int(row["itemId"])
    #     if itemId in items_book_popu_dic:
    #         popu = items_book_popu_dic[itemId]
    #         items_book_popu_dic[itemId] = popu + 1
    #     else:
    #         items_book_popu_dic[itemId] = 1
    # np.save("./items_book_popu_dic.npy", items_book_popu_dic)

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=book_rating)
    evaluate_data = sample_generator.evaluate_data
    # Specify the exact model
    # config = gmf_config
    # engine = GMFEngine(config)
    # config = mlp_config
    # engine = MLPEngine(config)
    config = neumf_config
    engine = NeuMFEngine(config)

    f=open("{}HRandNDCG-{}.txt".format(int(time.time()),config.get('alias')),"w")

    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg, ils, kendall,entropy= engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg, ils, kendall,entropy)

        f.write("hit_ratio:ndcg:ils:kendall:entropy:{}:{}:{}:{}:{}\n".format(hit_ratio, ndcg, ils, kendall,entropy))
        gc.collect()
    f.close()