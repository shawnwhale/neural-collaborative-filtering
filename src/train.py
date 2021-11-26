import pandas as pd
import numpy as np
import time
import gc
from src.gmf import GMFEngine
from src.mlp import MLPEngine
from src.neumf import NeuMFEngine
from src.data import SampleGenerator
from src.ENVIRONMENT import Environment


gmf_config = {'alias': 'gmf_factor16neg4k20-implict',
              'num_epoch': 60,
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
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 16,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format(
              'gmf_factor16neg4k10-implict_Epoch59_HR0.6712_NDCG0.3968 ILS0.6346  Kendall0.5500.model'),

              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f} ILS{:.4f}  Kendall{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_060000001',
              'num_epoch': 60,
              'batch_size':4096,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 16,
              'num_negative': 4,
              'layers': [32,128,64,32,16],
              # 'layers': [16,64,32,16,8], # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format(
              'mlp_factor8neg4_bz256_166432168_pretrain_reg_060000001_Epoch16_HR0.6806_NDCG0.4058 ILS0.6376 Kendall0.5478.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f} ILS{:.4f} Kendall{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 100,
                'batch_size': 2048,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                # 'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'use_cuda': True,
                'device_id': 0,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4-implict_Epoch59_HR0.6407_NDCG0.3689 ILS0.6336.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_bz256_166432168_pretrain_reg_060000001_Epoch59_HR0.6541_NDCG0.3798 ILS0.6334.model'),
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

if __name__ == '__main__':
    # Load Data
    ml1m_dir = 'data/ml-1m/ratings.dat'
    ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    # Reindex
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))



    # DataLoader for training
    sample_generator = SampleGenerator(ratings=ml1m_rating)
    evaluate_data = sample_generator.evaluate_data
    # Specify the exact model
    # config = gmf_config
    # engine = GMFEngine(config)
    config = mlp_config
    engine = MLPEngine(config)
    # config = neumf_config
    # engine = NeuMFEngine(config)

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