import torch
from src.engine import Engine
from src.utils import use_cuda, resume_checkpoint
import  numpy as np

class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.config = config
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        config = self.config
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
        self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item.weight.data = gmf_model.embedding_item.weight.data

        user_id = np.ndarray(shape=(1,6040))
        item_id = np.ndarray(shape=(1, 3706))
        for i in range(0,6039):
            user_id[0,i] = i
        for j in range(0,3705):
            item_id[0,j] = j
        user_indices =  torch.LongTensor(user_id).cuda()
        item_indices =  torch.LongTensor(item_id).cuda()
        user_embedding = self.embedding_user(user_indices).cpu()
        item_embedding = self.embedding_item(item_indices).cpu()
        user_embedding = user_embedding.detach().numpy()
        item_embedding = item_embedding.detach().numpy()
        np.save("./user_em", user_embedding)
        np.save("./item_em",item_embedding)


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = GMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
            print(next(self.model.parameters()).device)
        super(GMFEngine, self).__init__(config)

        if config['pretrain']:
            self.model.load_pretrain_weights()

    def get_item_vector(self):
        return self.model.embedding_item
