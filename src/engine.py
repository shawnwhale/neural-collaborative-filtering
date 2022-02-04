import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from src.utils import save_checkpoint, use_optimizer
from src.metrics import MetronAtK
import numpy as np

class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def get_item_vector(self):
        pass

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()
            test_scores = self.model(test_users, test_items)
            negative_scores = self.model(negative_users, negative_items)

            # # 转化， 流行度调整
            # items_popu_tran_dic = np.load("./items_popu_tran_dic.npy", allow_pickle=True).tolist()
            # test_items = test_items.cpu().detach().numpy()
            # test_scores = test_scores.cpu().detach().numpy()
            # negative_items = negative_items.cpu().detach().numpy()
            # negative_scores = negative_scores.cpu().detach().numpy()
            # m = 0  # 序号
            # for temp in test_items:
            #     arfa = items_popu_tran_dic[temp]  # 阿尔法是流行度调整参数
            #     # test_scores[m, :] = (test_scores[m, :] * 0.7 + 0.3)* arfa * 0.3 + test_scores[m, :]
            #     test_scores[m, :] = test_scores[m, :] * arfa
            #     m = m + 1
            # n = 0  # 序号
            # for temp in negative_items:
            #     arfa = items_popu_tran_dic[temp]  # 阿尔法是流行度调整参数
            #     # negative_scores[n, :] = (negative_scores[n, :] * 0.7 + 0.3)* arfa * 0.3 + negative_scores[n, :]
            #     negative_scores[n, :] = negative_scores[n, :] * arfa
            #     n = n + 1
            # test_scores = torch.FloatTensor(test_scores).cuda()
            # negative_scores = torch.FloatTensor(negative_scores).cuda()
            # test_items = torch.LongTensor(test_items).cuda()
            # negative_items = torch.LongTensor(negative_items).cuda()

            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        ils=self._metron.cal_ils()
        kendall = self._metron.cal_kendall()
        # entropy = self._metron.cal_entropy()
        # ils = 0
        # kendall = 0
        entropy = 0
        # self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        # self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}, ILS = {:.4f}, KENDALL = {:.4f}, entropy = {:.4f}'.format(epoch_id, hit_ratio, ndcg ,ils , kendall,entropy))
        return hit_ratio, ndcg,ils ,kendall,entropy

    def save(self, alias, epoch_id, hit_ratio, ndcg,ils,kendall,entropy):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg,ils,kendall,entropy)
        save_checkpoint(self.model, model_dir)