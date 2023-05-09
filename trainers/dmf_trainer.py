import random
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bunch import Bunch

from trainers.base_trainer import BaseTrainer
from trainers.losses import *
from utils.utils import sparse2tensor, minibatch

from data.data_loader import *




class DMF(nn.Module):
    def __init__(self, n_users, n_items, model_args):
        super(DMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items

        # load parameters info
        self.userlayers =model_args.userlayers
        self.itemlayers = model_args.itemlayers
        self.bce_loss =binary_ce_loss


        # 构建初始的embedding向量，这里user embedding是利用user对历史item的打分vector
        # item embedding 是利用item历史被若干user打分的vector
        self.user_item_matrix = self.generate_useritem_matrix()
        weight_user_item = torch.FloatTensor(self.user_item_matrix)
        weight_item_user = torch.FloatTensor(self.user_item_matrix.T)
        self.user_embed = nn.Embedding.from_pretrained(weight_user_item, freeze=True)
        self.item_embed = nn.Embedding.from_pretrained(weight_item_user, freeze=True)
        
        self.fc_user1 = nn.Linear(self.n_items, self.userlayers[0])
        self.fc_item1 = nn.Linear(self.n_users, self.itemlayers[0])
        self.fc_user2 = nn.Linear(self.userlayers[0], self.n_items)
        self.fc_item2 = nn.Linear(self.itemlayers[0],  self.n_items)
        
    def generate_useritem_matrix(self):
          tp = pd.read_csv("/home/yuanb/adv/revisit_adv_rec_other/data/gowalla/train.csv")
          rows, cols = tp['uid'], tp['sid']

          mat = sparse.dok_matrix((self.n_users, self.n_items),dtype=np.float64)
          mat[rows,cols]=1.0

          train_matrix = np.zeros([self.n_users, self.n_items], dtype=np.float64)
          for (u, i) in mat.keys():
             train_matrix[u][i] = 1.0

          return train_matrix

    def forward(self, user_ids, item_ids):
        user_input = self.user_embed(user_ids)
        item_input = self.item_embed(item_ids)
        hidden1_user = self.fc_user1(user_input)
        hidden1_item = self.fc_item1(item_input)
        user_output = F.relu(self.fc_user2(hidden1_user))
        item_output = F.relu(self.fc_item2(hidden1_item))
        norm_user_output = torch.sqrt(torch.sum(user_output**2, dim=1))
        norm_item_output = torch.sqrt(torch.sum(item_output**2, dim=1))
        predict = torch.sum(user_output*item_output,dim=1)/(norm_user_output*norm_item_output)
        predict = torch.sum(user_output*item_output,dim=1)
        predict = torch.clamp(predict,1e-6)
        return predict

    def loss(self,data,outputs):
        loss =data*torch.log(outputs.double()) + (1-data)*torch.log(1-outputs.double())
        return loss
        # return self.bce_loss(data,outputs)

class DMFTrainer(BaseTrainer):
    def __init__(self, n_users, n_items, args):
        super(DMFTrainer, self).__init__()
        self.args = args

        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")

        self.n_users = n_users
        self.n_items = n_items
           
        self.metrics = self.args.metrics

    def _initialize(self):
        model_args = Bunch(self.args.model)
        
        if model_args.model_type == "DMF":
            if not hasattr(model_args, "n_negatives"):
                model_args.n_negatives = 7
            self._n_negatives = model_args.n_negatives
            self.net = DMF(n_users=self.n_users, n_items=self.n_items,model_args=model_args).to(self.device)
        else:
            raise ValueError("Unknown model type {}".format(model_args.model_type))
        print(self)
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.args.lr)
        # For negative sampling.
        self.user_rated_items = None

    def _sample_negative(self, user_ids, pos_item_ids, n_negatives):
        samples = np.empty((user_ids.shape[0], n_negatives), np.int64)
        if self.user_rated_items is None:
            self.user_rated_items = dict()
            for user in range(self.n_users):
                self.user_rated_items[user] = pos_item_ids[user_ids == user]

        for i, u in enumerate(user_ids):
            j = 0
            rated_items = self.user_rated_items[u]
            while j < n_negatives:
                sample_item = int(random.random() * self.n_items)
                if sample_item not in rated_items:
                    samples[i, j] = sample_item
                    j += 1
        return samples

    def train_epoch(self, data):
        # Get training pairs and sample negatives.
        user_ids, pos_item_ids = (data > 0).nonzero()
        neg_item_ids = self._sample_negative(user_ids, pos_item_ids,self._n_negatives)
        user_ids = np.expand_dims(user_ids, 1)
        pos_item_ids = np.expand_dims(pos_item_ids, 1)
        combined_item_ids = np.concatenate([pos_item_ids, neg_item_ids], 1)

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)

        # Set model to training mode.
        model = self.net.to(self.device)
        model.train()
        np.random.shuffle(idx_list)

        epoch_loss = 0.0
        counter = 0
        for batch_idx in minibatch(idx_list, batch_size=self.args.batch_size):
            
            batch_users = user_ids[batch_idx]
            batch_items = combined_item_ids[batch_idx]
            batch_users = torch.LongTensor(batch_users).to(self.device)
            batch_items = torch.LongTensor(batch_items).to(self.device)
            batch_tensor = sparse2tensor(data[batch_idx]).to(self.device)
            # Compute loss
            outputs = model(user_ids=batch_users, item_ids=batch_items)
            # loss = model.loss(data=batch_tensor,outputs=outputs).mean()
            loss=model.loss(data=batch_tensor,outputs=outputs)
            loss = -torch.mean(loss)

            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            counter += 1

        return epoch_loss / counter

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)
        all_preds = list()
        with torch.no_grad():
            for batch_idx in minibatch(
                    idx_list, batch_size=self.args.valid_batch_size):
                cur_batch_size = batch_idx.shape[0]
                batch_data = data[batch_idx].toarray()

                batch_users = np.expand_dims(batch_idx, 1)
                batch_users = torch.LongTensor(batch_users).to(self.device)

                all_items = np.arange(self.n_items)[None, :]
                all_items = np.tile(all_items, (cur_batch_size, 1))
                all_items = torch.LongTensor(all_items).to(self.device)

                preds = model(user_ids=batch_users, item_ids=all_items)
                if return_preds:
                    all_preds.append(preds)
                if not allow_repeat:
                    preds[batch_data.nonzero()] = -np.inf
                if top_k > 0:
                    _, recs = preds.topk(k=top_k, dim=1)
                    recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, torch.cat(all_preds, dim=0).cpu()
        else:
            return recommendations
