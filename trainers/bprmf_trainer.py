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
from utils.utils import minibatch


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, model_args):
        super(BPRMF, self).__init__()

        self.dim = model_args.hidden_dims

        self.n_users = n_users
        self.n_items = n_items

        self.Embedding_User = nn.Embedding(self.n_users, self.dim)
        self.Embedding_Item = nn.Embedding(self.n_items, self.dim)
        nn.init.normal_(self.Embedding_User.weight, std=0.01)
        nn.init.normal_(self.Embedding_Item.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        users_emb = self.Embedding_User(user_ids)
        items_emb = self.Embedding_Item(item_ids)
        scores = torch.sum(users_emb * items_emb, dim=-1)
        return scores

    def loss(self,user_ids, pos_items, neg_items):
        users_emb = self.Embedding_User(user_ids)
        pos_items_emb = self.Embedding_Item(pos_items)
        neg_items_emb = self.Embedding_Item(neg_items)

        pos_scores = torch.sum(users_emb * pos_items_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=-1)
        loss = (pos_scores - neg_scores).sigmoid().log().mean()
        return -loss



class BPRMFTrainer(BaseTrainer):
    def __init__(self, n_users, n_items, args):
        super(BPRMFTrainer, self).__init__()
        self.args = args

        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")

        self.n_users = n_users
        self.n_items = n_items

        self.metrics = self.args.metrics

    def _initialize(self):
        model_args = Bunch(self.args.model)     
        if not hasattr(model_args, "n_negatives"):
                model_args.n_negatives = 5
        self._n_negatives = model_args.n_negatives
        model_args.l2 = self.args.l2
        self.net = BPRMF(n_users=self.n_users, n_items=self.n_items,model_args=model_args).to(self.device)     

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

        idx_list = np.arange(user_ids.shape[0])

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

            batch_pos_items = pos_item_ids[batch_idx]
            batch_neg_items =neg_item_ids[batch_idx]
            batch_pos_items = torch.LongTensor(batch_pos_items).to(self.device)
            batch_neg_items = torch.LongTensor(batch_neg_items).to(self.device)


            # Compute loss
            outputs = model(user_ids=batch_users, item_ids=batch_items)
            loss = model.loss(user_ids=batch_users,pos_items=batch_pos_items, neg_items=batch_neg_items)
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
