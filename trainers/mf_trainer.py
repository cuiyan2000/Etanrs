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
from trainers.losses import mse_loss1
from utils.utils import minibatch
import time

from utils.utils import  sparse2tensor, tensor2sparse, minibatch





class MF(nn.Module):
    def __init__(self, n_users, n_items, model_args):
        super(MF, self).__init__()

        hidden_dims = model_args.hidden_dims
        if len(hidden_dims) > 1:
            raise ValueError("WMF can only have one latent dimension.")

        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dims[0]

        self.Q = nn.Parameter(torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.01))
        self.P = nn.Parameter(torch.zeros([self.n_users, self.dim]).normal_(mean=0, std=0.01))


        self.params = nn.ParameterList([self.Q, self.P])
        
        # self.loss_func = model_args.loss_func


    def forward(self, user_id=None, item_id=None):
        if user_id is None and item_id is None:
            return torch.mm(self.P, self.Q.t())
        if user_id is not None:
            return torch.mm(self.P[[user_id]], self.Q.t())
        if item_id is not None:
            return torch.mm(self.P, self.Q[[item_id]].t())

    # def loss(self, outputs, data):
    #     return (self.loss_func(outputs,data).mean() )



    


class MFTrainer(BaseTrainer):
    def __init__(self, n_users, n_items, args):
        super(MFTrainer, self).__init__()
        self.args = args

        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")

        self.n_users = n_users
        self.n_items = n_items

        self.metrics = self.args.metrics
        

    def _initialize(self): 
        model_args = Bunch(self.args.model)
        if not hasattr(model_args, "optim_method"):
            model_args.optim_method = "sgd"
       
        # model_args.loss_func =  torch.nn.MSELoss().to(self.device)
        self.net = MF(
            n_users=self.n_users, n_items=self.n_items,
            model_args=model_args).to(self.device)
        
        print(self)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.l2)

        self.optim_method = model_args.optim_method
        self.dim = self.net.dim
        
    def train_epoch(self, *args, **kwargs):
        if self.optim_method not in ("sgd", "als"):
            raise ValueError("Unknown optim_method {} for WMF.".format(
                self.optim_method))

        if self.optim_method == "sgd":
            return self.train_sgd(*args, **kwargs)

       
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
                batch_data = data[batch_idx].toarray()

                preds = model(user_id=batch_idx)
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
 
        
    def train_sgd(self, data):
        n_rows = data.shape[0]
        n_cols = data.shape[1]
        idx_list = np.arange(n_rows)

        # Set model to training mode.
        model = self.net.to(self.device)
        model.train()
        np.random.shuffle(idx_list)

        epoch_loss = 0.0
        batch_size = (self.args.batch_size
                      if self.args.batch_size > 0 else len(idx_list))
        for batch_idx in minibatch(
                idx_list, batch_size=batch_size):
            batch_tensor = sparse2tensor(data[batch_idx]).to(self.device)

            # Compute loss
            outputs = model(user_id=batch_idx)
            # loss = model.loss(outputs=outputs,data=batch_tensor).sum()
            loss =mse_loss1(data=batch_tensor,logits=outputs).sum()
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss

    def fit_adv(self, *args, **kwargs):
        self._initialize()

        if self.optim_method not in ("sgd", "als"):
            raise ValueError("Unknown optim_method {} for WMF.".format(
                self.optim_method))

        if self.optim_method == "sgd":
            return self.fit_adv_sgd(*args, **kwargs)
        if self.optim_method == "als":
            return self.fit_adv_als(*args, **kwargs)

    def fit_adv_sgd(self, data_tensor, epoch_num, unroll_steps,
                    n_fakes, target_items):
        import higher
        
        
        

        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        data_tensor = data_tensor.to(self.device)
        target_tensor = torch.zeros_like(data_tensor)
        target_tensor[:, target_items] = 1.0
        n_rows = data_tensor.shape[0]
        n_cols = data_tensor.shape[1]
        idx_list = np.arange(n_rows)

        model = self.net.to(self.device)
        optimizer = self.optimizer

        batch_size = (self.args.batch_size
                      if self.args.batch_size > 0 else len(idx_list))
        for i in range(1, epoch_num - unroll_steps + 1):
            t1 = time.time()
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0
            for batch_idx in minibatch(idx_list, batch_size=batch_size):
                # outputs = model(user_id=batch_idx)
                # Compute loss
                # loss =  model.loss(outputs=outputs,data=data_tensor[batch_idx]).sum()
                loss=mse_loss1(data=data_tensor[batch_idx],logits=model(user_id=batch_idx)).sum()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                time.time() - t1, i, epoch_loss))

        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            print("Switching to higher mode...")
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                t1 = time.time()
                np.random.shuffle(idx_list)
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in minibatch(
                        idx_list, batch_size=batch_size):
                    # outputs = fmodel(user_id=batch_idx)
                    # Compute loss
                    # loss = model.loss(outputs=outputs,data=data_tensor[batch_idx]).sum()
                    loss = mse_loss1(data=data_tensor[batch_idx],logits=fmodel(user_id=batch_idx)).sum()
                    epoch_loss += loss.item()
                    diffopt.step(loss)

                print("Training (higher mode) [{:.1f} s],"
                      " epoch: {}, loss: {:.4f}".format(time.time() - t1, i, epoch_loss))

            print("Finished surrogate model training,"
                  " {} copies of surrogate model params.".format(len(fmodel._fast_params)))

            fmodel.eval()
            predictions = fmodel()
            # Compute adversarial (outer) loss.
            adv_loss = mult_ce_loss(
                logits=predictions[:-n_fakes, ],
                data=target_tensor[:-n_fakes, ]).sum()
            adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
            # Copy fmodel's parameters to default trainer.net().
            model.load_state_dict(fmodel.state_dict())

        return adv_loss.item(), adv_grads[-n_fakes:, ]
