"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import logging
import os
import time

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataloader import BasicDataset


class LightGCN(torch.nn.Module):
    def __init__(self, n_layers: int, latent_dim: int, n_users: int, n_items: int, training_graph: torch.Tensor, learning_rate: float, device: torch.device, lambda_: float):
        super(LightGCN, self).__init__()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.__n_users = n_users
        self.__n_items = n_items
        self.__lambda_ = lambda_
        self.__embedding_user = torch.nn.Embedding(num_embeddings=self.__n_users, embedding_dim=self.latent_dim)
        self.__embedding_item = torch.nn.Embedding(num_embeddings=self.__n_items, embedding_dim=self.latent_dim)
        # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
        torch.nn.init.normal_(self.__embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.__embedding_item.weight, std=0.1)
        self.__training_graph = training_graph
        self.__optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.__device = device
        self.to(device)
        logging.info("LightGCN model was successfully initialized")

    def fit(
        self,
        train_data: torch.utils.data.DataLoader,
        epochs: int,
        train_callbacks: list = [],
        validation_data: torch.utils.data.DataLoader = None,
        validation_callbacks: list = [],
        test_data: torch.utils.data.DataLoader = None,
        test_callbacks: list = [],
        savedir: str = None,
        print_interval: int = None,
    ):
        total_steps = len(train_data)
        if print_interval is None:
            print_interval = total_steps

        logging.info("************************** [START] **************************")
        logging.info(f"Runing on {self.__device}.")
        logging.info("Total steps {:n}".format(total_steps))

        for current_epoch in range(1, epochs + 1):
            for callback in train_callbacks:
                callback.start_epoch(current_epoch, epochs, total_steps)
            for step, (user_indices, positive_item_indices, negative_item_indices) in enumerate(train_data, start=1):
                # for callback in train_callbacks:
                #     if isinstance(callback, SavePredictionsCallback):
                #         callback(user_indices=user_indices, item_indices=item_indices, target_ratings=target_ratings)
                self.train_step(user_indices, positive_item_indices, negative_item_indices, callbacks=train_callbacks)
                if step % print_interval == 0 or step == total_steps:
                    for callback in train_callbacks:
                        if isinstance(callback, SaveMetricsCallback):
                            callback.print_metrics()

            if validation_data is not None:
                # for callback in validation_callbacks:
                #     callback.start_epoch(current_epoch, epochs, len(validation_data))
                self.eval_data(validation_data, callbacks=validation_callbacks)

            if test_data is not None:
                # for callback in test_callbacks:
                #     callback.start_epoch(current_epoch, epochs, len(test_data))
                self.eval_data(test_data, callbacks=test_callbacks)

            if savedir is not None:
                torch.save(
                    self.state_dict(),
                    os.path.join(savedir, f"epoch-{current_epoch}.pth"),
                )
            for callback in train_callbacks:
                callback.end_epoch()
        logging.info("************************** [END] **************************")

    def train_step(self, user_indices: torch.Tensor, positive_item_indices: torch.Tensor, negative_item_indices: torch.Tensor, callbacks: list):
        self.train()
        self.zero_grad()
        user_indices.to(self.__device)
        positive_item_indices.to(self.__device)
        negative_item_indices.to(self.__device)
        bpr_loss, reg_loss = self.__bpr_loss(user_indices, positive_item_indices, negative_item_indices)
        loss = bpr_loss + reg_loss * self.__lambda_
        for callback in callbacks:
            if isinstance(callback, SaveMetricsCallback):
                callback(bpr_loss=bpr_loss, reg_loss=reg_loss, loss=loss)
            else:
                raise NotImplementedError(f"Unsupported callback with type {type(callback)}")
        loss.backward()
        self.__optimizer.step()
        torch.cuda.empty_cache()

    # def eval_data(self, dataloader: torch.utils.data.DataLoader, callbacks: list):
    #     self.eval()
    #     with torch.no_grad():
    #         for graphs, user_indices, item_indices, target_ratings in dataloader:
    #             # for callback in callbacks:
    #             #     if isinstance(callback, SavePredictionsCallback):
    #             #         callback(user_indices=user_indices, item_indices=item_indices, target_ratings=target_ratings)
    #             self.predict_and_compute_loss(graphs, callbacks)
    #             torch.cuda.empty_cache()
    #     # for callback in callbacks:
    #     #     if isinstance(callback, SaveMetricsCallback):
    #     #         callback.print_metrics()
    #     #     callback.end_epoch()

    def __compute_embeddings(self):
        """
        Compute embeddings for all items and users
        """
        users_emb = self.__embedding_user.weight
        items_emb = self.__embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embeddings_per_layer = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.__training_graph, all_emb)
            embeddings_per_layer.append(all_emb)
        embeddings_per_layer = torch.stack(embeddings_per_layer, dim=1)
        light_out = torch.mean(embeddings_per_layer, dim=1)
        users, items = torch.split(light_out, [self.__n_users, self.__n_items])
        return users, items

    # For evaluation only
    def predict_users_rating(self, users: torch.Tensor):
        users_emb_all, items_emb = self.__compute_embeddings()
        users_emb = users_emb_all[users]
        rating = torch.sigmoid(users_emb @ items_emb.T)
        return rating

    def __get_embedding(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        all_users, all_items = self.__compute_embeddings()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.__embedding_user(users)
        pos_emb_ego = self.__embedding_item(pos_items)
        neg_emb_ego = self.__embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def __bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        # Bayesian Personalized Ranking
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.__get_embedding(users, pos, neg)
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss


class UniformSamplingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: BasicDataset
    ):
        super(UniformSamplingDataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.dataset.train_data_size

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        """
        Randomly select a user, one of his interacted items as a positive item and one of his non-interacted items as a negative item
        """
        user_index = np.random.randint(0, self.dataset.n_users)
        interacted_items = self.dataset.get_user_pos_items([user_index])[0]
        positive_item_index = np.random.choice(interacted_items)
        while True:
            negative_item_index = np.random.randint(0, self.dataset.m_items)
            # this loop will run indefinitely if user interacted with all items
            if negative_item_index in interacted_items:
                continue
            else:
                break
        return user_index, positive_item_index, negative_item_index


class SaveMetricsCallback:
    BPR_LOSS_ATTR = "bpr_loss"
    REGULARIZATION_LOSS_ATTR = "reg_loss"
    LOSS_ATTR = "loss"

    def __init__(self, subset_name: str, writer: SummaryWriter = None):
        self.subset_name = subset_name
        self.writer = writer
        self.dynamic_attributes = set()
        self.__reset()
        # Attributes that can be added by '__call__' method and they are deleted in 'end_epoch' method
        self.supported_keys = [self.BPR_LOSS_ATTR, self.REGULARIZATION_LOSS_ATTR, self.LOSS_ATTR]

    def start_epoch(self, current_epoch: int, total_epochs: int, total_steps: int):
        self.start_time = time.time()
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        self.current_step = 0
        self.total_steps = total_steps

    def print_metrics(self):
        elapsed_time = time.time() - self.start_time
        log_dict = {
            "Epoch": f"{self.current_epoch}/{self.total_epochs}",
            "Step": f"{self.current_step}/{self.total_steps}",
            "Time": f"{elapsed_time:2f}s",
            "Subset": self.subset_name,
        }
        log_dict.update(self.__aggregate_metrics())
        logging.info(self.get_progress_log(log_dict))

    def __aggregate_metrics(self):
        metrics = {}
        for param_to_agg in self.supported_keys:
            val = getattr(self, param_to_agg)
            metrics[f"{param_to_agg}"] = torch.mean(torch.Tensor(val)).item()
        return metrics

    def __call__(self, **kwargs):
        self.current_step += 1
        for key, val in kwargs.items():
            if key not in self.supported_keys:
                raise ValueError(f"{type(self).__name__}: Key '{key}' is not listed in the supported keys {self.supported_keys}")
            self.dynamic_attributes.add(key)
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu()
            if hasattr(self, key):
                getattr(self, key).append(val)
            else:
                setattr(self, key, [val])

    def __reset(self):
        self.current_step = 0
        for attr in self.dynamic_attributes:
            delattr(self, attr)
        self.dynamic_attributes.clear()

    def end_epoch(self):
        if self.writer is not None:
            for metric_name, value in self.__aggregate_metrics().items():
                self.writer.add_scalar(f"{metric_name}/{self.subset_name}", value, self.current_epoch)
        self.__reset()

    @staticmethod
    def get_progress_log(values: dict, delimiter: str = "; "):
        def format_number(v):
            if isinstance(v, float):
                return str(round(v, 4))
            return str(v)

        return delimiter.join(f"{key}: {format_number(val)}" for key, val in values.items())
