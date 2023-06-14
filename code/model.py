"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import logging

import torch

from dataloader import BasicDataset


class LightGCN(torch.nn.Module):
    def __init__(self, n_layers: int, latent_dim: int, A_split: bool, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.A_split = A_split
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = torch.nn.Sigmoid()
        self.Graph = self.dataset.get_sparse_graph()
        logging.info("LightGCN model was successfully initialized")

    def computer(self):
        """
        Compute embeddings for all items and users
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        g = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g)):
                    temp_emb.append(torch.sparse.mm(g[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def get_users_rating(self, users: torch.Tensor):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def get_embedding(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        # Bayesian Personalized Ranking
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
