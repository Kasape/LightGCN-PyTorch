"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
from os.path import join
from time import time
import logging

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp


class BasicDataset(torch.utils.data.Dataset):
    @property
    def n_users(self) -> int:
        raise NotImplementedError

    @property
    def m_items(self) -> int:
        raise NotImplementedError

    @property
    def train_data_size(self) -> int:
        raise NotImplementedError

    @property
    def test_dict(self) -> dict:
        """
        Returns dictionary where keys are users (integers),
        values are lists of items (integers) interacted by a user
        """
        raise NotImplementedError

    @property
    def all_positions(self) -> list:
        """
        Returns list with length n_users containing numpy arrays
        """
        raise NotImplementedError

    def get_user_pos_items(self, users: list) -> list:
        """
        Returns list of numpy array containing positions of interacted items for each given user
        """
        raise NotImplementedError

    def get_sparse_graph(self) -> torch.Tensor:
        """
        Sparse tensor layout torch.sparse_coo with shape (n_users + m_items, n_users + m_items)
        in matrix form described in Neural Graph Collaborative Filtering paper.

        Build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset type for pytorch
    Include graph information
    LastFM dataset
    """

    def __init__(self, device: torch.device, path="data/lastfm"):
        # train or test
        logging.info("Creating dataset LastFM")
        trainData = pd.read_table(join(path, "data1.txt"), header=None)
        testData = pd.read_table(join(path, "test1.txt"), header=None)
        trustNet = pd.read_table(join(path, "trustnetwork.txt"), header=None).to_numpy()
        trustNet -= 1
        trainData -= 1
        testData -= 1
        self.trustNet = trustNet
        self.trainData = trainData
        self.testData = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.train_data_size = len(self.trainUser)
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(testData[:][1])
        self.Graph = None
        logging.info(f"LastFM Sparsity : {((len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items * 100):2.f}%")

        # (users,users)
        self.socialNet = sp.csr_matrix((np.ones(len(trustNet)), (trustNet[:, 0], trustNet[:, 1])), shape=(self.n_users, self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet = sp.csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_users, self.m_items))

        # pre-calculate
        self.__all_positions = self.get_user_pos_items(list(range(self.n_users)))
        self.__test_dict = self.__build_test()

    @property
    def n_users(self):
        return 1892

    @property
    def m_items(self):
        return 4489

    @property
    def train_data_size(self):
        return len(self.trainUser)

    @property
    def test_dict(self):
        return self.__test_dict

    @property
    def all_positions(self):
        return self.__all_positions

    def get_sparse_graph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.0] = 1.0
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
            self.Graph = self.Graph.coalesce().to(self.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def __len__(self):
        return len(self.trainUniqueUsers)


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Include graph information
    Can be used for provided data of gowalla, yelp2018 and amazon-book dataset
    """

    def __init__(self, path: str, device: torch.device):
        # train or test
        logging.info(f"Creating dataset from path {path}")
        self.device = device
        self.n_user = 0
        self.m_item = 0
        train_file = path + "/train.txt"
        test_file = path + "/test.txt"
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.__train_data_size = 0
        self.__test_data_size = 0

        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip("\n").split(" ")
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.__train_data_size += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip("\n").split(" ")
                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.__test_data_size += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        logging.info(f"Dataset from '{self.path}' contains {self.__train_data_size} interactions for training")
        logging.info(f"Dataset from '{self.path}' contains {self.__test_data_size} interactions for testing")
        logging.info(f"Dataset from '{self.path}' has sparsity {(self.__train_data_size + self.__test_data_size) / self.n_users / self.m_items * 100:2f}%")

        # (users,items), bipartite graph
        self.UserItemNet = sp.csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item))
        # pre-calculate
        self.__all_positions = self.get_user_pos_items(list(range(self.n_user)))
        self.__test_dict = self.__build_test()
        logging.info(f"Dataset from '{self.path}' has been initialized")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def train_data_size(self):
        return self.__train_data_size

    @property
    def test_dict(self):
        return self.__test_dict

    @property
    def all_positions(self):
        return self.__all_positions

    def get_sparse_graph(self):
        logging.info("Loading adjacency matrix")
        path = join(self.path, "s_pre_adj_mat.npz")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(path)
                logging.info("Adjacency matrix was successfully loaded")
                norm_adj = pre_adj_mat
            except Exception:
                logging.info(f"Adjacency matrix was not found on {path}, generating new one and saving it")
                start = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[: self.n_users, self.n_users :] = R
                adj_mat[self.n_users :, : self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                logging.info(f"Adjacency matrix was created and it took {end - start}s, saving it...")
                sp.save_npz(path, norm_adj)

            self.Graph = self.__convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)
        return self.Graph

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append((self.UserItemNet[user].nonzero()[1]).astype(np.int64))
        return posItems

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.from_numpy(coo.row)
        col = torch.from_numpy(coo.col)
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
