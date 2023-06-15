"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import torch
from torch import optim
import numpy as np

from dataloader import BasicDataset
from model import LightGCN


class BPRLoss:
    def __init__(self, recmodel: LightGCN, lambda_: float, learning_rate: float):
        self.model = recmodel
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stage_one(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.lambda_
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def UniformSample(dataset: BasicDataset):
    """
    The original impliment of BPR Sampling in LightGCN

    Generate M triples of user_index, positive item (item interacted by the user), negative item (item not interacted by the user).
    Number of M is equal to the number of all interactions.
    Users are selected randomly with possible repetation.
    :return:
        np.array
    """
    user_num = dataset.train_data_size
    users = np.random.randint(0, dataset.n_users, user_num)
    all_positions = dataset.all_positions
    S = []
    for i, user in enumerate(users):
        posForUser = all_positions[user]
        # Users without any interacted items are skipped
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            # TODO fix: this loop will run be broken if user interacted with all items
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)


# ===================end samplers==========================
# =====================utils====================================


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def results_to_progress_log(results: dict):
    for key, val in results.items():
        if isinstance(val, np.ndarray):
            if len(val) == 1:
                results[key] = val[0]
            else:
                results[key] = val.tolist()
    return "; ".join(f"{key}: {val}" for key, val in results.items())


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size")
    if batch_size is None:
        raise ValueError("Missing parameter 'batch_size'")

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have " "the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)
    return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """

    from time import time

    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None) -> dict:
        return {key: val for key, val in timer.NAMED_TAPE.items() if select_keys is None or key in select_keys}

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get("name"):
            timer.NAMED_TAPE[kwargs["name"]] = timer.NAMED_TAPE[kwargs["name"]] if timer.NAMED_TAPE.get(kwargs["name"]) else 0.0
            self.named = kwargs["name"]
            if kwargs.get("group"):
                # TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {"recall": recall, "precision": precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1.0 / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1.0 / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)


def get_label(test_data, pred_data):
    r = []
    # Iterating over a batch
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype("float")


# ====================end Metrics=============================
# =========================================================
