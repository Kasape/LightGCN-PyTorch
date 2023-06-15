"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
"""
import multiprocessing
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader import BasicDataset
import utils
import model


def BPR_train_original(dataset: BasicDataset, batch_size: int, device: torch.device, loss_class: utils.BPRLoss, epoch: int, writer: SummaryWriter = None):
    with utils.timer(name="sampling"):
        S = utils.UniformSample(dataset)
    users = torch.from_numpy(S[:, 0])
    pos_items = torch.from_numpy(S[:, 1])
    neg_items = torch.from_numpy(S[:, 2])

    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    total_batch = len(users) // batch_size + int(len(users) % batch_size > 0)
    aver_loss = 0.0
    with utils.timer(name="predicting"):
        for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(utils.minibatch(users, pos_items, neg_items, batch_size=batch_size)):
            cri = loss_class.stage_one(batch_users, batch_pos, batch_neg)
            aver_loss += cri
            if writer:
                writer.add_scalar(f"BPRLoss/BPR", cri, epoch * int(len(users) / batch_size) + batch_i)
    aver_loss = aver_loss / total_batch
    result_dict = {**{"Loss": aver_loss}, **{f"Time of {key}": val for key, val in utils.timer.dict().items()}}
    utils.timer.zero()
    return result_dict


def test_one_batch(X, topks: list) -> dict:
    sorted_items = X[0].numpy()
    ground_true = X[1]
    r = utils.get_label(ground_true, sorted_items)
    res = {}
    for k in topks:
        ret = utils.RecallPrecision_ATk(ground_true, r, k)
        res[f"Recall@{k}"] = ret["recall"]
        res[f"Precision@{k}"] = ret["precision"]
        res[f"NDCG@{k}"] = utils.NDCGatK_r(ground_true, r, k)
    return res


def test(
    dataset: BasicDataset,
    rec_model: model.LightGCN,
    topks: list,
    batch_size: int,
    device: torch.device,
    n_threads: int,
    epoch: int,
    writer: SummaryWriter = None,
    multicore: bool = False,
):
    # eval mode
    rec_model.eval()
    max_K = max(topks)
    if multicore:
        pool = multiprocessing.Pool(n_threads)

    results = {}
    with torch.no_grad():
        users = list(dataset.test_dict.keys())
        try:
            assert batch_size <= len(users) / 10
        except AssertionError:
            logging.error(f"batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        ground_true_list = []
        total_batch = len(users) // batch_size + int(len(users) % batch_size > 0)
        for batch_users in utils.minibatch(users, batch_size=batch_size):
            all_positions = dataset.get_user_pos_items(batch_users)
            ground_true = [dataset.test_dict[u] for u in batch_users]
            batch_users_gpu = torch.tensor(batch_users).to(device)

            rating = rec_model.predict_users_rating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(all_positions):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            ground_true_list.append(ground_true)
        assert total_batch == len(users_list)
        X = zip(rating_list, ground_true_list)
        if multicore:
            pre_results = pool.map(test_one_batch, (X, topks))
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, topks))
        for result in pre_results:
            for key, val in result.items():
                if key not in results:
                    results[key] = val
                else:
                    results[key] += val
        for key, val in results.items():
            results[key] = val / len(users)
        if writer is not None:
            for key, val in results.items():
                writer.add_scalar(f"Test/{key}", val, epoch)
        if multicore:
            pool.close()
        return results
