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

import world
import utils
import model


def BPR_train_original(dataset, loss_class: utils.BPRLoss, epoch: int, w: SummaryWriter = None):
    with utils.timer(name="sampling"):
        S = utils.UniformSample(dataset)
    users = torch.Tensor(S[:, 0]).long()
    pos_items = torch.Tensor(S[:, 1]).long()
    neg_items = torch.Tensor(S[:, 2]).long()

    users = users.to(world.DEVICE)
    pos_items = pos_items.to(world.DEVICE)
    neg_items = neg_items.to(world.DEVICE)
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    total_batch = len(users) // world.CONFIG["bpr_batch_size"] + int(len(users) % world.CONFIG["bpr_batch_size"] > 0)
    aver_loss = 0.0
    with utils.timer(name="predicting"):
        for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(utils.minibatch(users, pos_items, neg_items, batch_size=world.CONFIG["bpr_batch_size"])):
            cri = loss_class.stageOne(batch_users, batch_pos, batch_neg)
            aver_loss += cri
            if world.TENSORBOARD:
                w.add_scalar(f"BPRLoss/BPR", cri, epoch * int(len(users) / world.CONFIG["bpr_batch_size"]) + batch_i)
    aver_loss = aver_loss / total_batch
    result_dict = {
        **{"Loss": aver_loss},
        **{f"Time of {key}": val for key, val in utils.timer.dict().items()}
    }
    utils.timer.zero()
    return result_dict


def test_one_batch(X) -> dict:
    sorted_items = X[0].numpy()
    ground_true = X[1]
    r = utils.get_label(ground_true, sorted_items)
    res = {}
    for k in world.TOPKS:
        ret = utils.RecallPrecision_ATk(ground_true, r, k)
        res[f"Recall@{k}"] = ret["recall"]
        res[f"Precision@{k}"] = ret["precision"]
        res[f"NDCG@{k}"] = utils.NDCGatK_r(ground_true, r, k)
    return res


def test(dataset, rec_model: model.LightGCN, epoch: int, w=None, multicore: bool = False):
    u_batch_size = world.CONFIG["test_u_batch_size"]
    # eval mode
    rec_model.eval()
    max_K = max(world.TOPKS)
    if multicore:
        pool = multiprocessing.Pool(world.CORES)

    results = {}
    with torch.no_grad():
        users = list(dataset.test_dict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            logging.error(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        ground_true_list = []
        total_batch = len(users) // u_batch_size + int(len(users) % u_batch_size > 0)
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            all_positions = dataset.get_user_pos_items(batch_users)
            ground_true = [dataset.test_dict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.DEVICE)

            rating = rec_model.get_users_rating(batch_users_gpu)
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
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        for result in pre_results:
            for key, val in result.items():
                if key not in results:
                    results[key] = val
                else:
                    results[key] += val
        for key, val in results.items():
            results[key] = val / len(users)
        if world.TENSORBOARD:
            for key, val in results.items():
                w.add_scalar(f"Test/{key}", val, epoch)
        if multicore:
            pool.close()
        return results
