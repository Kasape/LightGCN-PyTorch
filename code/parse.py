"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument("--n-threads", type=int, default=-1, help="Number of threads")
    parser.add_argument("--dataset", type=str, choices=["lastfm", "gowalla", "yelp2018", "amazon-book"])
    parser.add_argument("--layers", type=int, default=3, help="the number of layers lightGCN")
    parser.add_argument("--latent-dim", type=int, default=64, help="the embedding size of lightGCN")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-4, help="coefficient for l2 normalizaton")
    parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--train-batch", type=int, default=2048, help="the batch size for bpr loss training procedure")
    parser.add_argument("--test-batch", type=int, default=100, help="the batch size of users for testing")
    parser.add_argument("--topks", nargs="+", type=int, default=20, help="@k test list")
    parser.add_argument("--tensorboard", type=int, default=1, help="enable tensorboard")
    parser.add_argument("--load-previous", default=False, action="store_true", help="Load previous run")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    return parser.parse_args()
