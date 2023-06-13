"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn), modified by Petr Kasalicky (kasalpe1@fit.cvut.cz)
"""

import os
import logging

import torch

from parse import parse_args

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = os.path.join(ROOT_PATH, "code")
DATA_PATH = os.path.join(ROOT_PATH, "data")
BOARD_PATH = os.path.join(CODE_PATH, "runs")
FILE_PATH = os.path.join(CODE_PATH, "checkpoints")

os.makedirs(FILE_PATH, exist_ok=True)

if args.n_threads == -1:
    system_cpu = os.cpu_count()
    if system_cpu is None:
        logging.info("Number of threads using n-threads parameter were not explicitely provided and it cannot be obtained from the system. Setting number of threads to 1.")
    else:
        logging.info(f"Number of threads using n-threads parameter were not explicitely provided. Setting number of threads to {system_cpu} provided by the system")
    args.n_threads = system_cpu or 1

if torch.get_num_threads() != args.n_threads:
    torch.set_num_threads(args.n_threads)
if torch.get_num_interop_threads() != args.n_threads:
    torch.set_num_interop_threads(args.n_threads)

CONFIG = {}
all_dataset = ["lastfm", "gowalla", "yelp2018", "amazon-book"]
# config['batch_size'] = 4096
CONFIG["bpr_batch_size"] = args.bpr_batch
CONFIG["latent_dim_rec"] = args.recdim
CONFIG["lightGCN_n_layers"] = args.layer
CONFIG["A_n_fold"] = args.a_fold
CONFIG["test_u_batch_size"] = args.testbatch
CONFIG["multicore"] = args.multicore
CONFIG["lr"] = args.lr
CONFIG["decay"] = args.decay
CONFIG["A_split"] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORES = args.n_threads
SEED = args.seed
TRAIN_EPOCHS = args.epochs
LOAD_PREVIOUS = args.load_previous
TOPKS = args.topks
TENSORBOARD = args.tensorboard
DATASET = args.dataset

if DATASET not in all_dataset:
    raise NotImplementedError(f"Haven't supported {DATASET} yet!, try {all_dataset}")
