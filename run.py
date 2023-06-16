import time
import os
import logging
import argparse

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import dataloader
from model import LightGCN, UniformSamplingDataset, SaveMetricsCallback

LOGGING_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument("--n-threads", type=int, default=-1, help="Number of threads")
    parser.add_argument("--dataset", type=str, choices=["lastfm", "gowalla", "yelp2018", "amazon-book"])
    parser.add_argument("--layers", type=int, default=3, help="The number of layers lightGCN")
    parser.add_argument("--latent-dim", type=int, default=64, help="The embedding size of lightGCN")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-4, help="Coefficient for l2 normalizaton")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--train-batch", type=int, default=2048, help="The batch size for bpr loss training procedure")
    parser.add_argument("--eval-batch", type=int, default=100, help="The batch size of users for evaluation (validation and test)")
    parser.add_argument("--topks", nargs="+", type=int, default=20, help="List of K to measure NDCG@K and Recall@K")
    parser.add_argument("--tensorboard", type=int, default=1, help="Enable tensorboard")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    CODE_PATH = os.path.join(ROOT_PATH, "code")
    DATA_PATH = os.path.join(ROOT_PATH, "data")
    BOARD_PATH = os.path.join(CODE_PATH, "runs")

    if args.n_threads == -1:
        system_cpu = os.cpu_count()
        if system_cpu is None:
            logging.info(
                "Number of threads using n-threads parameter were not explicitely provided and it cannot be obtained from the system. Setting number of threads to 1."
            )
        else:
            logging.info(
                f"Number of threads using n-threads parameter were not explicitely provided. Setting number of threads to {system_cpu} provided by the system"
            )
        args.n_threads = system_cpu or 1

    if torch.get_num_threads() != args.n_threads:
        torch.set_num_threads(args.n_threads)
    if torch.get_num_interop_threads() != args.n_threads:
        torch.set_num_interop_threads(args.n_threads)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Arguments are: {vars(args)}")
    logging.info(f"Setting a random seed to: {args.seed}")
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset in ["gowalla", "yelp2018", "amazon-book"]:
        path = os.path.join("data", args.dataset)
        dataset = dataloader.Loader(path=path, device=DEVICE)
    elif args.dataset == "lastfm":
        dataset = dataloader.LastFM(device=DEVICE)
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    model = LightGCN(
        args.layers, args.latent_dim, n_users=dataset.n_users,
        n_items=dataset.m_items, training_graph=dataset.get_sparse_graph(), device=DEVICE,
        learning_rate=args.lr, lambda_=args.lambda_
    )

    train_dataloader = torch.utils.data.DataLoader(
        UniformSamplingDataset(dataset),
        shuffle=True,
        batch_size=args.train_batch,
        num_workers=args.n_threads
    )

    # init tensorboard
    if args.tensorboard:
        path = os.path.join(BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "LGN")
        writer = SummaryWriter(path)
        logging.info(f"Tensorboard was inicitialized to path={path}")
    else:
        writer = None
        logging.info("Tensorboard will not be used")

    train_callbacks = [SaveMetricsCallback(subset_name="train", writer=writer)]
    model.fit(train_dataloader, epochs=10, train_callbacks=train_callbacks)

# try:
#     for epoch in range(1, args.epochs + 1):
#         rec_model.train()
#         results = procedure.BPR_train_original(dataset, args.train_batch, DEVICE, bpr, epoch, writer=writer)
#         logging.info(f"[TRAIN] Epoch: {epoch}/{args.epochs}; {utils.results_to_progress_log(results)}")
#         if epoch % 1 == 0:
#             results = procedure.test(dataset, rec_model, args.topks, args.test_batch, DEVICE, epoch, writer)
#             logging.info(f"[TEST] Epoch: {epoch}/{args.epochs}; {utils.results_to_progress_log(results)}")
#             if epoch == 1:
#                 assert round(results["Recall@20"], 4) == round(0.08642416, 4)
#                 assert round(results["Precision@20"], 4) == round(0.02703128, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.07227541, 4)
# 
#             if epoch == 2:
#                 assert round(results["Recall@20"], 4) == round(0.09613762, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03018119, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.08025454, 4)
# 
#             if epoch == 3:
#                 assert round(results["Recall@20"], 4) == round(0.10085652, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03160627, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.08451248, 4)
# 
#             if epoch == 4:
#                 assert round(results["Recall@20"], 4) == round(0.10535001, 4)
#                 assert round(results["Precision@20"], 4) == round(0.0328103, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.08842302, 4)
# 
#             if epoch == 5:
#                 assert round(results["Recall@20"], 4) == round(0.10873527, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03369951, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.0915894, 4)
# 
#             if epoch == 6:
#                 assert round(results["Recall@20"], 4) == round(0.11204279, 4)
#                 assert round(results["Precision@20"], 4) == round(0.0345636, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.09433213, 4)
# 
#             if epoch == 7:
#                 assert round(results["Recall@20"], 4) == round(0.1144635, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03518655, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.09638503, 4)
# 
#             if epoch == 8:
#                 assert round(results["Recall@20"], 4) == round(0.1166778, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03579945, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.09808936, 4)
# 
#             if epoch == 9:
#                 assert round(results["Recall@20"], 4) == round(0.11843803, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03622815, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.09946748, 4)
# 
#             if epoch == 10:
#                 assert round(results["Recall@20"], 4) == round(0.12013993, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03667024, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.10055132, 4)
# 
#             if epoch == 11:
#                 assert round(results["Recall@20"], 4) == round(0.12134588, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03699344, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.10154486, 4)
# 
#             if epoch == 12:
#                 assert round(results["Recall@20"], 4) == round(0.12283394, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03736352, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.10243225, 4)
# 
#             if epoch == 13:
#                 assert round(results["Recall@20"], 4) == round(0.12388475, 4)
#                 assert round(results["Precision@20"], 4) == round(0.0376415, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.10322438, 4)
# 
#             if epoch == 14:
#                 assert round(results["Recall@20"], 4) == round(0.1251952, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03798814, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.10429674, 4)
# 
#             if epoch == 15:
#                 assert round(results["Recall@20"], 4) == round(0.12649249, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03834148, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.10514867, 4)
# 
#             if epoch == 16:
#                 assert round(results["Recall@20"], 4) == round(0.12744638, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03861612, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.1058937, 4)
# 
#             if epoch == 17:
#                 assert round(results["Recall@20"], 4) == round(0.12834675, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03886563, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.10649789, 4)
# 
#             if epoch == 18:
#                 assert round(results["Recall@20"], 4) == round(0.12956465, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03916203, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.10737581, 4)
# 
#             if epoch == 19:
#                 assert round(results["Recall@20"], 4) == round(0.1303976, 4)
#                 assert round(results["Precision@20"], 4) == round(0.03941155, 4)
#                 assert round(results["NDCG@20"], 4) == round(0.10797482, 4)
# 
#         torch.save(rec_model.state_dict(), weight_file)
# finally:
#     if writer is not None:
#         writer.close()
