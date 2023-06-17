import os
import logging
import argparse

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.RodrigoDataset import RodrigoDataset
from model import LightGCN, UniformSamplingDataset, AverageMetricsCallback, SparseMatrixDataset

LOGGING_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument("--n-threads", type=int, default=-1, help="Number of threads")
    parser.add_argument("--dataset-folder", help="Path to a folder containing 'ratings_train.npy'")
    parser.add_argument("--layers", type=int, default=3, help="The number of layers lightGCN")
    parser.add_argument("--emb-size", type=int, default=64, help="The embedding size of lightGCN")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-4, help="Coefficient for l2 normalizaton")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--train-batch", type=int, default=2048, help="The batch size for bpr loss training procedure")
    parser.add_argument("--eval-batch", type=int, default=100, help="The batch size of users for evaluation (validation and test)")
    parser.add_argument("--topks", nargs="+", type=int, default=20, help="List of K to measure NDCG@K and Recall@K")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--debug", default=False, action="store_true", help="When enabled, produced models are not saved and metrics are not logged into tensorboard")
    parser.add_argument("--print-interval", type=int, default=None, help="Interval for reporting metrics during training of model")
    return parser, parser.parse_args()


def setup_libraries(args: argparse.Namespace):
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

    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)


if __name__ == "__main__":
    parser, args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Arguments are: {vars(args)}")
    setup_libraries(args)

    dataset_name = os.path.split(args.dataset_folder.removesuffix("/"))[-1]
    args.dataset_folder += "/" if not args.dataset_folder.endswith("/") else ""
    logging.info(f"Name of the dataset extracted from dataset-folder={args.dataset_folder} is {dataset_name}")
    dataset = RodrigoDataset.load(args.dataset_folder)
    dataset.to_implicit(inplace=True)

    hyperparams_name = f"s_{args.seed}_bs_{args.train_batch}_ep_{args.epochs}_ly_{args.layers}_es_{args.emb_size}_la_{args.lambda_}_lr_{args.lr}"

    logging.info(f"Model name: {hyperparams_name}")

    if args.debug:
        savedir = None
        writer = None
    else:
        savedir = os.path.join(args.dataset_folder, hyperparams_name)
        os.makedirs(savedir, exist_ok=True)
        writer = SummaryWriter(log_dir=savedir)

    model = LightGCN(
        args.layers,
        args.emb_size,
        n_users=dataset.get_users_cnt(),
        n_items=dataset.get_items_cnt(),
        training_sparse_matrix=dataset.get_rating_matrix("train"),
        device=DEVICE,
        learning_rate=args.lr,
        lambda_=args.lambda_,
        topks=args.topks,
    )

    train_dataloader = torch.utils.data.DataLoader(
        UniformSamplingDataset(dataset.get_rating_matrix("train")),
        shuffle=False,
        batch_size=args.train_batch,
        num_workers=args.n_threads
    )

    # Special collate function to vertically stack tensors (1, n_items) to (batch_size, n_items)
    #    without it, the predicted output would be (batch_size, 1, n_items)
    def collate_fn(batch):
        # X is a list of two-element tuples; first element is a user_index (int) and second element is a dense tensor (1, n_items)
        user_indices, ground_true = torch.utils.data.default_collate(batch)
        return user_indices, ground_true[:, 0, :]
    # Data for evaluation are prepared using different method than data for training
    evaluation_dataloaders = {
        subset_name: torch.utils.data.DataLoader(
            SparseMatrixDataset(dataset.get_rating_matrix(subset_name), device=DEVICE),
            batch_size=args.eval_batch,
            shuffle=False,
            collate_fn=collate_fn
        )
        for subset_name in dataset.get_subset_names()
    }

    model.fit(
        epochs=args.epochs,
        train_data=train_dataloader,
        train_callbacks=[AverageMetricsCallback(writer=writer)],
        eval_data=evaluation_dataloaders,
        eval_callbacks=[AverageMetricsCallback(writer=writer)],
        print_interval=args.print_interval
    )
