import logging
import os

import torch
import numpy as np
import scipy

from utils.metrics import nDCG_at_ks, recall_at_ks, AverageMetricsCallback


class LightGCN(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        latent_dim: int,
        n_users: int,
        n_items: int,
        topks: list,
        training_sparse_matrix: scipy.sparse.csr_matrix,
        learning_rate: float,
        device: torch.device,
        lambda_: float,
    ):
        super(LightGCN, self).__init__()
        self.__n_layers = n_layers
        self.__n_users = n_users
        self.__n_items = n_items
        self.__lambda_ = lambda_
        self.__embedding_user = torch.nn.Embedding(num_embeddings=self.__n_users, embedding_dim=latent_dim, device=device)
        self.__embedding_item = torch.nn.Embedding(num_embeddings=self.__n_items, embedding_dim=latent_dim, device=device)
        # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
        torch.nn.init.normal_(self.__embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.__embedding_item.weight, std=0.1)
        # Sparse COO tensor with shape (n_users + n_items, n_users + n_items)
        # described in "Neural Graph Collaborative Filtering" paper (Equation 8)
        # containing only training data
        self.__laplacian_matrix = LightGCN.__get_laplacian_matrix(training_sparse_matrix).to(device)
        self.__optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.__device = device
        self.__reset_embedding_cache()
        self.__topks = topks
        self.to(device)
        logging.info("LightGCN model was successfully initialized")

    def fit(
        self,
        epochs: int,
        train_data: torch.utils.data.DataLoader,
        train_callbacks: list = [],
        eval_data: dict = {},
        eval_callbacks: list = [],
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
                callback.start_epoch(current_epoch, epochs, total_steps, subset_name="train")
            for step, (user_indices, positive_item_indices, negative_item_indices) in enumerate(train_data, start=1):
                self.train_step(user_indices, positive_item_indices, negative_item_indices, callbacks=train_callbacks)
                if (print_interval is not None and step % print_interval == 0) or step == total_steps:
                    for callback in train_callbacks:
                        if isinstance(callback, AverageMetricsCallback):
                            callback.print_metrics()

            if savedir is not None:
                torch.save(
                    self.state_dict(),
                    os.path.join(savedir, f"epoch-{current_epoch}.pth"),
                )
            for callback in train_callbacks:
                callback.end_epoch()

            for data_name, data_to_evaluate in eval_data.items():
                for callback in eval_callbacks:
                    callback.start_epoch(current_epoch, epochs, len(data_to_evaluate), data_name)
                self.eval_data(data_to_evaluate, topks=self.__topks, callbacks=eval_callbacks)
                for callback in eval_callbacks:
                    callback.end_epoch()

        logging.info("************************** [END] **************************")

    def train_step(self, user_indices: torch.Tensor, positive_item_indices: torch.Tensor, negative_item_indices: torch.Tensor, callbacks: list):
        self.train()
        self.zero_grad()
        user_indices = user_indices.to(self.__device)
        positive_item_indices = positive_item_indices.to(self.__device)
        negative_item_indices = negative_item_indices.to(self.__device)
        bpr_loss, reg_loss = self.__bpr_loss(user_indices, positive_item_indices, negative_item_indices)
        loss = bpr_loss + reg_loss * self.__lambda_
        for callback in callbacks:
            if isinstance(callback, AverageMetricsCallback):
                metrics = {"bpr_loss": bpr_loss, "reg_loss": reg_loss, "loss": loss, AverageMetricsCallback.WEIGHTS_ATTR: len(user_indices)}
                callback.append_metrics(**metrics)
            else:
                raise NotImplementedError(f"Unsupported callback with type {type(callback)}")
        loss.backward()
        self.__optimizer.step()
        self.__reset_embedding_cache()
        torch.cuda.empty_cache()

    # For evaluation only
    def __predict_users_rating(self, user_indices: torch.Tensor) -> torch.Tensor:
        users_emb_all, items_emb = self.__compute_embeddings()
        users_emb = users_emb_all[user_indices]
        predicted_ratings = torch.sigmoid(users_emb @ items_emb.T)
        return predicted_ratings

    def eval_data(self, dataloader: torch.utils.data.DataLoader, topks: list, callbacks: list):
        self.eval()
        max_k = min(max(topks), self.__n_items)
        with torch.no_grad():
            for step_index, (user_indices, ground_true) in enumerate(dataloader, start=1):
                user_indices = user_indices.to(self.__device)
                ground_true = ground_true.to(self.__device)
                predicted_ratings = self.__predict_users_rating(user_indices)
                top_max_k_indices = torch.topk(predicted_ratings, k=max_k, axis=-1).indices
                hits_max_k = torch.gather(ground_true, 1, top_max_k_indices)
                for callback in callbacks:
                    if isinstance(callback, AverageMetricsCallback):
                        ndcg = {f"NDCG@{k}": metric for k, metric in zip(topks, nDCG_at_ks(topks, ground_true, hits=hits_max_k).mean(axis=1))}
                        recall = {f"Recall@{k}": metric for k, metric in zip(topks, recall_at_ks(topks, ground_true, hits=hits_max_k).mean(axis=1))}
                        metrics = {**ndcg, **recall, AverageMetricsCallback.WEIGHTS_ATTR: len(user_indices)}
                        callback.append_metrics(**metrics)
                    else:
                        raise NotImplementedError(f"Unsupported callback with type {type(callback)}")

                torch.cuda.empty_cache()
        for callback in callbacks:
            if isinstance(callback, AverageMetricsCallback):
                callback.print_metrics()

    @staticmethod
    def __get_laplacian_matrix(rating_matrix: scipy.sparse.csr_matrix) -> torch.Tensor:
        # Create A (adj_mat)
        n_users, n_items = rating_matrix.shape
        rating_matrix = rating_matrix.astype(np.float32)
        empty_item_matrix = scipy.sparse.csr_matrix((n_items, n_items), dtype=np.float32)
        empty_user_matrix = scipy.sparse.csr_matrix((n_users, n_users), dtype=np.float32)
        adj_mat = scipy.sparse.vstack((scipy.sparse.hstack((empty_user_matrix, rating_matrix)), scipy.sparse.hstack((rating_matrix.T, empty_item_matrix))))
        # D is a diagonal degree matrix
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        diagonal_degree_matrix = scipy.sparse.diags(d_inv)
        # Compute L
        L = diagonal_degree_matrix.dot(adj_mat).dot(diagonal_degree_matrix)
        # Convert L (csr_matrix) to sparse COO tensor
        coo = L.tocoo()
        row = torch.from_numpy(coo.row)
        col = torch.from_numpy(coo.col)
        index = torch.vstack([row, col])
        data = torch.from_numpy(coo.data.astype(np.float32))
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def __reset_embedding_cache(self):
        self.__users_emb_cached = None
        self.__items_emb_cached = None
        torch.cuda.empty_cache()

    def __compute_embeddings(self):
        """
        Compute embeddings for all items and users
        """
        if self.__users_emb_cached is None:
            users_emb = self.__embedding_user.weight
            items_emb = self.__embedding_item.weight
            all_emb = torch.cat([users_emb, items_emb])
            embeddings_per_layer = [all_emb]
            logging.debug(f"all_emb: {all_emb} with dtype {all_emb.dtype}; laplacian_matrix: {self.__laplacian_matrix}")
            for layer in range(self.__n_layers):
                all_emb = torch.sparse.mm(self.__laplacian_matrix, all_emb)
                embeddings_per_layer.append(all_emb)
            embeddings_per_layer = torch.stack(embeddings_per_layer, dim=1)
            light_out = torch.mean(embeddings_per_layer, dim=1)
            users, items = torch.split(light_out, [self.__n_users, self.__n_items])
            self.__users_emb_cached = users
            self.__items_emb_cached = items
        return self.__users_emb_cached, self.__items_emb_cached

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
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss


class UniformSamplingDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_matrix: scipy.sparse.csr_matrix):
        super(UniformSamplingDataset, self).__init__()
        self.__csr_matrix = sparse_matrix
        self.__n_users, self.__n_items = sparse_matrix.shape
        self.__indices_to_sample = np.nonzero(self.__csr_matrix.sum(axis=1))[0]

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.__csr_matrix.nnz

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        """
        Randomly select a user, one of his interacted items as a positive item and one of his non-interacted items as a negative item
        """
        user_index = np.random.choice(self.__indices_to_sample)
        interacted_items = self.__csr_matrix[user_index].nonzero()[1].astype(np.int64)
        positive_item_index = np.random.choice(interacted_items)
        # TODO select negative item using more effective way and potentially infinite loop
        while True:
            negative_item_index = np.random.randint(0, self.__n_items)
            # this loop will run indefinitely if user interacted with all items
            if negative_item_index in interacted_items:
                continue
            else:
                break
        return user_index, positive_item_index, negative_item_index


# Copied from ELSA paper and modified to return only rows of a given sparse matrix with at least one interaction
# Along with slices of rows from sparse matrix, index of a user is also yielded. That's why the dimensions of the sparse
# matrix can contain empty rows.
class SparseMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_matrix: scipy.sparse.csr_matrix, device: torch.device, return_only_active_users: bool = True):
        super(SparseMatrixDataset, self).__init__()
        self.__csr_matrix = sparse_matrix
        self.__device = device
        self.__return_only_active_users = return_only_active_users
        if return_only_active_users:
            number_of_interacted_items_per_user = self.__csr_matrix.sum(axis=1)
            self.__indices_to_sample = np.nonzero(number_of_interacted_items_per_user)[0]
        else:
            self.__indices_to_sample = np.arange(self.__length)

    def __len__(self):
        return len(self.__indices_to_sample)

    def __getitem__(self, idx):
        """
        Extract a row of a sparse matrix converted to sparse coo tensor allocated on the CPU.
        To same memory bandwidth, it moves data in sparse format to given device (preferably GPU) and convert it to dense there
        """
        idx_to_user_index = self.__indices_to_sample[idx]
        scipy_coo = self.__csr_matrix[idx_to_user_index].tocoo()
        torch_coo = torch.sparse_coo_tensor(
            np.vstack([scipy_coo.row, scipy_coo.col]),
            scipy_coo.data.astype(np.float32),
            scipy_coo.shape,
        )
        return idx_to_user_index, torch_coo.to(self.__device).to_dense()
