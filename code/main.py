import time
import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

import dataloader
import model
import utils
import procedure
from parse import parse_args

LOGGING_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)

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
utils.set_seed(args.seed)

if args.dataset in ["gowalla", "yelp2018", "amazon-book"]:
    path = os.path.join("data", args.dataset)
    dataset = dataloader.Loader(A_split=False, folds=args.a_fold, path=path, device=DEVICE)
elif args.dataset == "lastfm":
    dataset = dataloader.LastFM(device=DEVICE)
else:
    raise NotImplementedError(f"Unsupported dataset {args.dataset}")

rec_model = model.LightGCN(args.layers, args.latent_dim, A_split=False, dataset=dataset)
rec_model = rec_model.to(DEVICE)
bpr = utils.BPRLoss(rec_model, args.lambda_, args.lr)

weight_file = os.path.join(FILE_PATH, f"lgn-{args.dataset}-{args.layers}-{args.latent_dim}.pth.tar")

logging.info(f"Weights will be saved to {weight_file}")
if args.load_previous:
    try:
        logging.info(f"Flag for loading previous weights was trigged, loading model with weights from {weight_file}")
        rec_model.load_state_dict(torch.load(weight_file, map_location=torch.device("cpu")))
    except FileNotFoundError:
        logging.warning(f"File {weight_file} does not exist, model will be trained from beginning")

# init tensorboard
if args.tensorboard:
    path = os.path.join(BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "LGN")
    writer = SummaryWriter(path)
    logging.info(f"Tensorboard was inicitialized to path={path}")

else:
    writer = None
    logging.info("Tensorboard will not be used")

try:
    for epoch in range(1, args.epochs + 1):
        rec_model.train()
        results = procedure.BPR_train_original(dataset, args.train_batch, DEVICE, bpr, epoch, writer=writer)
        logging.info(f"[TRAIN] Epoch: {epoch}/{args.epochs}; {utils.results_to_progress_log(results)}")
        if epoch % 1 == 0:
            results = procedure.test(dataset, rec_model, args.topks, args.test_batch, DEVICE, args.n_threads, epoch, writer)
            logging.info(f"[TEST] Epoch: {epoch}/{args.epochs}; {utils.results_to_progress_log(results)}")
            if epoch == 1:
                assert round(results["Recall@20"], 4) == round(0.08642416, 4)
                assert round(results["Precision@20"], 4) == round(0.02703128, 4)
                assert round(results["NDCG@20"], 4) == round(0.07227541, 4)

            if epoch == 2:
                assert round(results["Recall@20"], 4) == round(0.09613762, 4)
                assert round(results["Precision@20"], 4) == round(0.03018119, 4)
                assert round(results["NDCG@20"], 4) == round(0.08025454, 4)

            if epoch == 3:
                assert round(results["Recall@20"], 4) == round(0.10085652, 4)
                assert round(results["Precision@20"], 4) == round(0.03160627, 4)
                assert round(results["NDCG@20"], 4) == round(0.08451248, 4)

            if epoch == 4:
                assert round(results["Recall@20"], 4) == round(0.10535001, 4)
                assert round(results["Precision@20"], 4) == round(0.0328103, 4)
                assert round(results["NDCG@20"], 4) == round(0.08842302, 4)

            if epoch == 5:
                assert round(results["Recall@20"], 4) == round(0.10873527, 4)
                assert round(results["Precision@20"], 4) == round(0.03369951, 4)
                assert round(results["NDCG@20"], 4) == round(0.0915894, 4)

            if epoch == 6:
                assert round(results["Recall@20"], 4) == round(0.11204279, 4)
                assert round(results["Precision@20"], 4) == round(0.0345636, 4)
                assert round(results["NDCG@20"], 4) == round(0.09433213, 4)

            if epoch == 7:
                assert round(results["Recall@20"], 4) == round(0.1144635, 4)
                assert round(results["Precision@20"], 4) == round(0.03518655, 4)
                assert round(results["NDCG@20"], 4) == round(0.09638503, 4)

            if epoch == 8:
                assert round(results["Recall@20"], 4) == round(0.1166778, 4)
                assert round(results["Precision@20"], 4) == round(0.03579945, 4)
                assert round(results["NDCG@20"], 4) == round(0.09808936, 4)

            if epoch == 9:
                assert round(results["Recall@20"], 4) == round(0.11843803, 4)
                assert round(results["Precision@20"], 4) == round(0.03622815, 4)
                assert round(results["NDCG@20"], 4) == round(0.09946748, 4)

            if epoch == 10:
                assert round(results["Recall@20"], 4) == round(0.12013993, 4)
                assert round(results["Precision@20"], 4) == round(0.03667024, 4)
                assert round(results["NDCG@20"], 4) == round(0.10055132, 4)

            if epoch == 11:
                assert round(results["Recall@20"], 4) == round(0.12134588, 4)
                assert round(results["Precision@20"], 4) == round(0.03699344, 4)
                assert round(results["NDCG@20"], 4) == round(0.10154486, 4)

            if epoch == 12:
                assert round(results["Recall@20"], 4) == round(0.12283394, 4)
                assert round(results["Precision@20"], 4) == round(0.03736352, 4)
                assert round(results["NDCG@20"], 4) == round(0.10243225, 4)

            if epoch == 13:
                assert round(results["Recall@20"], 4) == round(0.12388475, 4)
                assert round(results["Precision@20"], 4) == round(0.0376415, 4)
                assert round(results["NDCG@20"], 4) == round(0.10322438, 4)

            if epoch == 14:
                assert round(results["Recall@20"], 4) == round(0.1251952, 4)
                assert round(results["Precision@20"], 4) == round(0.03798814, 4)
                assert round(results["NDCG@20"], 4) == round(0.10429674, 4)

            if epoch == 15:
                assert round(results["Recall@20"], 4) == round(0.12649249, 4)
                assert round(results["Precision@20"], 4) == round(0.03834148, 4)
                assert round(results["NDCG@20"], 4) == round(0.10514867, 4)

            if epoch == 16:
                assert round(results["Recall@20"], 4) == round(0.12744638, 4)
                assert round(results["Precision@20"], 4) == round(0.03861612, 4)
                assert round(results["NDCG@20"], 4) == round(0.1058937, 4)

            if epoch == 17:
                assert round(results["Recall@20"], 4) == round(0.12834675, 4)
                assert round(results["Precision@20"], 4) == round(0.03886563, 4)
                assert round(results["NDCG@20"], 4) == round(0.10649789, 4)

            if epoch == 18:
                assert round(results["Recall@20"], 4) == round(0.12956465, 4)
                assert round(results["Precision@20"], 4) == round(0.03916203, 4)
                assert round(results["NDCG@20"], 4) == round(0.10737581, 4)

            if epoch == 19:
                assert round(results["Recall@20"], 4) == round(0.1303976, 4)
                assert round(results["Precision@20"], 4) == round(0.03941155, 4)
                assert round(results["NDCG@20"], 4) == round(0.10797482, 4)

        torch.save(rec_model.state_dict(), weight_file)
finally:
    if writer is not None:
        writer.close()
