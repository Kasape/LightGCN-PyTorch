import time
from os.path import join
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

import dataloader
import model
import utils
import procedure
import world

LOGGING_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
# ==============================
logging.info(f"Setting a random seed to: {world.SEED}")
utils.set_seed(world.SEED)
# ==============================

if world.DATASET in ["gowalla", "yelp2018", "amazon-book"]:
    dataset = dataloader.Loader(path="../data/" + world.DATASET)
elif world.DATASET == "lastfm":
    dataset = dataloader.LastFM()

logging.info("===========config================")
for key, val in world.CONFIG.items():
    logging.info(f"{key}: {val}")
logging.info(f"Metrics will be measured for following @K: {world.TOPKS}")
logging.info("===========end===================")


rec_model = model.LightGCN(world.CONFIG, dataset)
rec_model = rec_model.to(world.DEVICE)
bpr = utils.BPRLoss(rec_model, world.CONFIG)

weight_file = utils.getFileName()
logging.info(f"Weights will be saved to {weight_file}")
if world.LOAD_PREVIOUS:
    try:
        logging.info(f"Flag for loading previous weights was trigged, loading model with weights from {weight_file}")
        rec_model.load_state_dict(torch.load(weight_file, map_location=torch.device("cpu")))
    except FileNotFoundError:
        logging.warning(f"File {weight_file} does not exist, model will be trained from beginning")

# init tensorboard
if world.TENSORBOARD:
    path = join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "LGN")
    writer = SummaryWriter(path)
    logging.info(f"Tensorboard was inicitialized to path={path}")

else:
    writer = None
    logging.info("Tensorboard will not be used")

try:
    for epoch in range(1, world.TRAIN_EPOCHS + 1):
        rec_model.train()
        results = procedure.BPR_train_original(dataset, bpr, epoch, w=writer)
        logging.info(f"[TRAIN] Epoch: {epoch}/{world.TRAIN_EPOCHS}; {utils.results_to_progress_log(results)}")
        if epoch % 1 == 0:
            results = procedure.test(dataset, rec_model, epoch, writer)
            logging.info(f"[TEST] Epoch: {epoch}/{world.TRAIN_EPOCHS}; {utils.results_to_progress_log(results)}")
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
    if world.TENSORBOARD:
        writer.close()
