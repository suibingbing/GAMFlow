import argparse
import os

import torch
import yaml
from ignite.contrib import metrics

from gamflow import constants as const
from gamflow import dataset
from gamflow import fastflow
from gamflow import utils

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
import numpy as np
import random
torch.backends.cudnn.benchmark=True
device = torch.device('cuda:0')  # 0表示第0号显卡
torch.cuda.set_device(0)  # 设置当前设备为第0号显卡
listauc_image=[]
listindex=[]
listauc_piexl=[]
listacc=[]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    print("category:",args.category)
    listindex.append(args.category)
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    print("category:",args.category)
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    print("step:",config["flow_step"])
    listindex.append(config["flow_step"])
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda(device)
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        # if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
        #     print(
        #         "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
        #             epoch + 1, step + 1, loss_meter.val, loss_meter.avg
        #         )
        #     )

def get_detection_auroc(preds, mask):
    image_score = np.max(preds, axis=(1, 2, 3))
    label = np.max(mask, axis=(1, 2, 3))
    auroc = roc_auc_score(label, image_score)
    return auroc
def get_accuracy(preds, mask):
    preds = np.max(preds, axis=(1, 2, 3))
    mask = np.max(mask, axis=(1, 2, 3))
    # Convert model outputs to binary predictions (0 or 1) based on a threshold
    threshold = 0.5
    binary_preds = (preds > threshold).astype(int)
    
    # Compute accuracy by comparing binary predictions to true labels
    accuracy = (binary_preds == mask).mean()
    return accuracy

def eval_once(dataloader, model,epoch):
    print("epoch: {}".format(epoch))
    model.eval()
    A = []
    B = []
    auroc_metric = metrics.ROC_AUC()
    for data, targets in dataloader:
        data, targets = data.cuda(device), targets.cuda(device)
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        A.append(targets.cpu().numpy())
        B.append(outputs.cpu().numpy())
        outputs = outputs.flatten()
        targets = targets.flatten()
        list2=targets
        list2 =list(map(int,list2))
        targets=torch.tensor(list2)
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    GT = np.concatenate(A, axis=0)    
    pred = np.concatenate(B, axis=0) 
    auroc2=get_detection_auroc(pred,GT)
    print("AUROC-image: {}".format(auroc2))
    listauc_image.append(auroc2)
    print("AUROC-piexl: {}".format(auroc))
    listauc_piexl.append(auroc)
    acc= get_accuracy(pred,GT)
    # print("ACC: {}".format(acc))
    # listacc.append(acc)
    


def train(args):
    # os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    # checkpoint_dir = os.path.join(
    #     const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    # )
    # os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda(device)

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model, epoch)
        # if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #         },
        #         os.path.join(checkpoint_dir, "%d.pt" % epoch),
        #     )


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda(device)
    eval_once(test_dataloader, model)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
t1=max(listauc_image)
t2=max(listauc_piexl)
# t3=max(listacc)
a1=(listauc_image.index(t1)+1)*10
a2=(listauc_piexl.index(t2)+1)*10
print(listindex)
# a3=(listacc.index(t3)+1)*10
print("AUROC-image-best: {}".format(t1),"epoch: {}".format(a1))
print("AUROC-piexl-best: {}".format(t2),"epoch: {}".format(a2))
# print("ACC-best: {}".format(t3),"epoch: {}".format(a3))
