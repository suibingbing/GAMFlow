import argparse
import os
from sklearn import metrics as metricss
import torch
import yaml
from ignite.contrib import metrics 
import constants as const
import dataset
import fastflow
import utils
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import json
#torch.backends.cudnn.benchmark=True
listauc_image=[]
listauc_piexl=[]

dic_output={}
dic_target={}
#mps = torch.device('mps')
torch.cuda.is_available()
torch.backends.cudnn.benchmark=True

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
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
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
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )
# def save_roc(mydict,n):
#     # 1. json.dumps(字典)：将字典转为JSON字符串，indent为多行缩进空格数，
#     # sort_keys为是否按键排序,ensure_ascii=False为不确保ascii，及不将中文等特殊字符转为\uXXX等
#     if n==0:
#         with open("outputcbam3.json", "w", encoding='utf-8') as f:
#         # json.dump(dict_, f)  # 写为一行
#            json.dump(mydict, f)  # 写为多行
#     else:
#         with open("targetcbam3.json", "w", encoding='utf-8') as f:
#         # json.dump(dict_, f)  # 写为一行
#            json.dump(mydict, f)  # 写为多行
def get_detection_auroc(preds, mask):
    image_score = np.max(preds, axis=(1, 2, 3))
    label = np.max(mask, axis=(1, 2, 3))
    print(image_score,preds)
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

def eval_once(dataloader, model):
    model.eval()
    # auroc_metric = metrics.ROC_AUC()
    # auroc_metric2=mec.ROC_AUC()
    arr=np.array([])
    arr2=np.array([])
    A = []
    B = []
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        A.append(targets.cpu().numpy())
        B.append(outputs.cpu().numpy())
    GT = np.concatenate(A, axis=0)    
    pred = np.concatenate(B, axis=0)    
    print(get_detection_auroc(pred,GT))
    listac.append(get_detection_auroc(pred,GT))
        # for i in range(0,len(targets)):
        #     if torch.equal(targets[i][0],target2[0])==True:
        #         list1.append(1)
        #     else:
        #         list1.append(0)
        # for i in range(0,len(outputs)):
        #     if torch.equal(outputs[i][0],target3[0])==True:
        #         list2.append(1)
        #     else:
        #         list2.append(0)
        
        # outputs = outputs.flatten()
        # targets = targets.flatten()
        # list2=targets
        # list2 =list(map(int,list2))
        # targets=torch.tensor(list2)
        # output2 = outputs.numpy()
        # target2 = targets.cpu().numpy()
        # arr=np.append(arr,output2)
        # arr2=np.append(arr2,target2)
        #target2= targets.flatten()
        #targets=targets.cpu().numpy()
        #fpr, tpr, thresholds = metricss.roc_curve(targets, outputs,target2)
        #print(fpr,tpr,thresholds)
        #targets=targets.cpu().numpy()
        #fpr, tpr, thresholds = metricss.roc_curve(targets, outputs)
        # auroc_metric.update((outputs, targets))
        #print(fpr,tpr,thresholds)
    '''''
    'argon2:$argon2id$v=19$m=10240,t=10,p=8$umBHs/9mxRqY7APihw4Hkg$leerDiBZVB46qTJr6LjqlEdVeqxfgdP1i4TgXaOwGwk'
    print(list1)
    print(len(list1))
    print(list2)
    print(len(list2))
    '''
    # auroc = auroc_metric.compute()
    # print("AUROC: {}".format(auroc))
    # listac.append(auroc)
    # l=str(len(listac)*10)
    # list1 = arr.tolist()
    # list2 = arr2.tolist()
    # dic_target[l]=list2
    # dic_output[l]=list1
    #if max(listac)==auroc:
        #save_roc(dic_output,0)
        #save_roc(dic_target,1)



def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    #os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % 1== 0: # 10
            eval_once(test_dataloader, model)
            '''
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )
            '''
            
            


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
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
t=max(listac)
a=(listac.index(t)+1)
print(t,a)
#save_roc(dic_output,0)
#save_roc(dic_target,1)