import torch
import os
from utils.util import fix_randomness, print_ans
import numpy as np
from torch.utils.data import DataLoader
from dataloader.dataloader import Builder
from trainer.ssc_pretrain_deep import ssc_pretrain_deep
from trainer.ssc_pretrain_rec import ssc_pretrain_rec
from trainer.ssc_pretrain_tiny import ssc_pretrain_tiny
from trainer.trainer import trainer


def get_filepath(dataset):
    path = None
    if dataset == "ISRUC":
        path = ""

    elif dataset == "SleepEDF":
        path = ""

    elif dataset == "HMC":
        path = ""

    return path


def get_path_loader(parser):
    path = None
    if parser["dataset"] == "ISRUC":
        path = [i for i in range(1, 101) if i not in [8, 40]]
    elif parser["dataset"] == "SleepEDF":
        path = []
        for i in range(83):
            if i not in [39, 68, 69, 78, 79]:
                if i < 10:
                    path.append(f"0{i}")
                else:
                    path.append(f"{i}")
    elif parser["dataset"] == "HMC":
        path = [i for i in range(1, 148)]
    path_name = {int(j): [[], []] for j in path}

    parser["info"] = {int(j): [[], []] for j in path}

    parser["two_step_info"] = {int(j): [[], []] for j in path}

    for t_idx in path:
        """Load data files"""
        num = 0
        file_path = parser['filepath'] + f"/{t_idx}/data"
        label_path = parser['filepath'] + f"/{t_idx}/label"
        while os.path.exists(file_path + f"/{num}.npy"):
            path_name[t_idx][0].append(file_path + f"/{num}.npy")
            path_name[t_idx][1].append(label_path + f"/{num}.npy")
            num += 1
    lens = 0
    for i in path_name.keys():
        lens += len(path_name[i][0])*20
    print(lens)
    return path, path_name


def get_idx(parser, path):
    dataset = parser["dataset"]
    fix_randomness(parser["rand"])
    if dataset == "ISRUC":
        if parser["Fold"] == 1:
            test_idx = [1, 2, 3, 4, 5, 6, 7, 9, 10]
        elif parser["Fold"] == 4:
            test_idx = [31, 32, 33, 34, 35, 36, 37, 38, 39]
        else:
            test_idx = [(parser["Fold"] - 1) * 10 + j for j in range(1, 11)]

        idx = [i for i in range(1, 101) if i not in test_idx]
        idx.remove(8)
        idx.remove(40)

        val_idx = list(np.random.choice(idx, 10, replace=False))
        train_idx = [i for i in idx if i not in val_idx]

    elif dataset == "HMC":
        if parser["Fold"] == 10:
            test_idx = [136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147]
        else:
            ff = (parser["Fold"] - 1) * 15
            test_idx = [ff + j for j in range(1, 16)]
        idx = [i for i in range(1, 148) if i not in test_idx]
        val_idx = list(np.random.choice(idx, 14, replace=False))
        train_idx = [i for i in idx if i not in val_idx]
    elif dataset == 'SleepEDF':
        if parser["Fold"] == 1:
            test_idx = ['00', '01', '02', '03', '04', '05', '06', '07']
        elif parser["Fold"] == 2:
            test_idx = ['08', '09', '10', '11', '12', '13', '14', '15']
        elif parser["Fold"] == 3:
            test_idx = ['16', '17', '18', '19', '20', '21', '22']
        elif parser["Fold"] == 4:
            test_idx = ['23', '24', '25', '26', '27', '28', '29']
        elif parser["Fold"] == 5:
            test_idx = ['30', '31', '32', '33', '34', '35', '36', '37']
        elif parser["Fold"] == 6:
            test_idx = ['38', '40', '41', '42', '43', '44', '45', '46']
        elif parser["Fold"] == 7:
            test_idx = ['47', '48', '49', '50', '51', '52', '53', '54']
        elif parser["Fold"] == 8:
            test_idx = ['55', '56', '57', '58', '59', '60', '61', '62']
        elif parser["Fold"] == 9:
            test_idx = ['63', '64', '65', '66', '67', '70', '71', '72']
        else:
            test_idx = ['73', '74', '75', '76', '77', '80', '81', '82']
        idx = sorted(list(set(path) - set(test_idx)))
        val_idx = list(np.random.choice(idx, 8, replace=False))
        train_idx = [i for i in idx if i not in val_idx]
        train_idx = [str(i) for i in train_idx]
        val_idx = [str(i) for i in val_idx]
        test_idx = [str(i) for i in test_idx]

    return train_idx, val_idx, test_idx


def get_loader(parser, path, path_name):
    train_path = [[], []]
    val_path = [[], []]
    train_idx, val_idx, test_idx = get_idx(parser, path)

    for t_idx in train_idx:
        train_path[0].extend(path_name[t_idx][0])
        train_path[1].extend(path_name[t_idx][1])

    for v_idx in val_idx:
        val_path[0].extend(path_name[v_idx][0])
        val_path[1].extend(path_name[v_idx][1])

    train_builder = Builder(train_path, parser["dataset"]).Dataset
    val_builder = Builder(val_path, parser["dataset"]).Dataset

    return train_builder, val_builder, test_idx


def main():
    parser = dict()
    parser["epoch"] = 50
    parser["batch"] = 32
    parser["KFold"] = 10
    parser["rand"] = 1024
    parser["lr"] = 0.0001
    parser["ssl_lr"] = 1e-7
    parser["ft_lr"] = 1e-7
    parser["gpu"] = 1
    BASELINE = ['DeepSleepNet', 'TinySleepNet', 'RecSleepNet']
    parser["set"] = BASELINE[0]  # choose one baseline
    parser["dataset"] = "ISRUC"
    parser["ssl_epoch"] = 5
    parser["finetune_epoch"] = 10
    parser["filepath"] = get_filepath(parser["dataset"])
    parser['save_path'] = ''
    parser["optimizer"] = "AdamW"
    parser["device"] = torch.device(f"cuda:{parser['gpu']}" if torch.cuda.is_available() else "cpu")
    parser["beta"] = [0.5, 0.99]
    parser["weight_decay"] = 3e-4
    parser["num_worker"] = 4
    parser["print_p"] = True
    parser["pretrain"] = False
    parser["cross_acc"] = []
    parser["cross_f1"] = []
    parser["cross_mtx"] = []
    parser["Fold"] = 1

    for key in parser.keys():
        print(f"{key}:  {parser[key]}")

    fix_randomness(parser["rand"])
    torch.multiprocessing.set_start_method('spawn')
    path, path_name = get_path_loader(parser)
    for fold in range(1, 11):

        fix_randomness(fold)
        train_dataset, val_dataset, test_idx = get_loader(parser, path, path_name)
        # 加载数据集
        train_loader = DataLoader(dataset=train_dataset, batch_size=parser['batch'],
                                  shuffle=True, num_workers=parser["num_worker"])
        val_loader = DataLoader(dataset=val_dataset, batch_size=parser['batch'],
                                shuffle=True, num_workers=parser["num_worker"])

        if parser["pretrain"]:
            if parser["set"] == "DeepSleepNet":
                ssc_pretrain_deep(train_loader, val_loader, parser)
            if parser["set"] == "RecSleepNet":
                ssc_pretrain_rec(train_loader, val_loader, parser)
            if parser["set"] == "TinySleepNet":
                ssc_pretrain_tiny(train_loader, val_loader, parser)
        else:
            trainer(test_idx, parser)
        parser["Fold"] += 1

    print_ans(parser)


if __name__ == '__main__':
    main()

