import torch
import numpy as np
import random


def set_requires_grad(model, requires_grad=True):
    """
    :param model: Instance of Part of Net
    :param requires_grad: Whether Need Gradient
    :return:
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def fix_randomness(SEED):
    """
    :param SEED:  Random SEED
    :return:
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_ans(parser):
    if parser["pretrain"]:
        print("**********Cross   ACC*******************")
        print(parser["cross_acc"])
        print("**********Cross   F1********************")
        print(parser["cross_f1"])

        print("Best Sleep Result")
        print(f"Mean Acc:", sum(parser["cross_acc"]) / len(parser["cross_acc"]),
              "Mean Macro F1:", sum(parser["cross_f1"]) / len(parser["cross_f1"]))
    else:

        print(parser["info"])
        print(parser["two_step_info"])

        be_acc = [parser["info"][idx][0][0] for idx in parser["info"].keys()]
        be_mf1 = [parser["info"][idx][1][0] for idx in parser["info"].keys()]

        print("=====================================")

        print(f"Before Mean ACC{np.mean(be_acc)}")
        print(f"Before Mean MF1{np.mean(be_mf1)}")

        ssl_acc = [parser["info"][idx][0][-1] for idx in parser["info"].keys()]
        ssl_mf1 = [parser["info"][idx][1][-1] for idx in parser["info"].keys()]

        print(f"After {parser['ssl_epoch']} Epoch SSL ACC{np.mean(ssl_acc)}")
        print(f"After {parser['ssl_epoch']} Epoch SSL MF1{np.mean(ssl_mf1)}")

        for cp in range(1, len(parser["two_step_info"][1][0])):
            tmp_acc = [parser["two_step_info"][idx][0][cp] for idx in parser["two_step_info"].keys()]
            tmp_mf1 = [parser["two_step_info"][idx][1][cp] for idx in parser["two_step_info"].keys()]
            print(f"=========={2 * cp} Epoch Finetune=================")
            print(f"{2 * cp} Epoch Finetune Mean ACC{np.mean(tmp_acc)}")
            print(f"{2 * cp} Epoch Finetune Mean MF1{np.mean(tmp_mf1)}")
