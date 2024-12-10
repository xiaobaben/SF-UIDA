import copy
import os
import torch
from model.model import FeatureExtractor_DSN, FeatureExtractorRec, FeatureExtractorTiny,\
    DeepSleepNetEncode, SleepMLPDSN, RecSleepNetEncode, SleepMLPRec, SleepMLPTiny, TinySleepNetEncode
from dataloader.dataloader import Builder
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from utils.config import ModelConfig
from model.algorithm import SCC
import numpy as np
import torch.nn as nn



def trainer(test_idx, args):
    for test_id in test_idx:
        if args["set"] == "DeepSleepNet":
            feature_extractor = FeatureExtractor_DSN(args)
            feature_encoder = DeepSleepNetEncode()
            sleep_classifier = SleepMLPDSN(args)

            feature_extractor_t = FeatureExtractor_DSN(args)
            feature_encoder_t = DeepSleepNetEncode()
            sleep_classifier_t = SleepMLPDSN(args)
        elif args["set"] == "RecSleepNet":
            feature_extractor = FeatureExtractorRec(args)
            feature_encoder = RecSleepNetEncode()
            sleep_classifier = SleepMLPRec(args)

            feature_extractor_t = FeatureExtractorRec(args)
            feature_encoder_t = RecSleepNetEncode()
            sleep_classifier_t = SleepMLPRec(args)

        elif args["set"] == "TinySleepNet":
            feature_extractor = FeatureExtractorTiny(args)
            feature_encoder = TinySleepNetEncode(args)
            sleep_classifier = SleepMLPTiny(args)

            feature_extractor_t = FeatureExtractorTiny(args)
            feature_encoder_t = TinySleepNetEncode(args)
            sleep_classifier_t = SleepMLPTiny(args)



        map_location = torch.device(f'cuda:{args["gpu"]}')
        feature_extractor.load_state_dict(
            torch.load(
                f"{args['save_path']}/{args['set']}/feature_extractor_parameter_{args['rand']}.pkl", map_location=map_location))
        feature_encoder.load_state_dict(
            torch.load(
                f"{args['save_path']}/{args['set']}/feature_encoder_parameter_{args['rand']}.pkl", map_location=map_location))
        sleep_classifier.load_state_dict(
            torch.load(
                f"{args['save_path']}/{args['set']}/sleep_classifier_parameter_{args['rand']}.pkl", map_location=map_location))

        feature_extractor_t.load_state_dict(
            torch.load(
                f"{args['save_path']}/{args['set']}/feature_extractor_parameter_{args['rand']}.pkl", map_location=map_location))
        feature_encoder_t.load_state_dict(
            torch.load(
                f"{args['save_path']}/{args['set']}/feature_encoder_parameter_{args['rand']}.pkl", map_location=map_location))
        sleep_classifier_t.load_state_dict(
            torch.load(
                f"{args['save_path']}/{args['set']}/sleep_classifier_parameter_{args['rand']}.pkl", map_location=map_location))
    blocks = (feature_extractor, feature_encoder, sleep_classifier)
    teacher_block = (feature_extractor_t, feature_encoder_t, sleep_classifier_t)

    sf_uida_train(test_id, blocks, teacher_block, args)


def get_test_loader(args, test_idx):
    test_path = [[], []]
    file_path = args['filepath'] + f"/{test_idx}/data"
    label_path = args['filepath'] + f"/{test_idx}/label"
    num = 0
    while os.path.exists(file_path + f"/{num}.npy"):
        test_path[0].append(file_path + f"/{num}.npy")
        test_path[1].append(label_path + f"/{num}.npy")
        num += 1

    test_builder = Builder(test_path, args["dataset"]).Dataset
    test_loader = DataLoader(dataset=test_builder, batch_size=args["batch"], shuffle=True, num_workers=4)

    return test_loader


def evaluator(model, args, test_loader):
    y_pred = []
    y_test = []

    device = args["device"]
    model[0].to(device)
    model[1].to(device)
    model[2].to(device)

    model[0].eval()
    model[1].eval()
    model[2].eval()

    model_param = ModelConfig(args["dataset"])

    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch_idx, data in enumerate(test_loader):
            eog, eeg, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            epoch_size = model_param.EpochLength

            eog = eog.view(-1, model_param.EogNum, epoch_size)
            eeg = eeg.view(-1, model_param.EegNum, epoch_size)

            eeg_eog_feature = model[0](eeg, eog)

            # EEG + EOG
            eeg_eog_feature = model[1](eeg_eog_feature)  # batch, 20, 512
            prediction = model[2](eeg_eog_feature)

            _, predicted = torch.max(prediction.data, dim=1)

            predicted, labels = torch.flatten(predicted), torch.flatten(labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            p_acc = correct / total
            predicted = predicted.tolist()
            y_pred.extend(predicted)
            labels = labels.tolist()
            y_test.extend(labels)

        p_macro_f1 = f1_score(y_test, y_pred, average="macro")
        return tuple((p_acc, p_macro_f1, correct, total))


def sf_uida_train(test_id, blocks, teach_blocks, args):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    test_loader = get_test_loader(args, test_id)
    before_report = evaluator(blocks, args, test_loader)

    args["info"][int(test_id)][0].append(before_report[0])
    args["info"][int(test_id)][1].append(before_report[1])


    algorithm = SCC(blocks, args)


    device = args["device"]
    check_point = [2 * (i + 1) for i in range(args["ssl_epoch"] // 2)]

    model_param = ModelConfig(args["dataset"])
    for epoch in range(1, args["ssl_epoch"] + 1):
        blocks[0].train()
        blocks[1].train()
        blocks[2].train()

        epoch_loss = []
        for batch_idx, data in enumerate(test_loader):
            eog, eeg, label = data[0].to(device), data[1].to(device), data[2].to(device)
            loss, tmp_blocks = algorithm.update(eeg, eog)
            epoch_loss.append(loss)
        print(f"TestID {test_id}  Epoch {epoch}  SSL Loss {np.mean(epoch_loss)}")

        if epoch in check_point:
            tmp_report = evaluator(tmp_blocks, args, test_loader)
            args["info"][int(test_id)][0].append(tmp_report[0])
            args["info"][int(test_id)][1].append(tmp_report[1])

    for ep in range(len(check_point) + 1):
        print(f"Test id:{test_id}  SSL-Epoch:{ep * 2}  "
              f"ACC:{args['info'][int(test_id)][0][ep]}  "
              f"MF1:{args['info'][int(test_id)][1][ep]}")

    args["two_step_info"][int(test_id)][0].append(args['info'][int(test_id)][0][-1])
    args["two_step_info"][int(test_id)][1].append(args['info'][int(test_id)][1][-1])

    optimizer = torch.optim.Adam([
                                  {"params": list(tmp_blocks[0].parameters())},
                                  {"params": list(tmp_blocks[1].parameters())},
                                  {"params": list(tmp_blocks[2].parameters())}],
                                 lr=args["ft_lr"], betas=(args['beta'][0], args['beta'][1]),
                                 weight_decay=args['weight_decay'])
    cross_entropy = nn.CrossEntropyLoss()

    softmax = nn.Softmax(dim=1)
    check_point_finetune = [2 * (i + 1) for i in range(args["finetune_epoch"] // 2)]
    for epoch in range(1, args["finetune_epoch"] + 1):
        starter.record()
        tmp_blocks[0].train()
        tmp_blocks[1].train()
        tmp_blocks[2].train()

        teach_blocks[0].train()
        teach_blocks[1].train()
        teach_blocks[2].train()

        teach_blocks[0].to(device)
        teach_blocks[1].to(device)
        teach_blocks[2].to(device)
        epoch_loss2 = []

        for batch_idx, data in enumerate(test_loader):
            eog, eeg, label = data[0].to(device), data[1].to(device), data[2].to(device)
            optimizer.zero_grad()

            eog = eog.view(-1, model_param.EogNum, 3000)
            eeg = eeg.view(-1, model_param.EegNum, 3000)

            """Sequence Confidence"""

            with torch.no_grad():
                ff = teach_blocks[0](eeg, eog)
                ff = teach_blocks[1](ff)
                mean_t_pred = teach_blocks[2](ff)
                mean_t_preds = softmax(mean_t_pred)
                pred_prob = mean_t_preds.max(1, keepdim=True)[0].squeeze()  # batch, 20
                target_pseudo_labels = mean_t_preds.max(1, keepdim=True)[1].squeeze()  # batch, 20
            target = tmp_blocks[0](eeg, eog)
            target = tmp_blocks[1](target)
            pred_target = tmp_blocks[2](target)
            flag = True
            pred_prob = pred_prob.cpu().numpy()
            pred_prob = np.reshape(pred_prob, (-1, 20))
            target_pseudo_labels = target_pseudo_labels.view(-1, 20)
            for bh in range(pred_prob.shape[0]):  # batch, 20
                confident_epoch_num_per_seq = np.sum(pred_prob[bh, :] >= 0.8)
                if confident_epoch_num_per_seq >= 15:
                    confident_label = target_pseudo_labels[bh, :].view(1, 20)
                    confident_pred = pred_target[bh, :, :].view(1, 5, 20)
                    if flag:
                        confident_labels = confident_label
                        confident_preds = confident_pred
                        flag = False
                    else:
                        confident_labels = torch.concat((confident_labels, confident_label), dim=0)
                        confident_preds = torch.concat((confident_preds, confident_pred), dim=0)
                    break
            if not flag:
                loss = cross_entropy(confident_preds, confident_labels.long())
                loss.backward()
                optimizer.step()
                epoch_loss2.append(loss.item())
            else:
                """If there not exist the corresponding confident sequence"""
                with torch.no_grad():
                    mean_t_pred = mean_t_pred.permute(0, 2, 1)
                    mean_t_pred = mean_t_pred.view(-1, 5)
                    mean_t_pred = softmax(mean_t_pred)  # 640, 5
                    pred_prob = mean_t_pred.max(1, keepdim=True)[0].squeeze()
                    target_pseudo_labels = mean_t_pred.max(1, keepdim=True)[1].squeeze()
                pred_target = pred_target.permute(0, 2, 1)
                pred_target = pred_target.view(-1, 5)
                confident_pred = pred_target[pred_prob > model_param.TeacherModel.confidence_level]
                confident_labels = target_pseudo_labels[pred_prob > model_param.TeacherModel.confidence_level]
                loss = cross_entropy(confident_pred, confident_labels.long())
                loss.backward()
                optimizer.step()
                epoch_loss2.append(loss.item())

            alpha = model_param.TeacherModel.momentum_wt
            for mean_param, param in zip(teach_blocks[0].parameters(), tmp_blocks[0].parameters()):
                mean_param.data.mul_(alpha).add_(param.data * (1 - alpha))

            for mean_param, param in zip(teach_blocks[1].parameters(), tmp_blocks[1].parameters()):
                mean_param.data.mul_(alpha).add_(param.data * (1 - alpha))

            for mean_param, param in zip(teach_blocks[2].parameters(), tmp_blocks[2].parameters()):
                mean_param.data.mul_(alpha).add_(param.data * (1 - alpha))

        print(f"TestID {test_id}  Epoch {epoch}  FineTune Loss {np.mean(epoch_loss2)}")

        if epoch in check_point_finetune:
            tmp_report = evaluator(tmp_blocks, args, test_loader)
            args["two_step_info"][int(test_id)][0].append(tmp_report[0])
            args["two_step_info"][int(test_id)][1].append(tmp_report[1])

    print(f"Test id:{test_id}  After {args['ssl_epoch']} Epoch SSL "
          f"ACC:{args['info'][int(test_id)][0][-1]}"
          f"MF1:{args['info'][int(test_id)][1][-1]}")

    for ep in range(len(check_point_finetune) + 1):
        print(f"Test id:{test_id}  FineTune-Epoch:{ep * 2}  "
              f"ACC:{args['two_step_info'][int(test_id)][0][ep]}  "
              f"MF1:{args['two_step_info'][int(test_id)][1][ep]}")
