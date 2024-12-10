import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.config import ModelConfig
from model.util_block import MultiHeadAttentionBlock, mmd_rbf


class SCC(object):
    def __init__(self, blocks, args):
        super(SCC, self).__init__()
        self.args = args
        self.feature_extractor = blocks[0]
        self.feature_encoder = blocks[1]
        self.classifier = blocks[2]
        self.model_param = ModelConfig(args["dataset"])
        if self.args["set"] == "DeepSleepNet":
            self.num_channels = self.model_param.EncoderParamDeep.d_model
            self.d_model = self.model_param.EncoderParamDeep.d_model
        elif self.args["set"] == "RecSleepNet":
            self.num_channels = self.model_param.EncoderParamRec.d_model
            self.d_model = self.model_param.EncoderParamRec.d_model
        elif self.args["set"] == "TinySleepNet":
            self.num_channels = self.model_param.EncoderParamTiny.d_model
            self.d_model = self.model_param.EncoderParamTiny.d_model
        self.timestep = self.model_param.cpc_step
        self.device = args["device"]
        self.Wk = nn.ModuleList([nn.Sequential(nn.Linear(self.d_model, self.d_model * 4),
                                               nn.Dropout(0.1),
                                               nn.GELU(),
                                               nn.Linear(self.d_model * 4, self.d_model)).to(self.device)
                                 for _ in range(self.timestep)])

        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.encoder = MultiHeadAttentionBlock(self.d_model,
                                               self.model_param.EncoderParamDeep.layer_num,
                                               self.model_param.EncoderParamDeep.drop,
                                               self.model_param.EncoderParamDeep.n_head).to(self.device)

        self.optimizer = torch.optim.Adam([{"params": list(self.feature_extractor.parameters())},
                                           {"params": list(self.feature_encoder.parameters())},
                                           {"params": list(self.encoder.parameters()), "lr": self.args["lr"]},
                                           {"params": list(self.Wk.parameters()), "lr": self.args["lr"]}],
                                          lr=self.args["ssl_lr"], betas=(self.args['beta'][0], self.args['beta'][1]),
                                          weight_decay=self.args['weight_decay'])

    def update(self, eeg, eog):
        # ====== Data =====================
        seq_len = self.model_param.SeqLength
        batch = eeg.shape[0]

        epoch_size = self.model_param.EpochLength

        eog = eog.view(-1, self.model_param.EogNum, epoch_size)
        eeg = eeg.view(-1, self.model_param.EegNum, epoch_size)

        eeg_aug1, eog_aug1 = eeg, eog
        eeg_aug2, eog_aug2 = torch.flip(eeg, dims=[1]), torch.flip(eog, dims=[1])

        # EEG + EOG
        eeg_eog_feature_aug1 = self.feature_extractor(eeg_aug1, eog_aug1)
        eeg_eog_feature_aug1 = self.feature_encoder(eeg_eog_feature_aug1)

        eeg_eog_feature_aug2 = self.feature_extractor(eeg_aug2, eog_aug2)
        eeg_eog_feature_aug2 = self.feature_encoder(eeg_eog_feature_aug2)

        self.optimizer.zero_grad()

        # normalize projection feature vectors
        features1 = F.normalize(eeg_eog_feature_aug1, dim=2)
        features2 = F.normalize(eeg_eog_feature_aug2, dim=2)

        """AUG1"""
        encode_samples1 = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples1[i - 1] = features2[:, i-1, :].view(batch, self.num_channels)
        forward_seq1 = features1[:, :17, :]

        output = self.encoder(forward_seq1)  # batch, 15, 128
        c_t = output[:, 16, :].view(batch, -1)  # batch, 128

        pred1 = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)  # 5, batch, 128
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred1[i] = linear(c_t)  # batch, 128

        """AUG2"""
        encode_samples2 = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples2[i - 1] = features1[:, 16 + i, :].view(batch, self.num_channels)
        forward_seq2 = features2[:, self.timestep:, :]

        output = self.encoder(forward_seq2)  # batch, 15, 128

        c_t = output[:, seq_len - self.timestep - 1, :].view(batch, -1)  # batch, 128

        pred2 = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)  # 5, batch, 128
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred2[i] = linear(c_t)  # batch, 128

        p_sequence = pred2
        p_sequence.permute(1, 0, 2)

        n_sequence = pred1
        n_sequence.permute(1, 0, 2)
        n_sequence = torch.flip(n_sequence, dims=[1])

        mmd_loss = mmd_rbf(p_sequence, n_sequence)

        self.optimizer.zero_grad()
        loss = mmd_loss
        loss.backward()

        self.optimizer.step()

        return loss.item(), [self.feature_extractor, self.feature_encoder, self.classifier]

