import torch.nn as nn
import torch
from utils.config import ModelConfig


class FeatureExtractor_DSN(nn.Module):
    def __init__(self, args):
        self.ModelParam = ModelConfig(args["dataset"])
        self.drop = self.ModelParam.ConvDrop
        super(FeatureExtractor_DSN, self).__init__()
        self.eeg_fe1 = nn.Sequential(
            nn.Conv1d(self.ModelParam.EegNum, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(self.drop),

            nn.Conv1d(64, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.eeg_fe2 = nn.Sequential(
            nn.Conv1d(self.ModelParam.EegNum, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(self.drop),

            nn.Conv1d(64, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.eog_fe1 = nn.Sequential(
            nn.Conv1d(self.ModelParam.EogNum, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(self.drop),

            nn.Conv1d(64, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.eog_fe2 = nn.Sequential(
            nn.Conv1d(self.ModelParam.EogNum, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(self.drop),

            nn.Conv1d(64, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fusion = nn.Linear(256, 128)

    def forward(self, eeg, eog):

        batch = eeg.shape[0]//self.ModelParam.SeqLength
        eeg1 = self.eeg_fe1(eeg)
        eeg2 = self.eeg_fe2(eeg)
        eeg = torch.concat((eeg1, eeg2), dim=2)

        eog1 = self.eog_fe1(eog)
        eog2 = self.eog_fe2(eog)
        eog = torch.concat((eog1, eog2), dim=2)

        xx = torch.concat((eeg, eog), dim=2)

        eeg = self.avg(eeg).view(batch*self.ModelParam.SeqLength, 1, 128)
        eog = self.avg(eog).view(batch*self.ModelParam.SeqLength, 1, 128)

        x = self.fusion(torch.concat((eeg, eog), dim=2))

        x = x.view(batch, self.ModelParam.SeqLength, -1)

        return x


class FEBlock(nn.Module):
    def __init__(self, args):
        super(FEBlock, self).__init__()
        self.ModelParam = ModelConfig(args["dataset"])
        self.time = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(self.ModelParam.ConvDrop),

            nn.Conv1d(64, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=6, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 512, kernel_size=6, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.frequency = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(self.ModelParam.ConvDrop),

            nn.Conv1d(64, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=8, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 512, kernel_size=8, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        batch = x.shape[0]//self.ModelParam.SeqLength
        x1 = self.time(x)
        x2 = self.frequency(x)
        x = torch.concat((x1, x2), dim=2)
        x = self.avg(x).view(batch*self.ModelParam.SeqLength, 1, 512)
        return x


class GMAPB(nn.Module):
    def __init__(self):
        super(GMAPB, self).__init__()
        self.mp = nn.MaxPool1d(kernel_size=4, stride=4)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.maxp = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.mp(x)
        x1 = self.avg(x)
        x2 = self.maxp(x)
        x = torch.concat((x1, x2), dim=-1)
        x = self.drop(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = x.view(-1, 20, 256)
        return x


class FeatureExtractorRec(nn.Module):
    def __init__(self, args):
        super(FeatureExtractorRec, self).__init__()
        self.drop = 0.5
        self.args = args
        self.ModelParam = ModelConfig(args["dataset"])
        self.cnn_sequential = nn.Sequential(
            nn.Conv1d(self.ModelParam.EegNum + self.ModelParam.EogNum, 128, kernel_size=50, stride=6, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(self.drop)
        )

        self.cnn_sequential1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.cnn_sequential2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.rec_sequential = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.ConvTranspose1d(128, 128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.gmapb = GMAPB()
        self.lstm1 = nn.LSTM(256, 256, 1)
        self.mlp = nn.Linear(256, 5)

    def forward(self, eeg, eog):
        batch = eeg.shape[0] // self.ModelParam.SeqLength

        x = torch.concat((eeg, eog), dim=1)

        x = self.cnn_sequential(x)

        x = self.cnn_sequential1(x)

        origin_x = x

        x = self.cnn_sequential2(x)

        rec_x = self.rec_sequential(x)

        x = self.gmapb(x)
        if self.args["pretrain"]:
            return x, origin_x, rec_x
        else:
            return x


class FeatureExtractorTiny(nn.Module):
    def __init__(self, args):
        super(FeatureExtractorTiny, self).__init__()
        self.drop = 0.5
        self.args = args
        self.moderparam = ModelConfig(self.args["dataset"])
        self.cnn_sequential = nn.Sequential(
            nn.Conv1d(self.moderparam.EogNum + self.moderparam.EegNum, 128, kernel_size=50, stride=10, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(self.drop),

            nn.Conv1d(128, 128, kernel_size=8, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, eeg, eog):
        x = torch.concat((eeg, eog), dim=1)
        x = self.cnn_sequential(x)
        x = self.avg(x).view(-1, 20, 128)
        return x


class DeepSleepNetEncode(nn.Module):  # current one!
    def __init__(self):
        super(DeepSleepNetEncode, self).__init__()
        self.lstm = nn.LSTM(128, 512, batch_first=True, bidirectional=True, dropout=0.5, num_layers=2)
        self.res = nn.Linear(128, 1024)
        self.linear = nn.Linear(1024, 128)

    def forward(self, x):
        # batch, 20, 128
        res = self.res(x)
        x, _ = self.lstm(x)
        x = x + res
        x = self.linear(x)
        return x


class SleepMLPDSN(nn.Module):
    def __init__(self, args):
        super(SleepMLPDSN, self).__init__()
        self.ModelParam = ModelConfig(args["dataset"])
        self.dropout_rate = self.ModelParam.SleepMlpParamDSN.drop
        self.sleep_stage_mlp = nn.Sequential(
            nn.Linear(self.ModelParam.SleepMlpParamDSN.first_linear[0], self.ModelParam.SleepMlpParamDSN.first_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Linear(self.ModelParam.SleepMlpParamDSN.second_linear[0], self.ModelParam.SleepMlpParamDSN.second_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
        )
        self.sleep_stage_classifier = nn.Linear(self.ModelParam.SleepMlpParamDSN.out_linear[0],
                                                self.ModelParam.SleepMlpParamDSN.out_linear[1], bias=False)

    def forward(self, x):
        x = self.sleep_stage_mlp(x)
        x = self.sleep_stage_classifier(x)
        x = x.permute(0, 2, 1)
        return x


class RecSleepNetEncode(nn.Module):  # current one!
    def __init__(self):
        super(RecSleepNetEncode, self).__init__()
        self.lstm1 = nn.LSTM(256, 256, 1)

    def forward(self, x):
        # batch, 20, 128
        x, _ = self.lstm1(x)
        return x


class TinySleepNetEncode(nn.Module):
    def __init__(self, args):
        self.drop = 0.5
        super(TinySleepNetEncode, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.args = args
        self.modelparam = ModelConfig(self.args["dataset"])
        self.dmodel = self.modelparam.EncoderParamTiny.d_model
        self.lstm = nn.LSTM(self.dmodel, self.dmodel, bidirectional=False)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.drop(x)
        return x


class SleepMLPTiny(nn.Module):
    def __init__(self, args):
        super(SleepMLPTiny, self).__init__()
        self.ModelParam = ModelConfig(args["dataset"])
        self.dropout_rate = self.ModelParam.SleepMlpParamTiny.drop
        self.sleep_stage_mlp = nn.Sequential(
            nn.Linear(self.ModelParam.SleepMlpParamTiny.first_linear[0],
                      self.ModelParam.SleepMlpParamTiny.first_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Linear(self.ModelParam.SleepMlpParamTiny.second_linear[0],
                      self.ModelParam.SleepMlpParamTiny.second_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
        )
        self.sleep_stage_classifier = nn.Linear(self.ModelParam.SleepMlpParamTiny.out_linear[0],
                                                self.ModelParam.SleepMlpParamTiny.out_linear[1], bias=False)

    def forward(self, x):
        x = self.sleep_stage_mlp(x)
        x = self.sleep_stage_classifier(x)
        x = x.permute(0, 2, 1)
        return x


class SleepMLPRec(nn.Module):
    def __init__(self, args):
        super(SleepMLPRec, self).__init__()
        self.ModelParam = ModelConfig(args["dataset"])
        self.dropout_rate = self.ModelParam.SleepMlpParamRec.drop
        self.sleep_stage_mlp = nn.Sequential(
            nn.Linear(self.ModelParam.SleepMlpParamRec.first_linear[0], self.ModelParam.SleepMlpParamRec.first_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Linear(self.ModelParam.SleepMlpParamRec.second_linear[0], self.ModelParam.SleepMlpParamRec.second_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
        )
        self.sleep_stage_classifier = nn.Linear(self.ModelParam.SleepMlpParamRec.out_linear[0],
                                                self.ModelParam.SleepMlpParamRec.out_linear[1], bias=False)

    def forward(self, x):
        x = self.sleep_stage_mlp(x)
        x = self.sleep_stage_classifier(x)
        x = x.permute(0, 2, 1)
        return x