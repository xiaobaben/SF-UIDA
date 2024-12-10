class ModelConfig(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.ConvDrop = 0.1
        self.EncoderParamDeep = EncoderConfigSmall()
        self.EncoderParamLarge = EncoderConfigLarge()
        self.EncoderParamRec = EncoderConfigRec()
        self.EncoderParamTiny = EncoderConfigTiny()
        self.SleepMlpParamDeep = SleepMlpParamSmall()
        self.SleepMlpParamLarge = SleepMlpParamLarge()
        self.SleepMlpParamDSN = SleepMlpParamDSN()
        self.SleepMlpParamRec = SleepMlpParamRec()
        self.SleepMlpParamTiny = SleepMlpParamTiny()
        self.TeacherModel = TeacherModel()
        self.NumClasses = 5
        self.ClassNames = ['W', 'N1', 'N2', 'N3', 'REM']
        self.SeqLength = 20
        self.cpc_step = 3
        self.BatchSize = 32
        self.EpochLength = 3000
        ans = self.get_channel_info()
        self.EegNum = ans[0]
        self.EogNum = ans[1]

    def get_channel_info(self):
        if self.dataset == "ISRUC":
            return [6, 2]
        elif self.dataset == "SleepEDF":
            return [2, 1]
        elif self.dataset == "HMC":
            return [4, 2]


class EncoderConfigSmall(object):
    def __init__(self):
        self.n_head = 8
        self.d_model = 128
        self.layer_num = 3
        self.drop = 0.1


class EncoderConfigRec(object):
    def __init__(self):
        self.n_head = 8
        self.d_model = 256
        self.layer_num = 3
        self.drop = 0.1


class EncoderConfigTiny(object):
    def __init__(self):
        self.n_head = 8
        self.d_model = 128
        self.layer_num = 3
        self.drop = 0.1

class EncoderConfigLarge(object):
    def __init__(self):
        self.n_head = 8
        self.d_model = 512
        self.layer_num = 3
        self.drop = 0.1


class SleepMlpParamSmall(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [128, 64]
        self.second_linear = [64, 32]
        self.out_linear = [32, 5]


class SleepMlpParamLarge(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 5]


class SleepMlpParamDSN(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [128, 64]
        self.second_linear = [64, 32]
        self.out_linear = [32, 5]


class SleepMlpParamRec(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [256, 128]
        self.second_linear = [128, 64]
        self.out_linear = [64, 5]


class SleepMlpParamTiny(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [128, 64]
        self.second_linear = [64, 32]
        self.out_linear = [32, 5]


class TeacherModel(object):
    def __init__(self):
        self.teacher_wt = 0.1
        self.confidence_level = 0.90
        self.momentum_wt = 0.996


