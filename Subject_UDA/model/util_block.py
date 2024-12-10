import torch.nn as nn
import functools
import math
import torch
from torch.nn.parameter import Parameter
from utils.config import ModelConfig
import copy
import numpy as np
import torch.nn.functional as F

D_METHOD = ["mean", "std"]


class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, len(D_METHOD)))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm1d(channel)
        self.activation = nn.Sigmoid()
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _,  = x.size()
        """
              MEAN POOLING + STD POOLING
        """
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)

        return t

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None]  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)
        return g


class EEGModalityFusion(nn.Module):
    """
    Input: EEG CHANNELS  F4 F3 C4 C3 O1 O2

    Output: EEG FUSION DATA
    """
    def __init__(self, eeg_channel_num):
        super(EEGModalityFusion, self).__init__()
        self.eeg_num = eeg_channel_num
        self.linear1 = nn.Linear(512*eeg_channel_num, 512)

    def forward(self, x):
        batch = x.shape[0]
        f4 = x[:, 0, :]
        f3 = x[:, 1, :]
        c4 = x[:, 2, :]
        c3 = x[:, 3, :]
        o1 = x[:, 4, :]
        o2 = x[:, 5, :]
        x = torch.concat((f4, f3, c4, c3, o1, o2), dim=1)
        x = self.linear1(x).view(batch, 1, -1)

        return x


class EOGModalityFusion(nn.Module):
    """
    Input: EOG CHANNELS  E1 E2

    Output: EOG FUSION DATA
    """
    def __init__(self, eog_channel_num):
        super(EOGModalityFusion, self).__init__()
        self.eeg_num = eog_channel_num
        self.linear1 = nn.Linear(512*eog_channel_num, 512)

    def forward(self, x):
        batch = x.shape[0]
        e1 = x[:, 0, :]
        e2 = x[:, 1, :]

        x = torch.concat((e1, e2), dim=1)
        x = self.linear1(x).view(batch, 1, -1)

        return x


class EMGModalityFusion(nn.Module):
    """
    Input: EMG CHANNELS  CHIN12 CHIN32 LegL LegR

    Output: EMG FUSION DATA
    """
    def __init__(self, emg_channel_num):
        super(EMGModalityFusion, self).__init__()
        self.eeg_num = emg_channel_num
        self.linear1 = nn.Linear(512*emg_channel_num, 512)

    def forward(self, x):
        batch = x.shape[0]
        chin12 = x[:, 0, :]
        leg_l = x[:, 1, :]
        leg_r = x[:, 2, :]

        x = torch.concat((chin12, leg_l, leg_r), dim=1)
        return self.linear1(x).view(batch, 1, -1)


class AllModalityFusion(nn.Module):
    def __init__(self ):
        super(AllModalityFusion, self).__init__()
        self.linear = nn.Linear(1024, 512)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.linear(x))

        return x


def clones(module, N):
    """用于生成相同网络层的克隆函数, 它的参数module表示要克隆的目标网络层, N代表需要克隆的数量"""
    # 在函数中, 我们通过for循环对module进行N次深度拷贝, 使其每个module成为独立的层,
    # 然后将其放在nn.ModuleList类型的列表中存放.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_head):
        super(MultiHeadAttention, self).__init__()
        self.w_query = nn.Linear(input_size, input_size)
        self.w_key = nn.Linear(input_size, input_size)
        self.w_value = nn.Linear(input_size, input_size)
        self.num_head = num_head
        self.dense = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.input_size = input_size
        self.att_dropout = nn.Dropout(0.25)


    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        query = self.w_query(input_tensor)
        key = self.w_key(input_tensor)
        value = self.w_value(input_tensor)
        query = query.view(batch_size, -1, self.num_head, self.input_size//self.num_head).permute(0, 2, 1 ,3)
        key = key.view(batch_size, -1, self.num_head, self.input_size // self.num_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_head, self.input_size // self.num_head).permute(0, 2, 1, 3)
        attention_score = torch.matmul(query, key.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(input_tensor.shape[2])
        attention_prob = nn.Softmax(dim=1)(attention_score)
        attention_prob = self.att_dropout(attention_prob)
        context = torch.matmul(attention_prob, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.num_head*(self.input_size//self.num_head))
        hidden_state = self.dense(context)
        return hidden_state


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        """初始化函数有三个输入参数分别是d_model, d_ff,和dropout=0.1，第一个是线性层的输入维度也是第二个线性层的输出维度，
           因为我们希望输入通过前馈全连接层后输入和输出的维度不变. 第二个参数d_ff就是第二个线性层的输入维度和第一个线性层的输出维度.
           最后一个是dropout置0比率."""
        super(FeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.relu(self.linear1(x)))
        x = self.drop(self.relu(self.linear2(x)))
        return x


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a1 = nn.Parameter(torch.ones(features))
        self.b1 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    # 根据features的形状初始化两个参数张量a2，和b2，第一个初始化为1张量，
    # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数，
    # 因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，
    # 使其即能满足规范化要求，又能不改变针对目标的表征.最后使用nn.parameter封装，代表他们是模型的参数。

    def forward(self, x):
        """输入参数x代表来自上一层的输出"""
        # 在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致.
        # 接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果，
        # 最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进行乘法操作，加上位移参数b2.返回即可.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a1 * (x - mean) / (std + self.eps) + self.b1


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """它输入参数有两个, size以及dropout， size一般是都是词嵌入维度的大小，
           dropout本身是对模型结构中的节点数进行随机抑制的比率，
           又因为节点被抑制等效就是该节点的输出都是0，因此也可以把dropout看作是对输出矩阵的随机置0的比率.
        """
        super(SublayerConnection, self).__init__()
        # 实例化了规范化对象self.norm
        self.norm = LayerNorm(size)
        # 又使用nn中预定义的droupout实例化一个self.dropout对象.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """前向逻辑函数中, 接收上一个层或者子层的输入作为第一个参数，
           将该子层连接中的子层函数作为第二个参数"""

        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作，
        # 随机停止一些网络中神经元的作用，来防止过拟合. 最后还有一个add操作，
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出.
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """它的初始化函数参数有四个，分别是size，其实就是我们词嵌入维度的大小，它也将作为我们编码器层的大小,
           第二个self_attn，之后我们将传入多头自注意力子层实例化对象, 并且是自注意力机制,
           第三个是feed_froward, 之后我们将传入前馈全连接层实例化对象, 最后一个是置0比率dropout."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.pos_emb = PositionalEncoding(size)
        # 如图所示, 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 把size传入其中
        self.size = size

    def forward(self, x):
        """forward函数中有两个输入参数，x和mask，分别代表上一层的输出，和掩码张量mask."""
        # 里面就是按照结构图左侧的流程. 首先通过第一个子层连接结构，其中包含多头自注意力子层，
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层. 最后返回结果.
        x = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        x = self.sublayer[0](x, lambda x: self.self_attn(x))
        return self.sublayer[1](x, self.feed_forward)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe的维度是（5000，512）
        pe = torch.zeros(max_len, d_model)

        # position是一个5000行1列的tensor
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term是一个256长度的一维tensor
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        # 最终的pe是一个torch.Size([5000, 1, 512])的维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, layer_num, drop_out, n_head):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multi_attention = MultiHeadAttention(d_model, d_model, n_head)
        self.dropout_rate = drop_out
        self.feedforward = FeedForward(d_model, d_model*4, self.dropout_rate)
        self.encoder = EncoderLayer(d_model, self_attn=self.multi_attention, feed_forward=self.feedforward,
                                    dropout=self.dropout_rate)
        self.layer_num = layer_num

    def forward(self, x):
        for _ in range(self.layer_num):
            x = self.encoder(x)
        return x


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))

    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss