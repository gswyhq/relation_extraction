#!/usr/bin/python3
# coding: utf-8
import math
import os
import time
from typing import Union, List
import logging
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from config import MODEL_PATH

logger = logging.getLogger(__name__)


class PCNN(nn.Module):
    def __init__(self, vocab_size):
        super(PCNN, self).__init__()

        self.embedding = Embedding(vocab_size)
        self.cnn = CNN()
        self.fc1 = nn.Linear(len([3, 5, 7]) * 100, 80)
        self.fc2 = nn.Linear(80, 11)
        self.dropout = nn.Dropout(0.3)


    def load(self, path, device):
        '''
        加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path, map_location=device))


    def save(self, epoch=0):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S')
        prefix = MODEL_PATH
        os.makedirs(prefix, exist_ok=True)
        name = os.path.join(prefix, 'cnn_' + time_prefix + '_' + f'epoch{epoch}' + '.pth')

        torch.save(self.state_dict(), name)
        return name

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        mask = seq_len_to_mask(lens)

        inputs = self.embedding(word, head_pos, tail_pos)
        out, out_pool = self.cnn(inputs, mask=mask)

        output = self.fc1(out_pool)
        output = F.leaky_relu(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output


class Embedding(nn.Module):
    def __init__(self, vocab_size):
        """
        word embedding: 一般 0 为 padding
        pos embedding:  一般 0 为 padding
        dim_strategy: [cat, sum]  多个 embedding 是拼接还是相加
        """
        super(Embedding, self).__init__()

        self.vocab_size = vocab_size
        self.word_dim = 60
        self.pos_size = 62
        self.pos_dim = 60
        self.dim_strategy = 'sum'

        self.wordEmbed = nn.Embedding(self.vocab_size, self.word_dim, padding_idx=0)
        self.headPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)
        self.tailPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)

    def forward(self, *x):
        word, head, tail = x
        word_embedding = self.wordEmbed(word)
        head_embedding = self.headPosEmbed(head)
        tail_embedding = self.tailPosEmbed(tail)

        if self.dim_strategy == 'cat':
            return torch.cat((word_embedding, head_embedding, tail_embedding), -1)
        elif self.dim_strategy == 'sum':
            # 此时 pos_dim == word_dim
            return word_embedding + head_embedding + tail_embedding
        else:
            raise Exception('dim_strategy must choose from [sum, cat]')


class CNN(nn.Module):
    """
    nlp 里为了保证输出的句长 = 输入的句长，一般使用奇数 kernel_size，如 [3, 5, 7, 9]
    当然也可以不等长输出，keep_length 设为 False
    此时，padding = k // 2
    stride 一般为 1
    """
    def __init__(self):
        """
        in_channels      : 一般就是 word embedding 的维度，或者 hidden size 的维度
        out_channels     : int
        kernel_sizes     : list 为了保证输出长度=输入长度，必须为奇数: 3, 5, 7...
        activation       : [relu, lrelu, prelu, selu, celu, gelu, sigmoid, tanh]
        pooling_strategy : [max, avg, cls]
        dropout:         : float
        """
        super(CNN, self).__init__()

        self.in_channels = 60
        self.out_channels = 100
        self.kernel_sizes = [3, 5, 7]
        self.activation = 'gelu'
        self.pooling_strategy = 'max'
        self.dropout = 0.3
        self.keep_length = True
        for kernel_size in self.kernel_sizes:
            assert kernel_size % 2 == 1, "kernel size has to be odd numbers."

        # convolution
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      stride=1,
                      padding=k // 2 if self.keep_length else 0,
                      dilation=1,
                      groups=1,
                      bias=False) for k in self.kernel_sizes
        ])

        # activation function
        assert self.activation in ['relu', 'lrelu', 'prelu', 'selu', 'celu', 'gelu', 'sigmoid', 'tanh'], \
            'activation function must choose from [relu, lrelu, prelu, selu, celu, gelu, sigmoid, tanh]'
        self.activations = nn.ModuleDict([
            ['relu', nn.ReLU()],
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()],
            ['selu', nn.SELU()],
            ['celu', nn.CELU()],
            ['gelu', GELU()],
            ['sigmoid', nn.Sigmoid()],
            ['tanh', nn.Tanh()],
        ])

        # pooling
        assert self.pooling_strategy in ['max', 'avg', 'cls'], 'pooling strategy must choose from [max, avg, cls]'

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask=None):
        """
            :param x: torch.Tensor [batch_size, seq_max_length, input_size], [B, L, H] 一般是经过embedding后的值
            :param mask: [batch_size, max_len], 句长部分为0，padding部分为1。不影响卷积运算，max-pool一定不会pool到pad为0的位置
            :return:
            """
        # [B, L, H] -> [B, H, L] （注释：将 H 维度当作输入 channel 维度)
        x = torch.transpose(x, 1, 2)

        # convolution + activation  [[B, H, L], ... ]
        act_fn = self.activations[self.activation]

        x = [act_fn(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1)

        # mask
        if mask is not None:
            # [B, L] -> [B, 1, L]
            mask = mask.unsqueeze(1)
            x = x.masked_fill_(mask, 1e-12)

        # pooling
        # [[B, H, L], ... ] -> [[B, H], ... ]
        if self.pooling_strategy == 'max':
            xp = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
            # 等价于 xp = torch.max(x, dim=2)[0]

        elif self.pooling_strategy == 'avg':
            x_len = mask.squeeze().eq(0).sum(-1).unsqueeze(-1).to(torch.float).to(device=mask.device)
            xp = torch.sum(x, dim=-1) / x_len

        else:
            # self.pooling_strategy == 'cls'
            xp = x[:, :, 0]

        x = x.transpose(1, 2)
        x = self.dropout(x)
        xp = self.dropout(xp)

        return x, xp  # [B, L, Hs], [B, Hs]


def seq_len_to_mask(seq_len: Union[List, np.ndarray, torch.Tensor], max_len=None, mask_pos_to_true=True):
    """
    将一个表示sequence length的一维数组转换为二维的mask，默认pad的位置为1。
    转变 1-d seq_len到2-d mask.

    :param list, np.ndarray, torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, logger.error(f"seq_len can only have one dimension, got {seq_len.dim()} != 1.")
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len.device)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise logger.error("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def main():
    pass


if __name__ == '__main__':
    main()

