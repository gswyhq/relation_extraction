#!/usr/bin/python3
# coding: utf-8

import os
PLOT_UTILS= 'tensorboard'  # [matplot, tensorboard]
Path = str
TRAIN_FILE = './data/train.csv'
VALID_FILE = './data/valid.csv'
TEST_FILE = './data/test.csv'
RELATION_FILE = './data/relation.csv'

OUTPUT_PATH = './output'  # 预处理数据后的存放目录

VOCAB_FILE = './output/vocab.pkl'
EPOCH = 20
BATCH_SIZE = 32

# 自定义模型存储的路径
MODEL_PATH = './checkpoints' # 模型结果保存路径
MODEL_FILE = os.getenv('MODEL_FILE', './checkpoints/cnn_2019-12-20_11-21-34_epoch14.pth')  # 预测时，加载的模型文件


def main():
    pass


if __name__ == '__main__':
    main()
