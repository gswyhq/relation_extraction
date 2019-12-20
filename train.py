#!/usr/bin/python3
# coding: utf-8


import logging
import os
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support

from config import PLOT_UTILS, OUTPUT_PATH, BATCH_SIZE, EPOCH
from preprocess import preprocess
from dataloader import load_pkl
from model import PCNN

logger = logging.getLogger(__name__)

def manual_seed(num: int = 1) -> None:
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)

class PRMetric():
    def __init__(self):
        """
        暂时调用 sklearn 的方法
        """
        self.y_true = np.empty(0)
        self.y_pred = np.empty(0)

    def reset(self):
        self.y_true = np.empty(0)
        self.y_pred = np.empty(0)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.argmax(y_pred, axis=-1)

        self.y_true = np.append(self.y_true, y_true)
        self.y_pred = np.append(self.y_pred, y_pred)

    def compute(self):
        p, r, f1, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='macro', warn_for=tuple())
        _, _, acc, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='micro', warn_for=tuple())

        return acc, p, r, f1

def train(epoch, model, dataloader, optimizer, criterion, device, writer):
    model.train()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        metric.update(y_true=y, y_pred=y_pred)
        losses.append(loss.item())

        data_total = len(dataloader.dataset)
        data_cal = data_total if batch_idx == len(dataloader) else batch_idx * len(y)
        if (batch_idx % 10 == 0) or batch_idx == len(dataloader):
            # p r f1 皆为 macro，因为micro时三者相同，定义为acc
            acc, p, r, f1 = metric.compute()
            logger.info(f'Train Epoch {epoch}: [{data_cal}/{data_total} ({100. * data_cal / data_total:.0f}%)]\t'
                        f'Loss: {loss.item():.6f}')
            logger.info(f'Train Epoch {epoch}: Acc: {100. * acc:.2f}%\t'
                        f'macro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')


    if PLOT_UTILS == 'matplot':
        plt.plot(losses)
        plt.title(f'epoch {epoch} train loss')
        plt.show()

    if PLOT_UTILS == 'tensorboard':
        for i in range(len(losses)):
            writer.add_scalar(f'epoch_{epoch}_training_loss', losses[i], i)

    return losses[-1]


def validate(epoch, model, dataloader, criterion, device):
    model.eval()

    metric = PRMetric()
    losses = []

    for batch_idx, (x, y) in enumerate(dataloader, 1):
        for key, value in x.items():
            x[key] = value.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = model(x)

            loss = criterion(y_pred, y)

            metric.update(y_true=y, y_pred=y_pred)
            losses.append(loss.item())

    loss = sum(losses) / len(losses)
    acc, p, r, f1 = metric.compute()
    data_total = len(dataloader.dataset)

    if epoch >= 0:
        logger.info(f'Valid Epoch {epoch}: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}')
        logger.info(f'Valid Epoch {epoch}: Acc: {100. * acc:.2f}%\tmacro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')
    else:
        logger.info(f'Test Data: [{data_total}/{data_total}](100%)\t Loss: {loss:.6f}')
        logger.info(f'Test Data: Acc: {100. * acc:.2f}%\tmacro metrics: [p: {p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

    return f1, loss


class CustomDataset(Dataset):
    """默认使用 List 存储数据"""
    def __init__(self, fp):
        self.file = load_pkl(fp)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)

def collate_fn():
    def collate_fn_intra(batch):
        batch.sort(key=lambda data: data['seq_len'], reverse=True)

        max_len = batch[0]['seq_len']

        def _padding(x, max_len):
            return x + [0] * (max_len - len(x))

        x, y = dict(), []
        word, word_len = [], []
        head_pos, tail_pos = [], []
        pcnn_mask = []
        for data in batch:
            word.append(_padding(data['token2idx'], max_len))
            word_len.append(data['seq_len'])
            y.append(int(data['rel2idx']))


            head_pos.append(_padding(data['head_pos'], max_len))
            tail_pos.append(_padding(data['tail_pos'], max_len))

        x['word'] = torch.tensor(word)
        x['lens'] = torch.tensor(word_len)
        y = torch.tensor(y)

        x['head_pos'] = torch.tensor(head_pos)
        x['tail_pos'] = torch.tensor(tail_pos)

        return x, y

    return collate_fn_intra

def main():
    batch_size = BATCH_SIZE

    # device
    device = torch.device('cpu')
    logger.info(f'device: {device}')

    # 如果不修改预处理的过程，这一步最好注释掉，不用每次运行都预处理数据一次
    if True:
        preprocess()

    train_data_path = os.path.join(OUTPUT_PATH, 'train.pkl')
    valid_data_path = os.path.join(OUTPUT_PATH, 'valid.pkl')
    test_data_path = os.path.join(OUTPUT_PATH, 'test.pkl')
    vocab_path = os.path.join(OUTPUT_PATH, 'vocab.pkl')

    vocab=load_pkl(vocab_path)

    vocab_size = vocab.count

    train_dataset = CustomDataset(train_data_path)
    valid_dataset = CustomDataset(valid_data_path)
    test_dataset = CustomDataset(test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn())
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn())

    model = PCNN(vocab_size)
    model.to(device)
    logger.info(f'\n {model}')

    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_epoch = -1, 0
    es_loss, es_f1, es_epoch, es_patience, best_es_epoch, best_es_f1, es_path, best_es_path = 1e8, -1, 0, 0, 0, -1, '', ''
    train_losses, valid_losses = [], []


    if PLOT_UTILS == 'tensorboard':
        writer = SummaryWriter('tensorboard')
    else:
        writer = None

    logger.info('=' * 10 + ' Start training ' + '=' * 10)

    for epoch in range(1, EPOCH+1):
        manual_seed(1 + epoch)
        train_loss = train(epoch, model, train_dataloader, optimizer, criterion, device, writer)
        valid_f1, valid_loss = validate(epoch, model, valid_dataloader, criterion, device)
        scheduler.step(valid_loss)
        model_path = model.save(epoch)
        # logger.info(model_path)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if best_f1 < valid_f1:
            best_f1 = valid_f1
            best_epoch = epoch
        # 使用 valid loss 做 early stopping 的判断标准
        if es_loss > valid_loss:
            es_loss = valid_loss
            es_f1 = valid_f1
            es_epoch = epoch
            es_patience = 0
            es_path = model_path
        else:
            es_patience += 1
            if es_patience >= 6:
                best_es_epoch = es_epoch
                best_es_f1 = es_f1
                best_es_path = es_path


    if PLOT_UTILS == 'matplot':
        plt.plot(train_losses, 'x-')
        plt.plot(valid_losses, '+-')
        plt.legend(['train', 'valid'])
        plt.title('train/valid comparison loss')
        plt.show()

    if PLOT_UTILS == 'tensorboard':
        for i in range(len(train_losses)):
            writer.add_scalars('train/valid_comparison_loss', {
                'train': train_losses[i],
                'valid': valid_losses[i]
            }, i)
        writer.close()

    logger.info(f'best(valid loss quota) early stopping epoch: {best_es_epoch}, '
                f'this epoch macro f1: {best_es_f1:0.4f}')
    logger.info(f'this model save path: {best_es_path}')
    logger.info(f'total {EPOCH} epochs, best(valid macro f1) epoch: {best_epoch}, '
                f'this epoch macro f1: {best_f1:.4f}')

    validate(-1, model, test_dataloader, criterion, device)


if __name__ == '__main__':
    main()
