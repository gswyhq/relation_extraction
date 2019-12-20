#!/usr/bin/python3
# coding: utf-8
import os
from collections import OrderedDict
from typing import List, Dict
from serializer import Serializer, logging
from dataloader import load_csv, save_pkl, load_pkl
from config import TRAIN_FILE, VALID_FILE, TEST_FILE, RELATION_FILE, OUTPUT_PATH, VOCAB_FILE
from vocab import Vocab

logger = logging.getLogger(__name__)

def preprocess():

    logger.info('===== start preprocess data =====')

    logger.info('load raw files...')
    train_data = load_csv(TRAIN_FILE)
    valid_data = load_csv(VALID_FILE)
    test_data = load_csv(TEST_FILE)
    relation_data = load_csv(RELATION_FILE)

    logger.info('convert relation into index...')
    rels = _handle_relation_data(relation_data)
    _add_relation_data(rels, train_data)
    _add_relation_data(rels, valid_data)
    _add_relation_data(rels, test_data)

    logger.info('verify whether use pretrained language models...')

    logger.info('serialize sentence into tokens...')
    serializer = Serializer(do_chinese_split=True, do_lower_case=True)
    serial = serializer.serialize
    _serialize_sentence(train_data, serial)
    _serialize_sentence(valid_data, serial)
    _serialize_sentence(test_data, serial)

    logger.info('build vocabulary...')
    vocab = Vocab('word')
    train_tokens = [d['tokens'] for d in train_data]
    valid_tokens = [d['tokens'] for d in valid_data]
    test_tokens = [d['tokens'] for d in test_data]
    sent_tokens = [*train_tokens, *valid_tokens, *test_tokens]
    for sent in sent_tokens:
        vocab.add_words(sent)
    vocab.trim(min_freq=3)

    logger.info('convert tokens into index...')
    _convert_tokens_into_index(train_data, vocab)
    _convert_tokens_into_index(valid_data, vocab)
    _convert_tokens_into_index(test_data, vocab)

    logger.info('build position sequence...')
    _add_pos_seq(train_data)
    _add_pos_seq(valid_data)
    _add_pos_seq(test_data)

    logger.info('save data for backup...')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    train_save_fp = os.path.join(OUTPUT_PATH, 'train.pkl')
    valid_save_fp = os.path.join(OUTPUT_PATH, 'valid.pkl')
    test_save_fp = os.path.join(OUTPUT_PATH, 'test.pkl')
    save_pkl(train_data, train_save_fp)
    save_pkl(valid_data, valid_save_fp)
    save_pkl(test_data, test_save_fp)


    vocab_save_fp = os.path.join(OUTPUT_PATH, 'vocab.pkl')
    vocab_txt = os.path.join(OUTPUT_PATH, 'vocab.txt')
    save_pkl(vocab, vocab_save_fp)
    logger.info('save vocab in txt file, for watching...')
    with open(vocab_txt, 'w', encoding='utf-8') as f:
        f.write(os.linesep.join(vocab.word2idx.keys()))

    logger.info('===== end preprocess data =====')



def _add_relation_data(rels: Dict, data: List) -> None:
    for d in data:
        d['rel2idx'] = rels[d['relation']]['index']
        d['head_type'] = rels[d['relation']]['head_type']
        d['tail_type'] = rels[d['relation']]['tail_type']


def _handle_relation_data(relation_data: List[Dict]) -> Dict:
    rels = OrderedDict()
    relation_data = sorted(relation_data, key=lambda i: int(i['index']))
    for d in relation_data:
        rels[d['relation']] = {
            'index': int(d['index']),
            'head_type': d['head_type'],
            'tail_type': d['tail_type'],
        }

    return rels

def _handle_pos_limit(pos: List[int], limit: int) -> List[int]:
    for i, p in enumerate(pos):
        if p > limit:
            pos[i] = limit
        if p < -limit:
            pos[i] = -limit
    return [p + limit + 1 for p in pos]

def _add_pos_seq(train_data: List[Dict]):
    for d in train_data:
        entities_idx = [d['head_idx'], d['tail_idx']
                        ] if d['head_idx'] < d['tail_idx'] else [d['tail_idx'], d['head_idx']]

        d['head_pos'] = list(map(lambda i: i - d['head_idx'], list(range(d['seq_len']))))
        d['head_pos'] = _handle_pos_limit(d['head_pos'], int(30))

        d['tail_pos'] = list(map(lambda i: i - d['tail_idx'], list(range(d['seq_len']))))
        d['tail_pos'] = _handle_pos_limit(d['tail_pos'], int(30))

def _convert_tokens_into_index(data: List[Dict], vocab):
    unk_str = '[UNK]'
    unk_idx = vocab.word2idx[unk_str]

    for d in data:
        d['token2idx'] = [vocab.word2idx.get(i, unk_idx) for i in d['tokens']]
        d['seq_len'] = len(d['token2idx'])

def _serialize_sentence(data: List[Dict], serial):
    for d in data:
        sent = d['sentence'].strip()
        sent = sent.replace(d['head'], ' head ', 1).replace(d['tail'], ' tail ', 1)
        d['tokens'] = serial(sent, never_split=['head', 'tail'])
        head_idx, tail_idx = d['tokens'].index('head'), d['tokens'].index('tail')
        d['head_idx'], d['tail_idx'] = head_idx, tail_idx
        d['tokens'][head_idx], d['tokens'][tail_idx] = 'HEAD_' + d['head_type'], 'TAIL_' + d['tail_type']


def _preprocess_data(data):
    vocab = load_pkl(VOCAB_FILE, verbose=False)
    relation_data = load_csv(RELATION_FILE, verbose=False)
    rels = _handle_relation_data(relation_data)
    vocab_size = vocab.count
    serializer = Serializer(do_chinese_split=True)
    serial = serializer.serialize

    _serialize_sentence(data, serial)
    _convert_tokens_into_index(data, vocab)
    _add_pos_seq(data)
    logger.info('start sentence preprocess...')

    return data, rels, vocab_size

def main():
    pass


if __name__ == '__main__':
    main()

