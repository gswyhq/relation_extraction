import logging
import matplotlib.pyplot as plt
import torch

# import models
# from hydra import utils
from config import MODEL_FILE
from model import PCNN
from preprocess import _preprocess_data

logger = logging.getLogger(__name__)


def main():

    data=[{'sentence': '《乡村爱情》是一部由知名导演赵本山在1985年所拍摄的农村青春偶像剧。', 'head': '乡村爱情', 'tail': '赵本山', 'head_type': '电视剧', 'tail_type': '人物'}]
    # preprocess data
    data, rels, vocab_size = _preprocess_data(data)
    # print("data={}, rels={}".format(data, rels))
    # data=[{'sentence': '《乡村爱情》是一部由知名导演赵本山在1985年所拍摄的农村青春偶像剧。', 'head': '乡村爱情', 'tail': '赵本山', 'head_type': '电视剧', 'tail_type': '人物', 'tokens': ['《', 'HEAD_电视剧', '》', '是', '一部', '由', '知名', '导演', 'TAIL_人物', '在', '1985', '年', '所', '拍摄', '的', '农村', '青春偶像', '剧', '。'], 'head_idx': 1, 'tail_idx': 8, 'token2idx': [18, 1, 20, 21, 31, 202, 504, 144, 52, 58, 793, 50, 811, 536, 30, 1721, 485, 318, 844], 'seq_len': 19, 'head_pos': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], 'tail_pos': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]}]
    # rels=OrderedDict([('None', {'index': 0, 'head_type': 'None', 'tail_type': 'None'}), ('导演', {'index': 1, 'head_type': '影视作品', 'tail_type': '人物'}), ('国籍', {'index': 2, 'head_type': '人物', 'tail_type': '国家'}), ('祖籍', {'index': 3, 'head_type': '人物', 'tail_type': '地点'}), ('主持人', {'index': 4, 'head_type': '电视综艺', 'tail_type': '人物'}), ('出生地', {'index': 5, 'head_type': '人物', 'tail_type': '地点'}), ('所在城市', {'index': 6, 'head_type': '景点', 'tail_type': '城市'}), ('所属专辑', {'index': 7, 'head_type': '歌曲', 'tail_type': '音乐专辑'}), ('连载网站', {'index': 8, 'head_type': '网络小说', 'tail_type': '网站'}), ('出品公司', {'index': 9, 'head_type': '影视作品', 'tail_type': '企业'}), ('毕业院校', {'index': 10, 'head_type': '人物', 'tail_type': '学校'})])

    device = torch.device('cpu')
    logger.info(f'device: {device}')

    model = PCNN(vocab_size)

    logger.info(f'\n {model}')
    model.load(MODEL_FILE, device=device)
    model.to(device)
    model.eval()

    x = dict()
    x['word'], x['lens'] = torch.tensor([data[0]['token2idx']]), torch.tensor([data[0]['seq_len']])

    x['head_pos'], x['tail_pos'] = torch.tensor([data[0]['head_pos']]), torch.tensor([data[0]['tail_pos']])

    for key in x.keys():
        x[key] = x[key].to(device)

    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.softmax(y_pred, dim=-1)[0]
        prob = y_pred.max().item()
        prob_rel = list(rels.keys())[y_pred.argmax().item()]
        print(f"\"{data[0]['head']}\" 和 \"{data[0]['tail']}\" 在句中关系为：\"{prob_rel}\"，置信度为{prob:.2f}。")

    if True:
        # maplot 默认显示不支持中文

        x = list(rels.keys())
        height = list(y_pred.cpu().numpy())
        # print('x={}'.format(x))
        # print('height={}'.format(height))
        # x=['None', '导演', '国籍', '祖籍', '主持人', '出生地', '所在城市', '所属专辑', '连载网站', '出品公司', '毕业院校']
        # height=[4.1607476e-05, 0.1968035, 0.0008424314, 0.002847272, 0.7719255, 0.0005891507, 0.0050658253, 0.0041004117, 0.01721394, 0.00051249505, 5.7812886e-05]
        plt.bar(x, height)
        for x, y in zip(x, height):
            plt.text(x, y, '%.2f' % y, ha="center", va="bottom", fontproperties="STKAITI")
        plt.xlabel('关系', fontproperties="STKAITI")
        plt.ylabel('置信度', fontproperties="STKAITI")
        plt.xticks(rotation=315, fontproperties="STKAITI")

        plt.show()


if __name__ == '__main__':
    main()
