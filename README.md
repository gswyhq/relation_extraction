# DeepKE

Pytorch 实现中文实体关系抽取

## 环境依赖:

> python >= 3.6

- 安装依赖：`pip3 install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com`  

## 主要目录

```
.
├── config.py                 # 配置文件
├── data                      # 数据目录
│   ├── relation.csv          # 关系种类
│   ├── test.csv              # 测试数据集
│   ├── train.csv             # 训练数据集
│   └── valid.csv             # 验证数据集
├── dataloader.py
├── Dockerfile
├── model.py
├── output                    # 预处理数据后的存放目录
│   ├── test.pkl
│   ├── train.pkl
│   ├── valid.pkl
│   ├── vocab.pkl
│   └── vocab.txt
├── predict.py                # 测试入口文件         
├── preprocess.py             # 预处理数据文件
├── README.md                 # read me 文件
├── requirements.txt
├── serializer.py             # 预处理数据过程序列化字符串文件
├── train.py                  # 训练入口文件        
└── vocab.py                  # token 词表构建函数文件

```

## 快速开始

数据为 csv 文件，样式范例为：

sentence|relation|head|head_offset|tail|tail_offset
:---:|:---:|:---:|:---:|:---:|:---:
《岳父也是爹》是王军执导的电视剧，由马恩然、范明主演。|导演|岳父也是爹|1|王军|8
《九玄珠》是在纵横中文网连载的一部小说，作者是龙马。|连载网站|九玄珠|1|纵横中文网|7
提起杭州的美景，西湖总是第一个映入脑海的词语。|所在城市|西湖|8|杭州|2

- 安装依赖： `pip install -r requirements.txt`

- 存放数据：在 `data` 文件夹下存放训练数据。训练文件主要有三个文件。更多数据建议使用百度数据库中[Knowledge Extraction](http://ai.baidu.com/broad/download)。

  - `train.csv`：存放训练数据集

  - `valid.csv`：存放验证数据集

  - `test.csv`：存放测试数据集

  - `relation.csv`：存放关系种类(共10类关系: 毕业院校、出品公司、出生地、导演、国籍、连载网站、所属专辑、所在城市、主持人、祖籍)

- 开始训练：python3 train.py

- 模型结果保存在 `checkpoints` 文件夹内。

## docker镜像
- 使用已训练好的模型进行预测： `docker run --rm -it gswyhq/relation-extraction:v20191220 python3 predict.py` 

## 资料来源

> [浙江大学基于深度学习的开源中文关系抽取工具](https://github.com/zjunlp/deepke)

> [说明](https://mp.weixin.qq.com/s/K4NDbbRKsUT4uvb0MDnhKg)

