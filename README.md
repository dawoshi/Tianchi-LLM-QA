## nlp-tutorial

<p align="center">
  <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" />
  <img width="100" src="https://upload.wikimedia.org/wikipedia/en/7/7d/Bazel_logo.svg" />
  <img width="100" src = "https://upload.wikimedia.org/wikipedia/commons/1/18/ISO_C%2B%2B_Logo.svg" />
</p>

阿里天池: 2023全球智能汽车AI挑战赛——赛道一：AI大模型检索问答

## 一、 Structures

```text
.
├── Dockerfile
├── README.md
├── bm25_retriever.py
├── build.sh
├── config.py
├── data
│   ├── result.json
│   ├── test_question.json
│   └── train_a.pdf
├── faiss_retriever.py
├── llm_model.py
├── pdf_parse.py
├── pre_train_model
│   ├── Qwen-7B-Chat
│   │   └── download.py
│   ├── bge-reranker-large
│   └── m3e-large
├── qwen_generation_utils.py
├── requirements.txt
├── rerank_model.py
├── run.py
├── run.sh
└── vllm_wrapper.py
```

### 赛题概述
#### 赛题：基于大模型的文档检索问答

任务：本次比赛要求参赛选手以大模型为中心制作一个问答系统，回答用户的汽车相关问题。参赛选手需要根据问题，在文档中定位相关信息的位置，并根据文档内容通过大模型生成相应的答案。本次比赛涉及的问题主要围绕汽车使用、维修、保养等方面，具体可参考下面的例子：

问题1：怎么打开危险警告灯？
答案1：危险警告灯开关在方向盘下方，按下开关即可打开危险警告灯。

问题2：车辆如何保养？
答案2：为了保持车辆处于最佳状态，建议您定期关注车辆状态，包括定期保养、洗车、内部清洁、外部清洁、轮胎的保养、低压蓄电池的保养等。

问题3：靠背太热怎么办？
答案3：您好，如果您的座椅靠背太热，可以尝试关闭座椅加热功能。在多媒体显示屏上依次点击空调开启按键→座椅→加热，在该界面下可以关闭座椅加热。

#### 数据(复赛数据官方只提供部分参考样式)

[初赛训练数据集.pdf](https://tianchi-race-prod-sh.oss-cn-shanghai.aliyuncs.com/file/race/documents/532154/%E5%88%9D%E8%B5%9B%E8%AE%AD%E7%BB%83%E9%9B%86/%E5%88%9D%E8%B5%9B%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E9%9B%86.pdf?Expires=1703022585&OSSAccessKeyId=LTAI5t7fj2oKqzKgLGz6kGQc&Signature=pg9tnYgHDLkAlfCU%2Bs3h3QBrvfA%3D&response-content-disposition=attachment%3B%20)

[测试问题.json](https://tianchi-race-prod-sh.oss-cn-shanghai.aliyuncs.com/file/race/documents/532154/%E5%85%B6%E5%AE%83/%E6%B5%8B%E8%AF%95%E9%97%AE%E9%A2%98.json?Expires=1703022684&OSSAccessKeyId=LTAI5t7fj2oKqzKgLGz6kGQc&Signature=kTn%2BN4ZnY9tftVmz5kjNKOCoFAs%3D&response-content-disposition=attachment%3B%20)


### 解决方案

#### pdf解析
##### pdf分块解析
##### pdf 滑窗法解析

#### 召回
##### 向量召回
##### bm25召回

#### 重排序
##### cross-encoder

#### 推理优化
##### vllm
##### tensorRT-LLM

### 排名

#### 初赛2名
#### 复赛12名

The overall performance of BERT on **dev**:

|              | Accuracy (entity)  | Recall (entity)    | F1 score (entity)  |
| ------------ | ------------------ | ------------------ | ------------------ |
| BERT+Softmax | 0.7897     | 0.8031     | 0.7963    |
| BERT+CRF     | 0.7977 | 0.8177 | 0.8076 |
| BERT+Span    | 0.8132 | 0.8092 | 0.8112 |
| BERT+Span+adv    | 0.8267 | 0.8073 | **0.8169** |
| BERT-small(6 layers)+Span+kd    | 0.8241 | 0.7839 | 0.8051 |
| BERT+Span+focal_loss    | 0.8121 | 0.8008 | 0.8064 |
| BERT+Span+label_smoothing   | 0.8235 | 0.7946 | 0.8088 |
