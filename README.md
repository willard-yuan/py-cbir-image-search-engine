## gosearch

gosearch是一个web图像搜索应用, 不同于过去传统的基于文本的图像检索(TBIR), gosearch是基于内容的图像检索(CBIR)。

### 框架

1. 首先提取图像库first500中各个图像的sift特征，每幅图像会对应一个提取的特征文件。
2. 采用BOW模型构建词汇向量，并配以tf-idf权重。
3. 进行匹配。
