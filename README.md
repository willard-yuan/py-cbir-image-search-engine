## gosearch

gosearch是一个web图像搜索应用, 不同于过去传统的基于文本的图像检索(TBIR), gosearch是基于内容的图像检索(CBIR)。

### 框架

1. 首先提取图像库first500中各个图像的sift特征，每幅图像会对应一个提取的特征文件。
2. 采用BOW模型构建词汇向量，并配以tf-idf权重。
3. 进行匹配。

### 安装

- 由于`PCV`库依赖matplotlib、numpy等，建议你安装`python(x,y)`，在安装时推荐以**full**的形式安装。
- 安装好`python(x,y)`后，进入到**gosearch**的`PCV`目录下，运行下面命令：

```sh
python setup.py install
```
安装好`PCV`库后，返回上一级目录，即**gosearch**目录，进入`CherryPy-3.2.4`目录下，同样运行上面的命令安装好CherryPy。

### 运行

- 完成上面安装后，即可运行本应用。
