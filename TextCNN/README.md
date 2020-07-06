## TextCNN

### 数据集：
##### SougouNews（http://www.sogou.com/labs/resource/cs.php）中选出10个类别的新闻，每个类别5000个样本，组成总量为50000的数据集：
    it、women、business、sports、yule、learning、travel、auto、health、house



### 数据形式：
##### label \t content



### 文件解释
* main.py —— 主文件
* model.py —— 模型结构
* config.py —— 配置参数
* Data_Generate_SogouNews.py —— SougouNews新闻数据集处理脚本
* /data —— 数据存放文件夹
* /save_model —— 模型存储文件夹



### 参考资料
* Convolutional Neural Networks for Sentence Classification（https://arxiv.org/pdf/1408.5882.pdf）
* https://www.cnblogs.com/bymo/p/9675654.html

