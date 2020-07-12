## Bi-LSTM+Attention (Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification)


### 数据集：
#### SougouNews (http://www.sogou.com/labs/resource/cs.php) 中选出10个类别的新闻，每个类别5000个样本，组成总量为50000的数据集：
    it、women、business、sports、yule、learning、travel、auto、health、house


### 数据形式：
#### label \t content


### 文件解释
* main.py —— 主文件
* model.py —— 模型结构
* config.py —— 配置参数
* Data_Generate_SogouNews.py —— SougouNews新闻数据集处理脚本
* /data —— 数据存放文件夹
* /save_model —— 模型存储文件夹


### 参考资料
* Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification Classification (https://www.aclweb.org/anthology/P16-2034.pdf)
* https://www.cnblogs.com/jiangxinyang/p/10208227.html

