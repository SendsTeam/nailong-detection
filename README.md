# nailong-detection

基于Pytorch的深度学习奶龙图像检测

## 环境配置

本项目使用venv虚拟环境，使用以下命令创建虚拟环境：
```shell
python -m venv venv
```

激活虚拟环境：
```shell
./venv/bin/activate
```

在虚拟环境中安装依赖：
```shell
pip install -r requirements.txt
```

退出虚拟环境
```shell
deactivate
```

## 数据集

本项目使用的数据集为奶龙图像数据集，数据集包含训练集和测试集，每个数据集包含正样本和负样本。数据集的目录结构如下：
```
data/
    train/
        positive/
            img1.jpg
            img2.jpg
            ...
        negative/
            img1.jpg
            img2.jpg
            ...
    valid/
        positive/
            img1.jpg
            img2.jpg
            ...
        negative/
            img1.jpg
            img2.jpg
            ...
```