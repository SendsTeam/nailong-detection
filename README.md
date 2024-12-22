# nailong-detection

基于 Pytorch 的深度学习奶龙图像检测

## 环境配置

本项目使用 venv 虚拟环境，使用以下命令创建虚拟环境：

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
    all/
        未归档的图片...
    train/
        nailong/
            0.jpg
            1.jpg
            2.jpg
            ...
        without-nailong/
            0.jpg
            1.jpg
            2.jpg
            ...
    valid/
        nailong/
            0.jpg
            1.jpg
            2.jpg
            ...
        without-nailong/
            0.jpg
            1.jpg
            2.jpg
            ...
```

## 训练

本项目提供一个已经训练好的模型参数，位于 `models/` 目录下。如果需要重新训练模型，请在项目根目录执行以下命令:

```shell
python src/train.py -d ./data -o ./models/best.pt -m resnet18
```

-d: 指定数据集路径
-o: 指定模型输出路径
-m: 指定模型名称，可选 resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2

## 运行

运行窗口应用程序：

```shell
python src/app.py
```
