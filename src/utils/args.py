import argparse
import getopt
import sys

def read_args():
    parser = argparse.ArgumentParser(description="训练模型的脚本。")

    # 定义 -d / --data 参数，指定数据目录
    parser.add_argument(
        '-d', '--data',
        type=str,
        required=True,
        help='数据集的路径，例如 "../data"'
    )

    # 定义 -o / --output 参数，指定输出目录
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='模型输出的路径，例如 "../models"'
    )

    # 定义 -m / --model 参数，指定模型名称
    parser.add_argument(
        '-m', '--model',
        type=str,
        choices=['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19'],
        default='resnet18',
        help='选择要训练的模型，例如 "resnet18"'
    )

    args = parser.parse_args()

    data_dir = args.data
    output_file = args.output
    model_name = args.model

    return data_dir, output_file, model_name