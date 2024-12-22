import os
from matplotlib import rcParams
# 设置字体为黑体
rcParams['font.sans-serif'] = ['SimHei']  # 指定字体
rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from utils.args import read_args
import time
import warnings
warnings.filterwarnings("ignore")
import copy

# 配置 transforms
data_transforms = {
    # 组合训练集的数据集组件
    'train':transforms.Compose([
        transforms.Resize([96,96]), # 将用于训练的图像统一裁剪为96x96的大小，以适应模型输入
        # 数据增强部分，随机水平垂直翻转图像，提高模型的泛化能力
        transforms.RandomRotation(45), # 随机旋转图像-45到45度
        transforms.CenterCrop(64), # 从中心裁剪64x64的图像
        transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转图像
        transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转图像
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), # 随机调整图像的亮度、对比度、饱和度和色调
        transforms.RandomGrayscale(p=0.025) , # 随机将图像转换为灰度图像
        transforms.ToTensor(), # 将图像转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 对图像进行归一化处理
    ]),
    # 组合验证集的数据集组件，注意这里的组件要和训练集的部分组件一致
    'valid':transforms.Compose([
        transforms.Resize([46,46]), # 将用于训练的图像统一裁剪为46x46的大小，以适应模型输入
        transforms.ToTensor(), # 将图像转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 对图像进行归一化处理
    ]),
}

# 读取命令行参数
data_dir, model_checkpoint_filename, model_name = read_args()

# 配置batch_size为128
batch_size = 128

# 使用DataLoader加载数据集
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']
}

# 创建数据加载器
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']
}

# 查看数据集大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

# 查看类别
class_names = image_datasets['train'].classes

# 模型分类数
model_classes_num = 2
feature_extract = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    return model

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
    elif model_name == 'resnet50':
        model_ft = models.resnet50()
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 64
    return model_ft, input_size


# 初始化模型
model_ft, input_size = initialize_model(model_name, model_classes_num, feature_extract, use_pretrained=True)

# 设置模型的运行设备(cpu/gpu)
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('GPU is available')
else:
    print('GPU is not available')
device = torch.device("cuda:0" if train_on_gpu else "cpu")
model_ft = model_ft.to(device)

# 是否训练所有层
params_to_update = model_ft.fc.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# 优化器设置(这里只添加本轮需要训练的参数)
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
# 学习率递减策略
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1) # 每10个epoch，学习率降低10倍
# 损失函数设置
criterion = nn.CrossEntropyLoss()


# 训练模型
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25,save_path='./best.pt'):
    # 开始训练时的时间
    since = time.time()
    # 最好的模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 最好的ACC
    best_acc = 0.0

    # 遍历epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都分为训练和验证两个阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0
            # 遍历数据集
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # 获取模型输出
                    _, preds = torch.max(outputs, 1) # 获取预测结果
                    loss = criterion(outputs, labels) # 计算损失

                    # 反向传播和优化，仅在训练阶段进行
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step() # 学习率递减

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 保存最好的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存最好的模型参数
                torch.save({
                    'state_dict':model.state_dict(),
                    'best_acc':best_acc,
                    'optimizer':optimizer
                }, save_path)
                print(f"Saved best model to {save_path}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最好的模型权重
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft,dataloaders,criterion,optimizer_ft,scheduler,num_epochs=25,save_path=model_checkpoint_filename)

# 解锁所有参数
for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有参数，学习率调小一点
optimizer = optim.Adam(model_ft.parameters(),lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 加载之前训练好的权重参数
checkpoint = torch.load(model_checkpoint_filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

# 训练模型
model_ft = train_model(model_ft,dataloaders,criterion,optimizer_ft,scheduler,num_epochs=9,save_path=model_checkpoint_filename)