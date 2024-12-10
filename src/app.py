import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models
from torch import nn

# 图片预处理
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 修改为模型输入的尺寸
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 确保和训练时一致
])

# 定义类别映射
classes_map = {
    '0': '奶龙',
    '1': '并非奶龙'
}

# 初始化模型
def init_model(path: str):
    model = models.resnet18()
    checkpoint = torch.load(path, map_location=torch.device('cpu'))  # 支持无GPU环境
    num_ftrs = model.fc.in_features
    num_classes = 2
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # 切换到评估模式
    return model


# 加载图片
def load_image(image_path: str):
    img = Image.open(image_path).convert('RGB')  # 确保是RGB格式
    img_tensor = image_transforms(img).unsqueeze(0)  # 增加一个batch维度
    return img_tensor, img


# 预测单张图片
def predict_single_image(image_path: str, model):
    # 加载图片
    img_tensor, original_image = load_image(image_path)
    
    # 前向传播
    with torch.no_grad():
        output = model(img_tensor)
    
    # 获取预测结果
    _, pred_tensor = torch.max(output, 1)
    predicted_label = pred_tensor.item()  # 转换为Python数据类型
    
    return predicted_label, original_image


# Tkinter 界面逻辑
class ImageClassifierApp:
    font_size = 14
    def __init__(self, root, model):
        self.root = root
        self.root.title("奶龙图片检测")
        self.root.geometry("500x600")
        
        # 标签
        self.label = Label(root, text="请上传图片并进行检测", font=("楷体", self.font_size))
        self.label.pack(pady=20)
        
        # 图片显示
        self.img_label = Label(root)
        self.img_label.pack(pady=20)
        
        # 检测结果
        self.result_label = Label(root, text="", font=("楷体", self.font_size))
        self.result_label.pack(pady=20)
        
        # 上传按钮
        self.upload_button = Button(root, text="上传图片", command=self.upload_image, font=("楷体", self.font_size))
        self.upload_button.pack(side='left', padx=75)
        
        # 检测按钮
        self.detect_button = Button(root, text="检测图片", command=self.detect_image, font=("楷体", self.font_size), state="disabled")
        self.detect_button.pack(side='right', padx=75)
        
        self.model = model
        self.image_path = None

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            # 显示图片
            img = Image.open(self.image_path)
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.img_label.configure(image=img_tk)
            self.img_label.image = img_tk
            self.result_label.config(text="")
            self.detect_button.config(state="normal")
    
    def detect_image(self):
        if self.image_path:
            predicted_label, _ = predict_single_image(self.image_path, self.model)
            category = classes_map[str(predicted_label)]
            self.result_label.config(text=f"预测结果: {category}")


# 加载模型
model = init_model("../models/best.pt")  # 替换为模型路径

# 创建Tkinter窗口
root = tk.Tk()
app = ImageClassifierApp(root, model)
root.mainloop()
