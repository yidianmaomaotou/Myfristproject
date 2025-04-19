import os
import torch
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import models

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 改进后的数据增强函数
def get_augmentations():
    return A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# 改进后的CBAM模块（调整reduction_ratio）
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_att = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        x = x * channel_att
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att

# 改进后的SE模块（调整reduction_ratio）
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = self.avg_pool(x)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.fc(squeeze).view(x.size(0), x.size(1), 1, 1)
        return x * excitation

# 多尺度特征融合模块（MLFI）
class MLFI(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3(x)
        feat3 = self.conv5x5(x)
        feat4 = self.conv7x7(x)
        return self.fusion(torch.cat([feat1, feat2, feat3, feat4], dim=1))

# 保留CBAM和SE模块的模型
class EfficientNet_Balanced(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
        
        # 保留关键模块
        self.cbam = CBAM(in_channels=1280, reduction_ratio=8)
        self.se = SEBlock(in_channels=1280, reduction_ratio=8)
        self.mlfi = MLFI(in_channels=1280, out_channels=1280)
        
        # 添加Dropout防止过拟合
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base._fc.in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = self.cbam(x)
        x = self.se(x)
        x = self.mlfi(x)
        x = self.base._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x), x

# EfficientNet-B4 模型
class EfficientNet_B4(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.base = models.efficientnet_b4(pretrained=True)
        self.base.classifier[1] = nn.Linear(self.base.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.base(x), x

# ResNeSt-50 模型
class ResNeSt50(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.base = models.resnet50(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base(x), x

# FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# 改进后的对抗训练
def train_epoch_with_adversarial(model, loader, optimizer, criterion, epsilon=0.01):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad = True
        
        # 原始前向传播
        outputs = model(inputs)
        loss = criterion(outputs[0], labels)
        
        # 生成对抗样本
        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad = inputs.grad.data
        perturbed_data = fgsm_attack(inputs, epsilon, data_grad).detach()

        # 对抗训练（调整权重）
        outputs_adv = model(perturbed_data)
        loss_adv = criterion(outputs_adv[0], labels)

        total_loss_batch = 0.5 * loss + 0.5 * loss_adv  # 调整后的权重
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item() * inputs.size(0)
    
    return total_loss / len(loader.dataset)

# 权重初始化函数
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

# 加载DeepWeeds标签和图片路径
def load_deepweeds_labels(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        filename = row['Filename'].replace(',', '.')  # 修复错误逗号
        img_path = os.path.join(img_dir, filename)
        label = int(row['Label'])
        data.append((img_path, label))
    return data

# 自定义Dataset类
class DeepWeedsDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.data)

# 加权交叉熵损失
class_weights = [1.0, 1.0, 1.0, 2.0, 1.5, 1.0, 1.0, 1.0, 0.5]
class_weights = torch.tensor(class_weights).float().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 修改后的训练验证函数，使用全部数据
def train_val_split(data, model_class, epochs=20):
    transform = get_augmentations()
    dataset = DeepWeedsDataset(data, transform=transform)
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    
    # 训练模型
    model = model_class(num_classes=9).to(device)
    model.apply(init_weights)  # 应用权重初始化
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # 调整学习率调度器
    
    best_acc = 0.0
    for epoch in range(epochs):
        # 训练并返回损失
        train_loss = train_epoch_with_adversarial(model, train_loader, optimizer, criterion, epsilon=0.03)
        
        # 验证评估
        val_acc = evaluate(model, val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 记录最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型
            torch.save(model.state_dict(), "best_model.pth")
        
        # 打印训练进度
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {lr:.2e} | "
              f"Best Acc: {best_acc:.4f}")
    
    return best_acc

# 评估函数
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)[0]  # 只需要分类结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 生成热力图
def generate_heatmap(model, image_path):
    # 读取图片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 应用数据增强
    transform = get_augmentations()
    augmented = transform(image=img)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs, features = model(image_tensor)
    
    # 获取预测类别
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    
    # 生成热力图
    heatmap = features[0].cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # 将热力图调整为与原始图像相同的大小
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # 将热力图叠加到原始图像上
    heatmap_image = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
    heatmap_image = cv2.addWeighted(img, 0.5, heatmap_image, 0.5, 0)
    
    return predicted_class, img, heatmap_image

# 生成混淆矩阵
def generate_confusion_matrix(model, loader, num_classes=9):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)[0]
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    return cm

# 创建图形界面
class WeedDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("杂草识别系统")
        
        # 加载模型
        self.model = EfficientNet_Balanced(num_classes=9).to(device)
        self.model.apply(init_weights)  # 应用权重初始化
        if os.path.exists("best_model.pth"):
            self.model.load_state_dict(torch.load("best_model.pth", map_location=device))
        else:
            print("模型权重文件不存在，将使用随机初始化的模型。")
        
        # 创建界面组件
        self.create_widgets()
    
    def create_widgets(self):
        # 按钮
        self.upload_button = Button(self.root, text="上传图片", command=self.upload_image)
        self.upload_button.pack(pady=10)
        
        # 图片显示区域
        self.image_frame = Frame(self.root)
        self.image_frame.pack(pady=10)
        
        # 结果显示区域
        self.result_frame = Frame(self.root)
        self.result_frame.pack(pady=10)
        
        # 混淆矩阵显示区域
        self.cm_frame = Frame(self.root)
        self.cm_frame.pack(pady=10)
        
        # 预测结果标签
        self.result_label = Label(self.result_frame, text="")
        self.result_label.pack()
        
        # 模型对比按钮
        self.compare_button = Button(self.root, text="模型对比", command=self.compare_models)
        self.compare_button.pack(pady=10)
    
    def upload_image(self):
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        # 生成热力图
        predicted_class, original_img, heatmap_img = generate_heatmap(self.model, file_path)
        
        # 显示原始图片
        original_img = Image.fromarray(original_img)
        original_img = original_img.resize((400, 400), Image.LANCZOS)
        original_img_tk = ImageTk.PhotoImage(original_img)
        if not hasattr(self, 'original_label'):
            self.original_label = Label(self.image_frame, image=original_img_tk)
            self.original_label.image = original_img_tk
            self.original_label.pack(side=tk.LEFT, padx=10)
        else:
            self.original_label.config(image=original_img_tk)
            self.original_label.image = original_img_tk
        
        # 显示热力图
        heatmap_img = Image.fromarray(heatmap_img)
        heatmap_img = heatmap_img.resize((400, 400), Image.LANCZOS)
        heatmap_img_tk = ImageTk.PhotoImage(heatmap_img)
        if not hasattr(self, 'heatmap_label'):
            self.heatmap_label = Label(self.image_frame, image=heatmap_img_tk)
            self.heatmap_label.image = heatmap_img_tk
            self.heatmap_label.pack(side=tk.LEFT, padx=10)
        else:
            self.heatmap_label.config(image=heatmap_img_tk)
            self.heatmap_label.image = heatmap_img_tk
        
        # 显示预测结果
        self.result_label.config(text=f"预测类别: {predicted_class}")
    
    def compare_models(self):
        # 加载数据
        train_data = load_deepweeds_labels("labels.csv", "images")
        
        # 修改后的模型对比函数，使用全部数据
        def compare_models_complete_data(train_data):
            # 训练并评估你的模型
            your_model_acc = train_val_split(train_data, EfficientNet_Balanced, epochs=20)
            
            # 训练并评估 EfficientNet-B4
            efficientnet_b4_acc = train_val_split(train_data, EfficientNet_B4, epochs=20)
            
            # 训练并评估 ResNeSt-50
            resnest50_acc = train_val_split(train_data, ResNeSt50, epochs=20)
            
            # 对比表格
            models = ["Your Model", "EfficientNet-B4", "ResNeSt-50"]
            accuracies = [your_model_acc, efficientnet_b4_acc, resnest50_acc]
            
            # 打印对比表格
            print("\nModel Comparison:")
            print(f"{'Model':<20} {'Accuracy':<15}")
            print("-" * 35)
            for i in range(len(models)):
                print(f"{models[i]:<20} {accuracies[i]:<15.4f}")
            
            # 可视化对比
            plt.figure(figsize=(10, 6))
            plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.title('Model Accuracy Comparison')
            plt.ylim(0, 1)
            plt.tight_layout()
            
            # 在图形界面中显示图表
            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.cm_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
            
            # 混淆矩阵对比
            transform = get_augmentations()
            dataset = DeepWeedsDataset(train_data, transform=transform)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
            
            # 你的模型的混淆矩阵
            your_model = EfficientNet_Balanced(num_classes=9).to(device)
            if os.path.exists("best_model.pth"):
                your_model.load_state_dict(torch.load("best_model.pth", map_location=device))
            your_cm = generate_confusion_matrix(your_model, val_loader)
            
            # EfficientNet-B4 的混淆矩阵
            efficientnet_b4 = EfficientNet_B4(num_classes=9).to(device)
            if os.path.exists("efficientnet_b4_best.pth"):
                efficientnet_b4.load_state_dict(torch.load("efficientnet_b4_best.pth", map_location=device))
            efficientnet_b4_cm = generate_confusion_matrix(efficientnet_b4, val_loader)
            
            # ResNeSt-50 的混淆矩阵
            resnest50 = ResNeSt50(num_classes=9).to(device)
            if os.path.exists("resnest50_best.pth"):
                resnest50.load_state_dict(torch.load("resnest50_best.pth", map_location=device))
            resnest50_cm = generate_confusion_matrix(resnest50, val_loader)
            
            # 绘制混淆矩阵
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            sns.heatmap(your_cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Your Model Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plt.subplot(1, 3, 2)
            sns.heatmap(efficientnet_b4_cm, annot=True, fmt='d', cmap='Blues')
            plt.title('EfficientNet-B4 Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plt.subplot(1, 3, 3)
            sns.heatmap(resnest50_cm, annot=True, fmt='d', cmap='Blues')
            plt.title('ResNeSt-50 Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plt.tight_layout()
            
            # 在图形界面中显示图表
            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.cm_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
        
        compare_models_complete_data(train_data)

# 主程序
if __name__ == "__main__":
    # 读取数据
    train_data = load_deepweeds_labels("labels.csv", "images")
    
    # 训练模型
    best_acc = train_val_split(train_data, EfficientNet_Balanced, epochs=20)
    
    print(f"最佳验证精度: {best_acc:.4f}")
    
    # 启动GUI
    root = tk.Tk()
    app = WeedDetectionApp(root)
    root.mainloop()