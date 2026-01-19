# 导入所需的库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# 检查是否有GPU可用，如果有则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据预处理：将图片调整为224x224、转为3通道，并转为Tensor格式
transform = transforms.Compose([
    transforms.Resize((224, 224)),                 # 将28x28图像调整为224x224（适配ResNet输入）
    transforms.Grayscale(num_output_channels=3),   # 将单通道灰度图转为3通道（适配ResNet预训练模型）
    transforms.ToTensor()                          # 转换为张量
    # 可选添加：transforms.Normalize((0.5,), (0.5,))
])

# 加载FashionMNIST训练集和测试集，自动下载数据到本地./data文件夹
train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 使用DataLoader包装数据集，支持批量加载与打乱顺序
batch_size = 64  # 批大小，影响训练速度与显存占用
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 定义一个使用ResNet18为基础的分类模型，适配10类输出
class FashionResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的ResNet18模型
        self.base = torchvision.models.resnet18(weights='DEFAULT')
        # 修改ResNet的第一个卷积层以接受3通道输入（默认是3通道，但防止灰度图输入报错）
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 获取原fc层输入特征维度
        num_features = self.base.fc.in_features
        # 替换fc层，使其输出10个类别的预测值
        self.base.fc = nn.Linear(num_features, 10)

    def forward(self, x):
        return self.base(x)  # 前向传播调用ResNet主体结构

# 实例化模型、定义损失函数和优化器
model = FashionResNet().to(device)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 定义训练过程函数
def train_model(model, epochs=20):
    train_loss = []  # 每轮的训练损失
    train_acc = []   # 每轮训练准确率
    val_acc = []     # 每轮验证准确率

    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        # 遍历整个训练集
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()           # 梯度清零
            outputs = model(images)         # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()                 # 反向传播
            optimizer.step()                # 更新参数

            running_loss += loss.item()     # 累计损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测标签
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # 禁用梯度计算，加快推理
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 记录每轮的指标
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(100 * correct / total)
        val_acc.append(100 * val_correct / val_total)

        print(f'Epoch [{epoch+1}/{epochs}] | Loss: {train_loss[-1]:.4f} | Train Acc: {train_acc[-1]:.2f}% | Val Acc: {val_acc[-1]:.2f}%')

    return train_loss, train_acc, val_acc

# 训练模型
train_loss, train_acc, val_acc = train_model(model, epochs=20)

# 可视化训练过程的损失和准确率变化
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# 模型评估与混淆矩阵可视化
def evaluate_model(model):
    model.eval()
    all_labels = []  # 所有真实标签
    all_preds = []   # 所有预测标签

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_set.classes,
                yticklabels=train_set.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.pause(0.1)  # 避免图像窗口卡死
    plt.show()

    # 保存混淆矩阵图像
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存至 confusion_matrix.png")

    # 打印分类报告，包括precision, recall, f1-score等
    print(classification_report(all_labels, all_preds, target_names=train_set.classes))

# 调用评估函数
evaluate_model(model)

# 创建模型保存目录
model_dir = "./saved_models"
os.makedirs(model_dir, exist_ok=True)

# 保存完整模型（包括架构+参数）
torch.save(model, f'{model_dir}/fashion_resnet_full.pth')

# 推荐保存方式：仅保存状态字典（需配合模型定义加载）
torch.save({
    'model_state': model.state_dict(),         # 模型参数
    'optimizer_state': optimizer.state_dict(), # 优化器状态（用于恢复训练）
    'transform': transform                     # 预处理流程（可记录数据变换方式）
}, f'{model_dir}/model_checkpoint.pth')

print(f"模型已保存至 {model_dir} 目录")
