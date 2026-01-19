import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
import torch.nn.functional as F  # 函数式API
import torchvision  # 用于计算机视觉的工具库
import torchvision.transforms as transforms  # 图像变换工具
from torch.utils.data import DataLoader  # 数据加载器
import matplotlib.pyplot as plt  # 绘图工具
import numpy as np  # 数值计算库
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # 用于评估模型

# 设置设备 - 如果有GPU就用GPU，否则用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============= 数据准备 =============
# 定义数据预处理
"""
transforms.Compose: 将多个变换操作组合在一起
- transforms.ToTensor(): 将PIL图像转换为张量，并将像素值从[0,255]归一化到[0,1]
- transforms.Normalize(): 对数据进行标准化，参数为(均值,标准差)
  - Fashion-MNIST是灰度图，所以只有一个通道
"""
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量，并将像素值归一化到[0,1]
    transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1,1]，参数是(均值,标准差)
])

# 加载Fashion-MNIST数据集
"""
FashionMNIST数据集包含10个类别的服饰图像：
0. T-shirt/top（T恤/上衣）
1. Trouser（裤子）
2. Pullover（套头衫）
3. Dress（连衣裙）
4. Coat（外套）
5. Sandal（凉鞋）
6. Shirt（衬衫）
7. Sneaker（运动鞋）
8. Bag（包）
9. Ankle boot（短靴）

参数解释:
- root: 数据保存的根目录
- train: True表示训练集，False表示测试集
- download: 如果数据不存在，是否下载
- transform: 应用的数据变换
"""
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 定义数据加载器
"""
DataLoader: 负责高效地加载批量数据
参数解释:
- dataset: 要加载的数据集
- batch_size: 每批数据的大小，较大的值可能加速训练但需要更多内存
- shuffle: 是否打乱数据（训练集通常需要打乱，测试集不必）
"""
batch_size = 64  # 一次处理64张图片，可以调整这个值
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建类别名称列表，用于显示结果
class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套', 
               '凉鞋', '衬衫', '运动鞋', '包', '短靴']

# ============= 模型定义 =============
# 定义CNN模型
class FashionCNN(nn.Module):
    def __init__(self):
        """
        初始化CNN模型结构
        
        模型结构:
        Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> ReLU -> FC
        """
        super(FashionCNN, self).__init__()
        
        # 第一个卷积块：卷积->激活->池化
        # 输入: 1通道 28x28图像，输出: 32通道的特征图
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        
        # 第二个卷积块：卷积->激活->池化
        # 输入: 32通道特征图，输出: 64通道的特征图
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Dropout层，随机丢弃25%的神经元，防止过拟合
        self.dropout = nn.Dropout2d(0.25)
        
        # 全连接层1：输入是特征图展平后的向量，输出128维特征
        # 7x7是两次最大池化后的特征图大小: 28/2/2 = 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # 全连接层2：输出层，输出10个类别的分数
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        定义前向传播过程
        
        参数:
        - x: 输入张量，形状为[batch_size, 1, 28, 28]
        
        返回:
        - 模型输出，形状为[batch_size, 10]
        """
        # 第一个卷积层 + ReLU激活 + 最大池化
        x = F.relu(self.conv1(x))      # 输出形状: [batch_size, 32, 28, 28]
        x = F.max_pool2d(x, 2)         # 输出形状: [batch_size, 32, 14, 14]
        
        # 第二个卷积层 + ReLU激活 + 最大池化
        x = F.relu(self.conv2(x))      # 输出形状: [batch_size, 64, 14, 14]
        x = F.max_pool2d(x, 2)         # 输出形状: [batch_size, 64, 7, 7]
        
        # Dropout，防止过拟合
        x = self.dropout(x)
        
        # 展平操作，将特征图转换为一维向量
        x = x.view(-1, 64 * 7 * 7)     # 输出形状: [batch_size, 64*7*7]
        
        # 全连接层1 + ReLU激活
        x = F.relu(self.fc1(x))        # 输出形状: [batch_size, 128]
        
        # 全连接层2（输出层）
        x = self.fc2(x)                # 输出形状: [batch_size, 10]
        
        return x

# 创建模型实例并移动到指定设备
model = FashionCNN().to(device)
print(model)  # 打印模型结构

# ============= 训练设置 =============
# 定义损失函数和优化器
"""
- criterion (损失函数): 用于计算预测值与真实值之间的差距
  - nn.CrossEntropyLoss() 是多分类问题的标准损失函数
  
- optimizer (优化器): 用于更新模型参数
  - optim.Adam 是一种自适应学习率的优化算法
  - model.parameters() 告诉优化器哪些参数需要更新
  - lr (学习率): 控制每次参数更新的步长
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学习率可以调整

# ============= 训练和评估函数 =============
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    """
    训练和评估模型的函数
    
    参数:
    - model: 神经网络模型
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数
    """
    # 用于存储训练过程中的指标
    train_losses = []     # 训练损失
    train_accuracies = [] # 训练准确率
    test_accuracies = []  # 测试准确率
    
    # 训练num_epochs轮
    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        model.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 遍历训练数据集
        for images, labels in train_loader:
            # 将数据移动到指定设备
            images, labels = images.to(device), labels.to(device)
            
            # 1. 清除之前的梯度
            optimizer.zero_grad()
            
            # 2. 前向传播
            outputs = model(images)
            
            # 3. 计算损失
            loss = criterion(outputs, labels)
            
            # 4. 反向传播
            loss.backward()
            
            # 5. 更新参数
            optimizer.step()
            
            # 累计损失
            running_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取最大值的索引
            total += labels.size(0)  # 累计样本总数
            correct += (predicted == labels).sum().item()  # 累计正确预测数量
        
        # 计算当前epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # 存储训练指标
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        # ===== 测试阶段 =====
        model.eval()  # 设置为评估模式
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        # 不计算梯度，加速推理并节省内存
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 获取预测结果
                _, predicted = torch.max(outputs.data, 1)
                
                # 计算测试准确率
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # 收集预测结果和标签，用于后续计算混淆矩阵
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算测试准确率
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        # 打印当前epoch的训练信息
        print(f'轮次 [{epoch+1}/{num_epochs}], 损失: {epoch_loss:.4f}, '
              f'训练准确率: {train_accuracy:.2f}%, 测试准确率: {test_accuracy:.2f}%')
    
    # ===== 训练结束后绘制图表 =====
    
    # 创建一个12x5英寸的图
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'r-', label='训练损失')
    plt.title('训练损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.grid(True)
    plt.legend()
    
    # 绘制训练准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, 'b-', label='训练准确率')
    plt.title('训练准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.grid(True)
    plt.legend()
    
    # 绘制测试准确率曲线
    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies, 'g-', label='测试准确率')
    plt.title('测试准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # ===== 绘制混淆矩阵 =====
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.xticks(rotation=45)  # 旋转类别标签，使其更易读
    plt.tight_layout()
    plt.show()
    
    # 返回训练历史
    return train_losses, train_accuracies, test_accuracies

# ============= 开始训练模型 =============
print("开始训练Fashion-MNIST分类模型...")
history = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)

# ============= 保存模型 =============
# 保存模型权重
torch.save(model.state_dict(), 'fashion_mnist_model.pth')
print("模型已保存到 fashion_mnist_model.pth")

# ============= 可视化一些预测结果 =============
def visualize_predictions(model, test_loader, class_names, num_images=5):
    """
    可视化模型的预测结果
    
    参数:
    - model: 训练好的模型
    - test_loader: 测试数据加载器
    - class_names: 类别名称列表
    - num_images: 要显示的图像数量
    """
    model.eval()  # 设置为评估模式
    
    # 获取一批数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 只选择前num_images个样本
    images = images[:num_images]
    labels = labels[:num_images]
    
    # 将图像移动到指定设备
    images = images.to(device)
    
    # 获取预测结果
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # 将图像和预测结果可视化
    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        # 反归一化图像以正确显示
        img = images[i].cpu().numpy().transpose((1, 2, 0))  # 调整通道顺序为(H,W,C)
        img = img * 0.5 + 0.5  # 反归一化
        img = np.clip(img, 0, 1)
        plt.imshow(img.squeeze(), cmap='gray')  # 灰度图像
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(f"预测: {class_names[preds[i]]}\n真实: {class_names[labels[i]]}", 
                  color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 可视化一些预测结果
print("可视化一些预测结果...")
visualize_predictions(model, test_loader, class_names)