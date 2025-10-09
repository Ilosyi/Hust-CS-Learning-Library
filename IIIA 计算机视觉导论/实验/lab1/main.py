# --- 0. 导入必要的库 ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import os
from datetime import datetime

# ==============================================================================
# --- 1. 实验配置区 ---
#在这里集中修改所有实验参数，方便进行多组对比实验
# ==============================================================================
CONFIG = {
    # --- 数据与模型配置 ---
    "file_path": "dataset.csv",
    # 最佳架构：浅层网络 [32] - 测试准确率99.75% (最高)
    "hidden_layers": [32],
    # 最佳激活函数：ReLU
    "activation_function": nn.ReLU,
    
    # --- 训练参数 ---
    "train_ratio": 0.9,
    # 标准批次大小：32 - 来自最佳实验结果
    "batch_size": 32,
    # 标准学习率：0.001 - 来自最佳实验结果
    "learning_rate": 0.001,
    # 标准训练轮数：15 - 来自最佳实验结果
    "epochs": 15,
    
    # --- 正则化参数 ---
    # 无Dropout：0.0 - 来自最佳实验结果
    "dropout_rate": 0.0,
    
    # --- 结果记录 ---
    "log_file": "experiment_log.csv"
}


# --- 2. 动态神经网络模型定义 ---
# 这个类现在可以根据CONFIG动态生成不同结构的网络
class DynamicFNN(nn.Module):
    """一个可以根据配置动态构建的前馈神经网络，支持Dropout"""
    def __init__(self, input_size, hidden_layers_config, num_classes, activation_fn, dropout_rate=0.0):
        super(DynamicFNN, self).__init__()
        
        # 创建一个用于存放网络层的列表
        layers = []
        current_input_size = input_size
        
        # 遍历配置中的隐藏层，并逐层添加到列表中
        for i, layer_size in enumerate(hidden_layers_config):
            # 添加一个全连接层 (Linear)
            layers.append(nn.Linear(current_input_size, layer_size))
            # 添加指定的激活函数
            layers.append(activation_fn())
            
            # 添加Dropout（除了最后一层）
            if dropout_rate > 0 and i < len(hidden_layers_config) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            # 更新下一层的输入尺寸
            current_input_size = layer_size
            
        # 添加最后的输出层
        layers.append(nn.Linear(current_input_size, num_classes))
        
        # 使用 nn.Sequential 将所有层组合成一个网络模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """定义数据在网络中的前向传播路径"""
        return self.network(x)

# --- 3. 辅助函数 ---
def evaluate_model(model, loader, criterion, device):
    """
    在给定的数据集上评估模型的平均损失和准确率。
    
    Args:
        model (nn.Module): 要评估的模型。
        loader (DataLoader): 包含评估数据的数据加载器。
        criterion: 损失函数。
        device: 计算设备 (cpu 或 cuda)。

    Returns:
        tuple: (平均损失, 准确率)。
    """
    model.eval()  # 将模型设置为评估模式（这会关闭dropout等层）
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估时，我们不需要计算梯度，可以节省计算资源
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            # 前向传播获取模型输出
            outputs = model(data)
            # 计算损失
            loss = criterion(outputs, targets)
            # 累加总损失（乘以批量大小以得到批次总损失）
            total_loss += loss.item() * len(targets)
            # 找到概率最高的类作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 累加样本总数
            total += targets.size(0)
            # 累加预测正确的样本数
            correct += (predicted == targets).sum().item()
    
    # 计算整个数据集的平均损失和准确率
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def log_experiment(config, results, log_file):
    """
    将实验配置和结果记录到CSV文件中。

    Args:
        config (dict): 包含实验超参数的字典。
        results (dict): 包含实验最终性能指标的字典。
        log_file (str): 日志文件名。
    """
    # 将网络结构和激活函数转换为易于阅读的字符串格式
    architecture = f"Input(2)-{'-'.join(map(str, config['hidden_layers']))}-Output(4)"
    activation = config['activation_function'].__name__
    
    # 准备要记录的一行数据
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'architecture': architecture,
        'activation': activation,
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'epochs': config['epochs'],
        'final_train_loss': results['train_loss'],
        'final_train_acc': results['train_acc'],
        'final_test_loss': results['test_loss'],
        'final_test_acc': results['test_acc']
    }
    
    log_df = pd.DataFrame([log_data])
    
    # 如果日志文件不存在，则创建并写入表头
    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False)
    # 否则，追加写入，不包含表头
    else:
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    print(f"\n实验结果已记录到 {log_file}")


# --- 4. 主执行流程 ---
if __name__ == '__main__':
    # --- 数据准备 ---
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 使用pandas加载CSV，假设没有表头
    df = pd.read_csv(CONFIG['file_path'], header=None)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    # 将标签从1-4转换为0-3
    labels = labels - 1

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=1 - CONFIG['train_ratio'], random_state=42, shuffle=True
    )

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch Tensors并移动到指定设备
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建Dataset和DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    # 用于全量评估的加载器
    full_train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

    # --- 模型初始化 ---
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(labels))
    
    # 根据CONFIG实例化动态模型
    model = DynamicFNN(
        input_size=input_dim, 
        hidden_layers_config=CONFIG['hidden_layers'],
        num_classes=output_dim,
        activation_fn=CONFIG['activation_function'],
        dropout_rate=CONFIG.get('dropout_rate', 0.0)
    ).to(device)
    
    print("\n--- 神经网络架构 ---")
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # --- 训练与评估 ---
    # 创建列表用于存储历史数据以供后续可视化
    history = {
        'steps': [],
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': []
    }
    
    print("\n--- 开始训练 ---")
    global_step = 0
    for epoch in range(CONFIG['epochs']):
        model.train() # 确保模型处于训练模式
        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_test_loss = 0
        epoch_test_acc = 0
        batch_count = 0
        
        for i, (batch_features, batch_labels) in enumerate(train_loader):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            # 前向传播
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # === 在每个mini-batch训练后，在完整的训练集和测试集上评估 ===
            train_loss, train_acc = evaluate_model(model, full_train_loader, criterion, device)
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            
            # 记录数据用于绘制minibatch曲线
            history['steps'].append(global_step)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            # 累加用于计算epoch平均值
            epoch_train_loss += train_loss
            epoch_train_acc += train_acc
            epoch_test_loss += test_loss
            epoch_test_acc += test_acc
            batch_count += 1
            
            global_step += 1
        
        # 计算epoch平均值并输出
        avg_train_loss = epoch_train_loss / batch_count
        avg_train_acc = epoch_train_acc / batch_count
        avg_test_loss = epoch_test_loss / batch_count
        avg_test_acc = epoch_test_acc / batch_count
        
        print(f'Epoch [{epoch+1:2d}/{CONFIG["epochs"]}] | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | '
              f'Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.2f}%')

    print("--- 训练完成 ---\n")

    # --- 最终评估与记录 ---
    final_train_loss, final_train_acc = evaluate_model(model, full_train_loader, criterion, device)
    final_test_loss, final_test_acc = evaluate_model(model, test_loader, criterion, device)

    print("--- 最终评估结果 ---")
    print(f'最终训练集 -> 损失: {final_train_loss:.4f}, 准确率: {final_train_acc:.2f}%')
    print(f'最终测试集 -> 损失: {final_test_loss:.4f}, 准确率: {final_test_acc:.2f}%')

    # 将本次实验结果记录到日志文件
    final_results = {
        'train_loss': final_train_loss, 'train_acc': final_train_acc,
        'test_loss': final_test_loss, 'test_acc': final_test_acc
    }
    log_experiment(CONFIG, final_results, CONFIG['log_file'])

    # --- 5. 结果可视化 ---
    print("\n正在生成可视化图表...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制损失曲线 (Minibatch级别)
    ax1.plot(history['steps'], history['train_loss'], label='Train Loss', alpha=0.7, linewidth=1)
    ax1.plot(history['steps'], history['test_loss'], label='Test Loss', alpha=0.7, linewidth=1)
    ax1.set_title('Loss vs Mini-batch Steps\n(Best Config: [32] layers, LR=0.001, Dropout=0.0)')
    ax1.set_xlabel('Mini-batch Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)  # 损失从0开始显示

    # 绘制准确率曲线 (Minibatch级别)
    ax2.plot(history['steps'], history['train_acc'], label='Train Accuracy', alpha=0.7, linewidth=1)
    ax2.plot(history['steps'], history['test_acc'], label='Test Accuracy', alpha=0.7, linewidth=1)
    ax2.set_title('Accuracy vs Mini-batch Steps\n(Final: Train={:.1f}%, Test={:.1f}%)'.format(
        final_train_acc, final_test_acc))
    ax2.set_xlabel('Mini-batch Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(95, 100)  # 准确率从95%开始显示，更好地展示变化

    plt.tight_layout()
    plt.show()
    
    # 输出训练总结
    print(f"\n📊 训练总结:")
    print(f"   架构: Input(2) → {CONFIG['hidden_layers']} → Output(4)")
    print(f"   优化器: Adam (lr={CONFIG['learning_rate']})")
    print(f"   正则化: Dropout({CONFIG.get('dropout_rate', 0.0)})")
    print(f"   批次大小: {CONFIG['batch_size']}")
    print(f"   训练轮数: {CONFIG['epochs']}")
    print(f"   最终测试准确率: {final_test_acc:.2f}%")
    print(f"   总训练步数: {len(history['steps'])}")
