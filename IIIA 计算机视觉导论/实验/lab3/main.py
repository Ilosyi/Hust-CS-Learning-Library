# ============================
# 导入必要的库
# ============================
import torch  # PyTorch核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器
from torchvision import datasets, transforms  # 计算机视觉工具和数据转换
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
import seaborn as sns  # 统计可视化库
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score  # 评估指标
import json  # JSON处理
import os  # 操作系统接口
from pathlib import Path  # 路径处理
from datetime import datetime  # 日期时间处理
from typing import Optional, Dict, Any, List, Tuple  # 类型注解

# 检查PyTorch是否支持混合精度训练(AMP)
# AMP可以加速训练并减少显存使用
HAS_TORCH_AMP = hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast')
HAS_TORCH_GRADSCALER = HAS_TORCH_AMP and hasattr(torch.amp, 'GradScaler')

import random
import copy

# ============================
# PyTorch性能优化设置
# ============================
# 启用cudnn benchmark模式,自动寻找最优卷积算法
torch.backends.cudnn.benchmark = True

# 如果有CUDA(GPU),启用TF32格式以加速矩阵运算
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # 矩阵乘法使用TF32
        torch.backends.cudnn.allow_tf32 = True  # cuDNN使用TF32
    except AttributeError:
        pass  # 旧版本PyTorch可能不支持此设置

# 设置浮点数矩阵乘法精度为medium(平衡精度和速度)
try:
    torch.set_float32_matmul_precision('medium')
except AttributeError:
    pass  # 旧版本PyTorch不支持

# ============================
# MNIST数据集的标准化参数
# ============================
# 这些值是MNIST数据集的统计特征
MNIST_MEAN = 0.1307  # 像素均值
MNIST_STD = 0.3081   # 像素标准差

# ============================
# 数据加载器的工作进程数设置
# ============================
# 多进程加载数据可以加速训练
DEFAULT_NUM_WORKERS = 0  # 默认不使用多进程
cpu_count = os.cpu_count()  # 获取CPU核心数
if cpu_count and cpu_count > 2:
    # 如果CPU核心数大于2,使用2到8个工作进程
    # 保留2个核心给主进程使用
    DEFAULT_NUM_WORKERS = max(2, min(8, cpu_count - 2))

# ============================
# 配置参数字典
# ============================
# 所有超参数集中管理,方便调整和实验
CONFIG = {
    'experiment_name': 'mnist_siamese_cnn',  # 实验名称
    'batch_size': 256,  # 批次大小,每次训练处理256对图片
    'epochs': 25,  # 训练轮数
    'learning_rate': 8e-4,  # 学习率(0.0008)
    'weight_decay': 1e-4,  # L2正则化系数,防止过拟合
    'train_ratio': 0.1,  # 使用10%的MNIST训练集(6000张图片)
    'test_ratio': 0.1,   # 使用10%的MNIST测试集(1000张图片)
    'pairs_per_epoch': 12000,  # 每个epoch生成12000对样本
    'seed': 42,  # 随机种子,确保实验可重复
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 自动选择GPU或CPU
    'embedding_dim': 128,  # 图像特征向量的维度
    'max_grad_norm': 1.0,  # 梯度裁剪阈值,防止梯度爆炸
    'use_amp': True,  # 是否使用混合精度训练
    'scheduler_factor': 0.5,  # 学习率衰减因子
    'scheduler_patience': 2,  # 学习率调度器的耐心值
    'min_lr': 1e-5,  # 最小学习率
    'num_workers': DEFAULT_NUM_WORKERS,  # 数据加载的工作进程数
    'augment': True,  # 是否使用数据增强
    'fast_dev_run': False,  # 快速开发模式(用于调试)
    'prefetch_factor': 4,  # 每个worker预取的batch数量
    'persistent_workers': True,  # 保持workers活跃,避免重启开销
    'compile_model': False  # 是否使用torch.compile加速(PyTorch 2.0+)
}


def set_global_seeds(seed: int) -> None:
    """
    设置所有随机数生成器的种子,确保实验结果可重复
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)  # Python内置随机数
    np.random.seed(seed)  # NumPy随机数
    torch.manual_seed(seed)  # PyTorch CPU随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU随机数


# 在程序开始时设置随机种子
set_global_seeds(CONFIG['seed'])


# ============================
# 快速开发模式设置
# ============================
# 通过环境变量启用快速测试模式
FAST_DEV_RUN = os.environ.get("FAST_DEV_RUN", "").lower() in ("1", "true", "yes")
if FAST_DEV_RUN:
    print("⚡ FAST_DEV_RUN enabled: using a reduced configuration for smoke testing.")
    # 使用极小的配置进行快速测试
    CONFIG.update({
        'epochs': 1,  # 只训练1轮
        'pairs_per_epoch': 256,  # 只生成256对
        'batch_size': 32,  # 小批次
        'train_ratio': 0.02,  # 只用2%的数据
        'test_ratio': 0.02,
        'fast_dev_run': True,
        'num_workers': 0,  # 单进程
        'prefetch_factor': 2,
        'persistent_workers': False
    })


def _get_subset_labels(subset) -> np.ndarray:
    """
    从数据集子集中提取所有样本的标签
    
    参数:
        subset: 数据集或数据集子集
        
    返回:
        包含所有标签的numpy数组
    """
    # 如果是Subset对象(从完整数据集中抽取的子集)
    if isinstance(subset, torch.utils.data.Subset):
        base_dataset = subset.dataset  # 获取原始完整数据集
        indices = np.array(list(subset.indices))  # 子集包含的样本索引
        
        # 尝试获取标签(不同数据集的属性名可能不同)
        if hasattr(base_dataset, 'targets'):
            targets = base_dataset.targets
        elif hasattr(base_dataset, 'labels'):
            targets = base_dataset.labels
        else:
            # 如果没有直接的标签属性,逐个获取
            targets = [int(base_dataset[i][1]) for i in indices]

        # 统一转换为numpy数组
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        else:
            targets = np.array(targets)

        return targets[indices]  # 返回子集对应的标签
    else:
        # 如果是普通数据集,直接逐个获取标签
        return np.array([int(subset[i][1]) for i in range(len(subset))])


def _format_ratio_tag(ratio: float) -> str:
    """
    将比例值转换为固定格式的字符串标签
    
    例如: 0.1 -> "01000" (表示10.00%)
    
    参数:
        ratio: 0到1之间的比例值
        
    返回:
        5位数字的字符串
    """
    return f"{int(round(ratio * 10000)):05d}"


def load_or_create_indices(
    kind: str,  # 'train' 或 'test'
    dataset_length: int,  # 完整数据集的长度
    ratio: float,  # 要抽取的比例
    seed: int,  # 随机种子
    cache_root: Path,  # 缓存文件保存路径
    expected_size: Optional[int] = None  # 期望的子集大小
) -> np.ndarray:
    """
    加载或创建数据集索引,并缓存到文件中
    
    这个函数确保每次运行使用相同的数据划分,避免结果不一致
    
    参数:
        kind: 数据集类型('train'或'test')
        dataset_length: 完整数据集的样本数量
        ratio: 要抽取的比例(0-1之间)
        seed: 随机种子,确保可重复性
        cache_root: 缓存文件的保存目录
        expected_size: 期望的子集大小(如果为None则根据ratio计算)
        
    返回:
        包含随机选择的样本索引的numpy数组
    """
    # 创建缓存目录
    cache_root.mkdir(parents=True, exist_ok=True)
    
    # 生成缓存文件名,包含所有关键参数
    ratio_tag = _format_ratio_tag(ratio)
    cache_file = cache_root / f"{kind}_len{dataset_length}_ratio{ratio_tag}_seed{seed}.npy"

    # 如果缓存文件存在,尝试加载
    if cache_file.exists():
        indices = np.load(cache_file)
        
        # 验证缓存的索引是否有效
        if indices.size == 0:
            raise ValueError(f"Cached indices in {cache_file} are empty.")
        
        # 如果大小不匹配,删除缓存并重新生成
        if expected_size is not None and indices.shape[0] != expected_size:
            cache_file.unlink()
            print(f"Regenerating {kind} indices due to size mismatch (expected {expected_size}, found {indices.shape[0]}).")
        else:
            return indices  # 返回缓存的索引

    # 生成新的随机索引
    rng = np.random.default_rng(seed)  # 创建随机数生成器
    subset_size = expected_size if expected_size is not None else max(int(dataset_length * ratio), 1)
    # 从0到dataset_length-1中随机选择subset_size个不重复的索引
    indices = rng.choice(dataset_length, subset_size, replace=False).astype(np.int64)
    
    # 保存到缓存文件
    np.save(cache_file, indices)
    print(f"Saved fixed {kind} indices to {cache_file} (size={subset_size}).")
    return indices


def seed_worker(worker_id: int) -> None:
    """
    为DataLoader的每个worker进程设置独立的随机种子
    
    这确保多进程数据加载时,每个进程的随机性是可控且不同的
    
    参数:
        worker_id: worker进程的ID编号
    """
    base_seed = CONFIG['seed']
    worker_seed = base_seed + worker_id  # 每个worker使用不同的种子
    
    # 设置所有随机数生成器
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # 如果dataset是MNISTPairsDataset,也设置它的随机数生成器
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and isinstance(worker_info.dataset, MNISTPairsDataset):
        worker_info.dataset.rng = np.random.default_rng(worker_seed)


# ============================
# 数据集构建
# ============================
class MNISTPairsDataset(Dataset):
    """
    生成MNIST图片对的数据集
    
    这个数据集的核心功能:
    1. 随机生成图片对(同类或异类)
    2. 50%的概率生成同类对(标签为1)
    3. 50%的概率生成异类对(标签为0)
    
    Siamese网络的关键:学习判断两张图片是否属于同一类别
    """

    def __init__(self, mnist_dataset, num_pairs: int = 6000, seed: Optional[int] = None):
        """
        初始化图片对数据集
        
        参数:
            mnist_dataset: 基础的MNIST数据集
            num_pairs: 要生成的图片对数量
            seed: 随机种子
        """
        self.mnist_dataset = mnist_dataset
        self.num_pairs = num_pairs
        self.seed = seed if seed is not None else np.random.randint(0, 1_000_000)
        self.rng = np.random.default_rng(self.seed)  # 随机数生成器

        # 获取所有样本的标签
        labels = _get_subset_labels(mnist_dataset)
        
        # 构建标签到索引的映射字典
        # 例如: {0: [3, 15, 27, ...], 1: [2, 8, 19, ...], ...}
        # 这样可以快速找到某个类别的所有样本
        self.label_to_indices: Dict[int, np.ndarray] = {}
        for idx, label in enumerate(labels.tolist()):
            self.label_to_indices.setdefault(label, []).append(idx)

        # 过滤掉没有样本的类别,并转换为numpy数组
        self.label_to_indices = {
            label: np.array(indices, dtype=np.int64)
            for label, indices in self.label_to_indices.items()
            if len(indices) > 0
        }

        # same_labels: 至少有2个样本的类别(可以形成同类对)
        self.same_labels = [label for label, indices in self.label_to_indices.items() if len(indices) > 1]
        # diff_labels: 所有有样本的类别
        self.diff_labels = list(self.label_to_indices.keys())

        # 至少需要2个不同类别才能生成异类对
        if len(self.diff_labels) < 2:
            raise ValueError("Dataset must contain at least two distinct digit classes to form pairs.")

    def __len__(self) -> int:
        """返回数据集的大小(样本对数量)"""
        return self.num_pairs

    def _sample_same_pair(self) -> Tuple[int, int, float]:
        """
        生成同类样本对
        
        返回:
            (索引1, 索引2, 标签1.0)
        """
        # 随机选择一个类别
        label = int(self.rng.choice(self.same_labels))
        # 从该类别中随机选择两个不同的样本
        idx1, idx2 = self.rng.choice(self.label_to_indices[label], size=2, replace=False)
        return int(idx1), int(idx2), 1.0  # 标签为1表示"相同"

    def _sample_diff_pair(self) -> Tuple[int, int, float]:
        """
        生成异类样本对
        
        返回:
            (索引1, 索引2, 标签0.0)
        """
        # 随机选择两个不同的类别
        label1, label2 = self.rng.choice(self.diff_labels, size=2, replace=False)
        # 分别从两个类别中各选一个样本
        idx1 = self.rng.choice(self.label_to_indices[int(label1)])
        idx2 = self.rng.choice(self.label_to_indices[int(label2)])
        return int(idx1), int(idx2), 0.0  # 标签为0表示"不同"

    def __getitem__(self, index: int):
        """
        获取一个样本对
        
        注意: index参数实际未被使用,因为我们是随机生成样本对
        这样设计是为了与PyTorch的DataLoader兼容
        
        返回:
            (图片1, 图片2, 标签)
        """
        # 50%概率生成同类对,50%概率生成异类对
        use_same = bool(self.same_labels) and (self.rng.random() < 0.5)
        
        if use_same:
            idx1, idx2, target = self._sample_same_pair()
        else:
            idx1, idx2, target = self._sample_diff_pair()

        # 从基础数据集中获取实际的图片
        img1, _ = self.mnist_dataset[idx1]  # _表示我们不需要原始标签
        img2, _ = self.mnist_dataset[idx2]
        
        # 将目标标签转换为PyTorch张量
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return img1, img2, target_tensor


# ============================
# 神经网络架构
# ============================
class SiameseCompareNet(nn.Module):
    """
    Siamese神经网络,用于比较两张图片是否相似
    
    网络结构:
    1. 特征提取器: 共享的卷积神经网络,将28x28的图片编码为128维向量
    2. 投影层: 将卷积特征映射到嵌入空间
    3. 比较层: 接收两个嵌入向量,输出相似度分数
    
    核心思想:
    - 两张图片通过**相同**的特征提取器(权重共享)
    - 比较它们的特征向量来判断是否属于同一类别
    """

    def __init__(self, embedding_dim: int = 128):
        """
        初始化网络
        
        参数:
            embedding_dim: 嵌入向量的维度
        """
        super().__init__()

        # 特征提取器: 3个卷积块
        self.feature_extractor = nn.Sequential(
            # 第一个卷积块: 1通道 -> 32通道
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # 输入1通道(灰度图),输出32通道
            nn.BatchNorm2d(32),  # 批归一化,加速训练并提高稳定性
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(2),  # 2x2最大池化,将28x28降为14x14

            # 第二个卷积块: 32通道 -> 64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7

            # 第三个卷积块: 64通道 -> 128通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 自适应平均池化,输出1x1,得到128维特征向量
        )

        # 投影层: 将卷积特征投影到嵌入空间
        self.projection = nn.Sequential(
            nn.Flatten(),  # 展平为一维向量
            nn.Linear(128, embedding_dim),  # 线性变换到目标维度
            nn.ReLU(inplace=True)
        )

        # 分类器: 比较两个嵌入向量
        # 输入是两个embedding_dim维向量的组合(2 * embedding_dim维)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),  # 第一层全连接
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # Dropout防止过拟合
            nn.Linear(256, 64),  # 第二层全连接
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # 输出1个值(相似度分数)
        )

        # 初始化网络权重
        self._initialize_weights()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        将图片编码为嵌入向量
        
        参数:
            x: 输入图片张量 [batch_size, 1, 28, 28]
            
        返回:
            嵌入向量 [batch_size, embedding_dim]
        """
        features = self.feature_extractor(x)  # 提取卷积特征
        embedding = self.projection(features)  # 投影到嵌入空间
        return embedding

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        前向传播:比较两张图片
        
        参数:
            img1: 第一张图片 [batch_size, 1, 28, 28]
            img2: 第二张图片 [batch_size, 1, 28, 28]
            
        返回:
            相似度分数 [batch_size] (使用sigmoid后值在0-1之间)
        """
        # 分别编码两张图片
        embedding1 = self._encode(img1)
        embedding2 = self._encode(img2)
        
        # 构造比较特征
        # 使用两种方式组合嵌入向量:
        # 1. 绝对差值: 衡量距离
        # 2. 元素乘积: 衡量相似性
        features = torch.cat([
            torch.abs(embedding1 - embedding2),  # 特征差异
            embedding1 * embedding2  # 特征相似性
        ], dim=1)
        
        # 通过分类器得到最终分数
        logits = self.classifier(features)
        return logits.squeeze(dim=1)  # 移除最后一维,返回[batch_size]

    def _initialize_weights(self) -> None:
        """
        初始化网络权重
        
        使用合适的初始化方法可以加速训练收敛
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # 卷积层使用Kaiming初始化(适合ReLU)
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # 全连接层使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # 批归一化层
                nn.init.ones_(module.weight)  # gamma初始化为1
                nn.init.zeros_(module.bias)   # beta初始化为0


# ============================
# 训练和评估函数
# ============================
def get_current_lr(optimizer: optim.Optimizer) -> float:
    """
    获取优化器当前的学习率
    
    参数:
        optimizer: PyTorch优化器
        
    返回:
        当前学习率
    """
    return optimizer.param_groups[0]['lr']


def train_epoch(
    model: nn.Module,  # 要训练的模型
    dataloader: DataLoader,  # 训练数据加载器
    criterion: nn.Module,  # 损失函数
    optimizer: optim.Optimizer,  # 优化器
    device: torch.device,  # 计算设备(CPU/GPU)
    scaler: Optional[torch.cuda.amp.GradScaler] = None,  # 混合精度训练的缩放器
    max_grad_norm: Optional[float] = None,  # 梯度裁剪阈值
    epoch: int = 1,  # 当前轮数
    total_epochs: int = 1,  # 总轮数
    show_progress: bool = False  # 是否显示进度
) -> Tuple[float, float]:
    """
    训练一个epoch
    
    返回:
        (平均损失, 准确率)
    """
    model.train()  # 设置为训练模式(启用dropout等)
    running_loss = 0.0  # 累计损失
    all_preds = []  # 所有预测结果
    all_targets = []  # 所有真实标签
    
    # 检查是否使用混合精度训练
    amp_enabled = scaler is not None and scaler.is_enabled()

    num_batches = len(dataloader)
    last_display = -1
    progress_step = 5
    
    # 设置进度显示
    if show_progress and num_batches > 0:
        progress_step = max(1, 100 // min(100, num_batches))
        print(f"Epoch {epoch}/{total_epochs} progress: 0%", end='\r', flush=True)

    # 遍历所有batch
    for batch_idx, (imgs1, imgs2, targets) in enumerate(dataloader):
        # 将数据移动到GPU(如果可用)
        # non_blocking=True 允许异步数据传输
        imgs1 = imgs1.to(device, non_blocking=True)
        imgs2 = imgs2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 清空梯度
        # set_to_none=True 比 zero_grad() 更高效
        optimizer.zero_grad(set_to_none=True)

        # 如果使用混合精度训练
        if amp_enabled and device.type == 'cuda':
            # 自动混合精度上下文
            autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if HAS_TORCH_AMP else torch.cuda.amp.autocast()
            with autocast_ctx:
                # 前向传播
                logits = model(imgs1, imgs2)
                loss = criterion(logits, targets)
            
            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪(防止梯度爆炸)
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)  # 恢复梯度的真实值
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()  # 更新缩放因子
        else:
            # 不使用混合精度的标准训练流程
            logits = model(imgs1, imgs2)
            loss = criterion(logits, targets)
            loss.backward()  # 反向传播
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()  # 更新参数

        # 累计损失
        running_loss += loss.detach().item() * imgs1.size(0)

        # 计算预测结果
        probs = torch.sigmoid(logits)  # 将logits转换为概率(0-1之间)
        preds = (probs > 0.5).float()  # 概率>0.5预测为1(相同),否则为0(不同)
        
        # 记录预测和真实标签
        all_preds.extend(preds.cpu().numpy().astype(int).tolist())
        all_targets.extend(targets.cpu().numpy().astype(int).tolist())

        # 显示训练进度
        if show_progress and num_batches > 0:
            progress = int(((batch_idx + 1) / num_batches) * 100)
            if progress - last_display >= progress_step or progress == 100:
                print(
                    f"Epoch {epoch}/{total_epochs} progress: {progress:3d}%",
                    end='\r' if progress < 100 else '\n',
                    flush=True
                )
                last_display = progress
    # 计算整个epoch的平均损失
    epoch_loss = running_loss / len(dataloader.dataset)
    # 计算整个epoch的准确率
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,  # 要评估的模型
    dataloader: DataLoader,  # 测试数据加载器
    criterion: nn.Module,  # 损失函数
    device: torch.device  # 计算设备
) -> Tuple[float, float, List[int], List[int], float]:
    """
    评估模型性能
    
    返回:
        (平均损失, 准确率, 预测列表, 真实标签列表, ROC-AUC分数)
    """
    model.eval()  # 设置为评估模式(禁用dropout、BatchNorm使用移动平均等)
    running_loss = 0.0
    all_preds = []  # 存储所有预测的类别(0或1)
    all_targets = []  # 存储所有真实标签
    all_probs = []  # 存储所有预测概率(用于计算AUC)

    # 评估时不需要计算梯度,可以节省内存和计算
    with torch.no_grad():
        for imgs1, imgs2, targets in dataloader:
            # 将数据移到设备上
            imgs1 = imgs1.to(device, non_blocking=True)
            imgs2 = imgs2.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 前向传播
            logits = model(imgs1, imgs2)
            loss = criterion(logits, targets)

            # 累计损失
            running_loss += loss.item() * imgs1.size(0)

            # 计算预测概率和类别
            probs = torch.sigmoid(logits)  # 转换为概率
            preds = (probs > 0.5).float()  # 阈值0.5分类

            # 记录结果(移到CPU并转换为Python列表)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().astype(int).tolist())
            all_targets.extend(targets.cpu().numpy().astype(int).tolist())

    # 计算平均损失
    epoch_loss = running_loss / len(dataloader.dataset)
    # 计算准确率
    epoch_acc = accuracy_score(all_targets, all_preds)

    # 计算ROC-AUC分数(需要至少有两个类别)
    if len(set(all_targets)) > 1:
        test_auc = roc_auc_score(all_targets, all_probs)
    else:
        # 如果只有一个类别,AUC无法计算
        test_auc = float('nan')

    return epoch_loss, epoch_acc, all_preds, all_targets, test_auc


# ============================
# 可视化与辅助函数
# ============================
def sanitize_history(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理训练历史中的NaN值,确保可以保存为JSON
    
    JSON不支持NaN,需要转换为None
    
    参数:
        history: 训练历史字典
        
    返回:
        清理后的字典
    """
    cleaned: Dict[str, Any] = {}
    for key, values in history.items():
        if isinstance(values, list):
            # 将列表中的NaN替换为None
            cleaned[key] = [None if isinstance(v, float) and np.isnan(v) else v for v in values]
        else:
            # 处理单个值
            cleaned[key] = None if isinstance(values, float) and np.isnan(values) else values
    return cleaned


def sanitize_metric(value: Optional[float]) -> Optional[float]:
    """
    清理单个指标值中的NaN
    
    参数:
        value: 指标值
        
    返回:
        清理后的值(NaN转为None)
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


def plot_training_curves(history: Dict[str, list], save_path: str) -> None:
    """
    绘制训练过程的曲线图
    
    参数:
        history: 包含训练历史的字典
        save_path: 图片保存路径
    """
    # 检查是否有AUC数据
    has_auc = bool('test_auc' in history and any(v is not None for v in history['test_auc']))
    
    # 检查是否有学习率数据
    lr_values = history.get('lr', [])
    has_lr = isinstance(lr_values, (list, tuple)) and len(lr_values) > 0
    
    # 计算需要绘制的子图数量
    # 基础: 损失曲线和准确率曲线(2个)
    # 可选: AUC曲线和学习率曲线
    num_plots = 2 + int(has_auc) + int(has_lr)

    # 创建子图
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    axes = np.atleast_1d(axes)

    # 第一个子图: 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['test_loss'], label='Test Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)

    # 第二个子图: 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['test_acc'], label='Test Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True)

    axis_idx = 2
    # 如果有AUC数据,绘制AUC曲线
    if has_auc:
        axes[axis_idx].plot([
            v if v is not None else np.nan for v in history['test_auc']
        ], label='Test AUC', marker='^')
        axes[axis_idx].set_xlabel('Epoch')
        axes[axis_idx].set_ylabel('ROC-AUC')
        axes[axis_idx].set_title('Test ROC-AUC')
        axes[axis_idx].grid(True)
        axes[axis_idx].legend()
        axis_idx += 1

    # 如果有学习率数据,绘制学习率曲线
    if has_lr:
        axes[axis_idx].plot(lr_values, label='Learning Rate', marker='d')
        axes[axis_idx].set_xlabel('Epoch')
        axes[axis_idx].set_ylabel('LR')
        axes[axis_idx].set_title('Learning Rate Schedule')
        axes[axis_idx].grid(True)
        axes[axis_idx].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    """
    绘制混淆矩阵
    
    混淆矩阵显示模型预测的详细情况:
    - 真阳性(TP): 正确预测为"相同"
    - 真阴性(TN): 正确预测为"不同"
    - 假阳性(FP): 错误预测为"相同"
    - 假阴性(FN): 错误预测为"不同"
    
    参数:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        save_path: 图片保存路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Different', 'Same'],  # 0表示不同,1表示相同
                yticklabels=['Different', 'Same'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    反归一化图片,用于可视化
    
    将标准化后的图片还原为原始像素值范围[0, 1]
    
    参数:
        img_tensor: 标准化后的图片张量
        
    返回:
        反归一化后的numpy数组
    """
    img = img_tensor.detach().cpu().numpy()
    # 反归一化公式: img = img * std + mean
    img = img * MNIST_STD + MNIST_MEAN
    # 裁剪到[0, 1]范围
    return np.clip(img, 0.0, 1.0)


def visualize_samples(
    dataset: Dataset,  # 数据集
    model: nn.Module,  # 模型
    device: torch.device,  # 计算设备
    save_path: str,  # 保存路径
    num_samples: int = 8  # 要可视化的样本数
) -> None:
    """
    可视化模型的预测结果
    
    将图片对、真实标签、预测标签和预测概率一起显示
    
    参数:
        dataset: 要可视化的数据集
        model: 训练好的模型
        device: 计算设备
        save_path: 图片保存路径
        num_samples: 要显示的样本数量
    """
    model.eval()  # 设置为评估模式
    
    # 设置子图布局(2行,每行显示一半样本)
    rows = 2
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    # 实际显示的样本数量(不超过数据集大小和子图数量)
    count = min(num_samples, len(axes), len(dataset))

    with torch.no_grad():
        for idx in range(count):
            # 获取一个样本对
            img1, img2, target = dataset[idx]
            
            # 添加batch维度并移到设备上
            img1_batch = img1.unsqueeze(0).to(device)
            img2_batch = img2.unsqueeze(0).to(device)

            # 模型预测
            logit = model(img1_batch, img2_batch)
            prob = torch.sigmoid(logit).item()  # 转换为概率
            pred = int(prob > 0.5)  # 0.5阈值分类

            # 反归一化图片用于显示
            img1_disp = _denormalize(img1.squeeze())
            img2_disp = _denormalize(img2.squeeze())
            # 将两张图片水平拼接
            combined = np.hstack([img1_disp, img2_disp])

            # 显示图片
            axes[idx].imshow(combined, cmap='gray', vmin=0.0, vmax=1.0)
            axes[idx].axis('off')
            
            # 设置标题,显示真实标签、预测标签和概率
            # 预测正确时标题为绿色,错误时为红色
            axes[idx].set_title(
                f"True:{int(target.item())} Pred:{pred}\nProb:{prob:.2f}",
                color='green' if pred == int(target.item()) else 'red'
            )

    # 隐藏多余的子图
    for j in range(count, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_last_conv_activation_stats(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统计最后一层卷积(ReLU之后)在整个数据集上的平均激活

    返回:
        (平均特征图 [C, H, W], 每个通道的平均激活值 [C])
    """
    model.eval()
    activation_sum: Optional[torch.Tensor] = None
    sample_count = 0

    def hook(_: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal activation_sum, sample_count
        activations = output.detach()
        batch_size = activations.shape[0]
        batch_sum = activations.sum(dim=0).to(dtype=torch.float64)
        if activation_sum is None:
            activation_sum = torch.zeros_like(batch_sum, dtype=torch.float64, device=activations.device)
        activation_sum += batch_sum
        sample_count += batch_size

    handle = model.feature_extractor[10].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for batch in dataloader:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                imgs = imgs.to(device, non_blocking=True)
                _ = model._encode(imgs)
    finally:
        handle.remove()

    if activation_sum is None or sample_count == 0:
        raise RuntimeError("无法统计最后一层卷积的激活值,请检查数据加载流程。")

    mean_maps = (activation_sum / sample_count).cpu().numpy().astype(np.float32)
    channel_mean = mean_maps.reshape(mean_maps.shape[0], -1).mean(axis=1)
    return mean_maps, channel_mean


def plot_mean_feature_maps(mean_feature_maps: np.ndarray, save_path: str, num_cols: int = 16) -> None:
    """绘制最后一层卷积的平均特征图网格"""
    num_channels = mean_feature_maps.shape[0]
    num_cols = max(1, num_cols)
    num_rows = int(np.ceil(num_channels / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.6, num_rows * 1.6))
    axes = np.atleast_1d(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx < num_channels:
            ax.imshow(mean_feature_maps[idx], cmap='viridis')
            ax.set_title(f"Ch {idx}", fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def prune_last_conv_channels(model: nn.Module, channels: np.ndarray) -> None:
    """
    将最后一层卷积的指定通道权重置零,模拟剪枝
    """
    if len(channels) == 0:
        return

    conv_layer = model.feature_extractor[8]
    bn_layer = model.feature_extractor[9]
    idx_tensor = torch.as_tensor(channels, dtype=torch.long, device=conv_layer.weight.device)

    with torch.no_grad():
        conv_layer.weight[idx_tensor] = 0
        if conv_layer.bias is not None:
            conv_layer.bias[idx_tensor] = 0
        if isinstance(bn_layer, nn.BatchNorm2d):
            bn_layer.weight[idx_tensor] = 0
            bn_layer.bias[idx_tensor] = 0
            bn_layer.running_mean[idx_tensor] = 0
            bn_layer.running_var[idx_tensor] = 1


def evaluate_pruning_sensitivity(
    model: nn.Module,
    test_loader: DataLoader,
    channel_order: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    embedding_dim: int
) -> List[Dict[str, float]]:
    """
    评估剪枝数量K与分类准确率之间的关系 (K=0..P-1)
    """
    base_state = copy.deepcopy(model.state_dict())
    num_channels = len(channel_order)
    results: List[Dict[str, float]] = []

    # baseline: 未剪枝(K=0)
    baseline_model = model.__class__(embedding_dim=embedding_dim).to(device)
    baseline_model.load_state_dict(base_state)
    _, base_acc, _, _, _ = evaluate(baseline_model, test_loader, criterion, device)
    results.append({'k': 0, 'accuracy': float(base_acc)})

    # 剪枝K=1..P-1
    for k in range(1, num_channels):
        pruned_model = model.__class__(embedding_dim=embedding_dim).to(device)
        pruned_model.load_state_dict(base_state)
        prune_last_conv_channels(pruned_model, channel_order[:k])
        _, acc, _, _, _ = evaluate(pruned_model, test_loader, criterion, device)
        results.append({'k': int(k), 'accuracy': float(acc)})

    return results


def plot_pruning_curve(results: List[Dict[str, float]], save_path: str) -> None:
    """绘制剪枝数量K与准确率的关系曲线"""
    ks = [item['k'] for item in results]
    accs = [item['accuracy'] for item in results]

    # 输出并保存坐标数据,方便报告分析
    data_path = Path(save_path).with_suffix('.csv')
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write('k,accuracy\n')
        print("Pruning curve data (k, accuracy):")
        for k_value, acc_value in zip(ks, accs):
            line = f"{k_value},{acc_value:.6f}"
            print(line)
            f.write(line + '\n')
    print(f"剪枝曲线数据已保存: {data_path}")

    plt.figure(figsize=(7, 5))
    plt.plot(ks, accs, marker='o', markersize=3, linewidth=1.2)
    plt.xlabel('K (Pruned Channels)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Pruned Channels')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================
# 主训练流程
# ============================
def main():
    """
    主函数:执行完整的训练流程
    
    流程:
    1. 初始化设备和配置
    2. 加载和预处理数据
    3. 构建模型
    4. 训练模型
    5. 评估和可视化结果
    6. 保存模型和报告
    """
    # 设置计算设备(优先使用GPU)
    device = torch.device(CONFIG['device'])

    # 打印欢迎信息
    print("=" * 70)
    print("MNIST Siamese CNN Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Configuration: {json.dumps(CONFIG, indent=2)}")
    print("=" * 70)

    # 显示GPU信息
    if device.type == 'cuda':
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        print(f"Detected GPU: {gpu_name}")
    else:
        print("⚠️ 未检测到可用的CUDA设备,将使用CPU进行训练。如需使用GPU,请确认已安装CUDA版本的 PyTorch 并正确配置驱动。")

    # 创建实验目录(用于保存所有结果)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{CONFIG['experiment_name']}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # 保存配置文件
    with open(f"{exp_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    # ============================
    # 构建数据转换pipeline
    # ============================
    # 训练集数据增强(如果启用)
    if CONFIG['augment']:
        train_transform = transforms.Compose([
            # 随机仿射变换:旋转、平移、缩放、剪切
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=12,  # 随机旋转±12度
                    translate=(0.12, 0.12),  # 随机平移±12%
                    scale=(0.9, 1.1),  # 随机缩放90%-110%
                    shear=8  # 随机剪切±8度
                )
            ], p=0.7),  # 70%概率应用
            # 随机高斯模糊
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.2),  # 20%概率应用
            transforms.ToTensor(),  # 转换为张量[0,1]
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))  # 标准化
        ])
    else:
        # 不使用数据增强
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
        ])

    # 测试集不需要数据增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    # ============================
    # 加载MNIST数据集
    # ============================
    print("\n加载MNIST数据集...")
    data_root = './data'
    # 完整的训练集(60000张)和测试集(10000张)
    full_train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
    full_test_dataset = datasets.MNIST(data_root, train=False, download=True, transform=test_transform)

    # 计算子集大小
    train_size = max(int(len(full_train_dataset) * CONFIG['train_ratio']), 1)
    test_size = max(int(len(full_test_dataset) * CONFIG['test_ratio']), 1)

    # 加载或创建固定的索引(确保每次运行使用相同的数据划分)
    indices_cache_root = Path(data_root) / 'splits'
    train_indices = load_or_create_indices(
        'train', len(full_train_dataset), CONFIG['train_ratio'], 
        CONFIG['seed'], indices_cache_root, expected_size=train_size
    )
    test_indices = load_or_create_indices(
        'test', len(full_test_dataset), CONFIG['test_ratio'], 
        CONFIG['seed'], indices_cache_root, expected_size=test_size
    )

    # 创建子集
    train_subset = torch.utils.data.Subset(full_train_dataset, train_indices.tolist())
    test_subset = torch.utils.data.Subset(full_test_dataset, test_indices.tolist())

    print(f"训练集大小: {len(train_subset)} 张图片")
    print(f"测试集大小: {len(test_subset)} 张图片")

    # ============================
    # 创建图片对数据集
    # ============================
    # 训练集:每个epoch生成固定数量的图片对
    train_pairs_dataset = MNISTPairsDataset(
        train_subset, CONFIG['pairs_per_epoch'], seed=CONFIG['seed']
    )
    # 测试集:生成较少的图片对(用于快速评估)
    test_pairs_dataset = MNISTPairsDataset(
        test_subset, max(CONFIG['pairs_per_epoch'] // 3, 1), seed=CONFIG['seed'] + 1
    )

    print(f"训练集样本对数量: {len(train_pairs_dataset)}")
    print(f"测试集样本对数量: {len(test_pairs_dataset)}")

    # ============================
    # 创建数据加载器
    # ============================
    # 是否固定内存(GPU训练时可加速数据传输)
    pin_memory = device.type == 'cuda'

    # 数据加载器的公共参数
    loader_common_kwargs = {
        'batch_size': CONFIG['batch_size'],
        'num_workers': CONFIG['num_workers'],  # 多进程加载
        'pin_memory': pin_memory,
        'worker_init_fn': seed_worker,  # 为每个worker设置种子
        'drop_last': False  # 不丢弃最后一个不完整的batch
    }

    # 多进程模式的额外参数
    if CONFIG['num_workers'] > 0:
        loader_common_kwargs['prefetch_factor'] = CONFIG['prefetch_factor']
        loader_common_kwargs['persistent_workers'] = CONFIG['persistent_workers']

    # 训练数据加载器(需要打乱)
    train_loader = DataLoader(
        train_pairs_dataset,
        shuffle=True,  # 打乱数据
        **loader_common_kwargs
    )
    # 测试数据加载器(不需要打乱)
    test_loader = DataLoader(
        test_pairs_dataset,
        shuffle=False,
        **loader_common_kwargs
    )

    # ============================
    # 构建模型和训练组件
    # ============================
    # 创建Siamese网络并移到设备上
    model = SiameseCompareNet(CONFIG['embedding_dim']).to(device)
    
    # 尝试使用torch.compile加速(PyTorch 2.0+)
    if CONFIG['compile_model']:
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='default')
                print("模型已通过 torch.compile 加速。")
            except Exception as compile_exc:
                print(f"⚠️ torch.compile 失败:{{compile_exc}}. 将以未编译模式继续。")
        else:
            print("当前 PyTorch 版本不支持 torch.compile,跳过编译。")
    
    # 损失函数:二元交叉熵(带logits,数值更稳定)
    criterion = nn.BCEWithLogitsLoss()
    
    # 优化器:AdamW(带权重衰减的Adam)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['weight_decay']
    )
    
    # 学习率调度器:当验证损失不再下降时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控指标越小越好
        factor=CONFIG['scheduler_factor'],  # 衰减因子0.5
        patience=CONFIG['scheduler_patience'],  # 等待2个epoch
        min_lr=CONFIG['min_lr']  # 最小学习率
    )
    
    # 混合精度训练的梯度缩放器
    if HAS_TORCH_GRADSCALER:
        scaler = torch.amp.GradScaler('cuda', enabled=CONFIG['use_amp'] and device.type == 'cuda')
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=CONFIG['use_amp'] and device.type == 'cuda')

    # ============================
    # 打印模型信息
    # ============================
    print("\n" + "=" * 70)
    print("网络架构:")
    print("=" * 70)
    print(model)
    print("=" * 70)

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("=" * 70)

    # ============================
    # 初始化训练历史记录
    # ============================
    history: Dict[str, list] = {
        'train_loss': [],  # 每个epoch的训练损失
        'train_acc': [],   # 每个epoch的训练准确率
        'test_loss': [],   # 每个epoch的测试损失
        'test_acc': [],    # 每个epoch的测试准确率
        'test_auc': [],    # 每个epoch的测试ROC-AUC
        'lr': []           # 每个epoch的学习率
    }

    # 记录最佳模型
    best_test_acc = 0.0  # 最佳测试准确率
    best_epoch = 0  # 最佳epoch
    best_snapshot: Dict[str, Any] = {}  # 最佳模型的指标快照
    best_model_path = os.path.join(exp_dir, 'best_model.pth')

    # ============================
    # 开始训练循环
    # ============================
    print("\n开始训练...")
    print("=" * 70)

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 70)

        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            max_grad_norm=CONFIG['max_grad_norm'],
            epoch=epoch + 1,
            total_epochs=CONFIG['epochs'],
            show_progress=True  # 显示进度条
        )

        # 在测试集上评估
        test_loss, test_acc, test_preds, test_targets, test_auc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        # 更新学习率(基于测试损失)
        scheduler.step(test_loss)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_auc'].append(None if np.isnan(test_auc) else test_auc)
        history['lr'].append(get_current_lr(optimizer))

        # 格式化AUC输出
        auc_text = f"{test_auc:.4f}" if not np.isnan(test_auc) else "n/a"
        
        # 打印当前epoch的结果
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {auc_text}")
        print(f"Current LR: {get_current_lr(optimizer):.6f}")

        # 如果当前模型更好,保存它
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_snapshot = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_auc': test_auc
            }
            # 保存模型参数
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ 保存最佳模型 (Epoch {best_epoch}, Test Acc: {best_test_acc:.4f})")

    # ============================
    # 训练完成
    # ============================
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)

    # 加载最佳模型重新评估
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_test_loss, best_test_acc, best_test_preds, best_test_targets, best_test_auc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    # 打印最终结果
    print("\n最终结果 (Best Checkpoint):")
    print(f"最佳Epoch: {best_epoch}")
    print(f"最佳测试准确率: {best_test_acc:.4f}")
    if not np.isnan(best_test_auc):
        print(f"最佳测试ROC-AUC: {best_test_auc:.4f}")
    print(f"最佳测试Loss: {best_test_loss:.4f}")

    print(f"最终训练准确率(Last Epoch): {history['train_acc'][-1]:.4f}")
    print(f"最终测试准确率(Last Epoch): {history['test_acc'][-1]:.4f}")

    # ============================
    # 特征图分析与剪枝实验
    # ============================
    print("\n分析最后一层卷积特征图...")
    single_image_loader = DataLoader(
        test_subset,
        shuffle=False,
        **loader_common_kwargs
    )
    mean_feature_maps, channel_activation = compute_last_conv_activation_stats(
        model,
        single_image_loader,
        device
    )
    mean_maps_path = os.path.join(exp_dir, 'last_conv_mean_feature_maps.png')
    plot_mean_feature_maps(mean_feature_maps, mean_maps_path)
    print(f"平均特征图已保存: {mean_maps_path}")

    print("执行剪枝实验...")
    channel_order = np.argsort(channel_activation)
    pruning_results = evaluate_pruning_sensitivity(
        model,
        test_loader,
        channel_order,
        criterion,
        device,
        embedding_dim=CONFIG['embedding_dim']
    )
    pruning_curve_path = os.path.join(exp_dir, 'pruning_accuracy_curve.png')
    plot_pruning_curve(pruning_results, pruning_curve_path)
    print(f"剪枝准确率曲线已保存: {pruning_curve_path}")

    # ============================
    # 保存训练历史
    # ============================
    history_path = os.path.join(exp_dir, 'history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(sanitize_history(history), f, indent=2, ensure_ascii=False)

    # ============================
    # 生成可视化结果
    # ============================
    print("\n绘制训练曲线...")
    plot_training_curves(history, os.path.join(exp_dir, 'training_curves.png'))

    print("绘制混淆矩阵...")
    plot_confusion_matrix(best_test_targets, best_test_preds, os.path.join(exp_dir, 'confusion_matrix.png'))

    print("可视化样本...")
    visualize_samples(test_pairs_dataset, model, device, os.path.join(exp_dir, 'sample_predictions.png'))

    # ============================
    # 生成实验报告
    # ============================
    report = {
        'experiment_name': CONFIG['experiment_name'],
        'timestamp': timestamp,
        'config': CONFIG,
        'dataset_info': {
            'train_images': len(train_subset),
            'test_images': len(test_subset),
            'train_pairs': len(train_pairs_dataset),
            'test_pairs': len(test_pairs_dataset)
        },
        'model_info': {
            'architecture': 'SiameseCompareNet',
            'embedding_dim': CONFIG['embedding_dim'],
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'results': {
            'best_epoch': best_epoch,
            'best_test_accuracy': best_test_acc,
            'best_test_auc': sanitize_metric(best_test_auc),
            'best_test_loss': best_test_loss,
            'best_train_accuracy': sanitize_metric(best_snapshot.get('train_acc')),
            'best_train_loss': sanitize_metric(best_snapshot.get('train_loss')),
            'best_test_loss_snapshot': sanitize_metric(best_snapshot.get('test_loss')),
                        'best_test_auc_snapshot': sanitize_metric(best_snapshot.get('test_auc')),
            'final_train_accuracy': history['train_acc'][-1],
            'final_test_accuracy': history['test_acc'][-1],
            'final_learning_rate': history['lr'][-1]
        },
        'pruning_analysis': {
            'mean_feature_maps_path': mean_maps_path,
            'pruning_curve_path': pruning_curve_path,
            'mean_activation_per_channel': channel_activation.tolist(),
            'channel_order_low_to_high': channel_order.tolist(),
            'accuracy_vs_k': pruning_results
        }
    }

    # 保存实验报告为JSON文件
    report_path = os.path.join(exp_dir, 'report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印完成信息
    print(f"\n所有结果已保存到: {exp_dir}/")
    print("=" * 70)


# ============================
# 程序入口
# ============================
if __name__ == "__main__":
    main()

