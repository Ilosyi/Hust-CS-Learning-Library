# Fashion-MNIST 服装分类项目

## 任务描述

本项目使用 PyTorch 框架对 Fashion-MNIST 数据集进行图像分类，目标是识别不同类型的服饰（如 T 恤、外套、鞋子等）。Fashion-MNIST 是 MNIST 的替代数据集，图像为 28×28 灰度图，共包含 10 个服饰类别。

数据集来源：
[https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

## 项目简介

本项目实现了一个基于预训练 **ResNet-18** 的图像分类模型，适配 Fashion-MNIST 的输入，支持 GPU 加速训练，并提供完整的训练、评估与可视化流程。

## 特点与亮点

* 使用 PyTorch 官方预训练的 **ResNet-18** 模型
* 通过 `Grayscale → 3通道 → Resize` 预处理适配 ResNet 输入
* 自动下载并加载 Fashion-MNIST 数据集
* 支持训练过程中的损失和准确率可视化
* 评估包括混淆矩阵与分类报告
* 模型训练完成后支持完整保存和状态字典保存

## 环境要求

* Python ≥ 3.7
* PyTorch ≥ 1.10
* torchvision
* matplotlib
* seaborn
* scikit-learn
* CUDA（可选，用于加速）

## 安装步骤

1. 克隆项目(注意，该仓库在结课前为private)：

   ```bash
   git clone https://github.com/Ilosyi/Fashion-MNIST-Classification.git
   cd Fashion-MNIST-Classification
   ```

2. 安装依赖(可能存在版本错误，或使用了被淘汰的函数等问题，建议手动安装)：

   ```bash
   pip install -r requirements.txt
   ```

手动安装核心依赖：

   ```bash
   pip install torch torchvision matplotlib seaborn scikit-learn
   ```

## 使用方法

1. 运行主程序：

   ```bash
   python main.py
   ```

2. 程序将执行以下操作：

   * 下载 Fashion-MNIST 数据集并进行转换（28x28 → 224x224 + 灰度 → 3通道）
   * 使用 ResNet-18 训练模型
   * 输出每个 epoch 的训练与验证准确率
   * 训练结束后生成可视化图表和评估报告

3. 输出文件包括：

   * `confusion_matrix.png`：测试集混淆矩阵图像
   * `fashion_resnet_full.pth`：完整模型保存（含结构与权重）
   * `model_checkpoint.pth`：仅保存模型与优化器状态字典

## 模型架构说明

* 使用预训练的 **ResNet-18** 模型（`torchvision.models.resnet18(weights='DEFAULT')`）
* 替换输入层以兼容 3 通道输入
* 修改全连接输出层为 `Linear(..., 10)`，对应 10 类输出

## 模型训练细节

* 优化器：Adam（学习率 0.001）
* 损失函数：交叉熵损失（CrossEntropyLoss）
* 批大小：64
* 训练轮数：20
* 未使用早停机制、学习率调度、数据增强（但可以按需添加）

## 模型评估

* 使用 `classification_report()` 输出精度、召回率、F1 值
* 可视化混淆矩阵图（`confusion_matrix.png`）
* 自动保存训练完成的模型及其状态

## 可视化示例

训练结束后将展示如下图表：

* 训练损失曲线（Loss vs Epoch）
* 训练与验证准确率曲线（Accuracy vs Epoch）
* 混淆矩阵图

## 注意事项与建议

* 当前模型未使用数据增强、早停机制、学习率调度等高级策略
* 如需提高模型性能，可考虑：

  * 添加 `transforms.RandomHorizontalFlip()`、`RandomRotation()` 等数据增强
  * 使用 `ReduceLROnPlateau` 或 `CosineAnnealingLR` 学习率调度器
  * 添加验证集划分与早停机制
* 若显存不足可将 batch size 设为 32 或更小

## 目录结构

```text
.
├──.git 					#版本管理文件
├──data/					#下载的数据集
├──.idea/					#项目配置文件夹
├── confusion_matrix.png          # 测试集混淆矩阵图
├── fashion_resnet_full.pth       # 保存的完整模型
├── model_checkpoint.pth          # 模型参数与优化器状态
├── saved_models/                # 模型保存目录
└── main.py                       # 主程序
└── test.py                       # 测试程序
└── study.py                       # 初学时使用，简单的CNN，对应保存的模型为fashion_mnist_model.pth
```

## License

[MIT License](LICENSE)

---
