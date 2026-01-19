import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.metrics import classification_report

class FashionResNet(nn.Module):
    # 必须与训练时完全相同的模型定义
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=False)
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.base.fc.in_features
        self.base.fc = nn.Linear(num_features, 10)

    def forward(self, x):
        return self.base(x)

def load_model(checkpoint_path):
    # 加载完整模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 方式1：加载完整模型（需要类定义在相同命名空间）
    # model = torch.load(checkpoint_path, map_location=device)

    # 方式2：加载状态字典（推荐）
    # 使用 weights_only=False 允许加载任意对象
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FashionResNet().to(device)
    model.load_state_dict(checkpoint['model_state'])

    model.eval()
    return model, checkpoint['transform']

def test_saved_model():
    # 加载模型
    model, transform = load_model('./saved_models/model_checkpoint.pth')

    # 加载测试数据（必须使用相同预处理）
    test_set = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # 执行测试
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(model.base.conv1.weight.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(classification_report(all_labels, all_preds,
                                target_names=test_set.classes))

if __name__ == "__main__":
    test_saved_model()
