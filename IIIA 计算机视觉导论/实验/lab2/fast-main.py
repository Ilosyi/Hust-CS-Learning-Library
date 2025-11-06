import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from pathlib import Path

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

CONFIG = {
   'batch_size': 256,
   'epochs': 15,
   'learning_rate': 8e-4,
   'weight_decay': 1e-4,
   'train_ratio': 0.1,
   'test_ratio': 0.1,
   'pairs_per_epoch': 5000,
   'seed': 42,
   'device': 'cuda' if torch.cuda.is_available() else 'cpu',
   'embedding_dim': 128,
}


def set_seed(seed):
   np.random.seed(seed)
   torch.manual_seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)


set_seed(CONFIG['seed'])


def get_subset_labels(subset):
   if isinstance(subset, torch.utils.data.Subset):
       base_dataset = subset.dataset
       indices = np.array(list(subset.indices))
       targets = base_dataset.targets if hasattr(base_dataset, 'targets') else base_dataset.labels
       if isinstance(targets, torch.Tensor):
           targets = targets.cpu().numpy()
       return targets[indices]
   return np.array([int(subset[i][1]) for i in range(len(subset))])


class MNISTPairsDataset(Dataset):
   def __init__(self, mnist_dataset, num_pairs=5000, seed=42):
       self.mnist_dataset = mnist_dataset
       self.num_pairs = num_pairs
       self.rng = np.random.default_rng(seed)
       
       labels = get_subset_labels(mnist_dataset)
       self.label_to_indices = {}
       for idx, label in enumerate(labels.tolist()):
           self.label_to_indices.setdefault(label, []).append(idx)
       
       self.label_to_indices = {
           label: np.array(indices, dtype=np.int64)
           for label, indices in self.label_to_indices.items()
           if len(indices) > 0
       }
       
       self.same_labels = [l for l, idx in self.label_to_indices.items() if len(idx) > 1]
       self.diff_labels = list(self.label_to_indices.keys())
   
   def __len__(self):
       return self.num_pairs
   
   def __getitem__(self, index):
       if self.same_labels and self.rng.random() < 0.5:
           label = int(self.rng.choice(self.same_labels))
           idx1, idx2 = self.rng.choice(self.label_to_indices[label], size=2, replace=False)
           target = 1.0
       else:
           label1, label2 = self.rng.choice(self.diff_labels, size=2, replace=False)
           idx1 = self.rng.choice(self.label_to_indices[int(label1)])
           idx2 = self.rng.choice(self.label_to_indices[int(label2)])
           target = 0.0
       
       img1, _ = self.mnist_dataset[int(idx1)]
       img2, _ = self.mnist_dataset[int(idx2)]
       return img1, img2, torch.tensor(target, dtype=torch.float32)


class SiameseNet(nn.Module):
   def __init__(self, embedding_dim=128):
       super().__init__()
       self.feature_extractor = nn.Sequential(
           nn.Conv2d(1, 32, kernel_size=5, padding=2),
           nn.BatchNorm2d(32),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(2),
           nn.Conv2d(32, 64, kernel_size=3, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(2),
           nn.Conv2d(64, 128, kernel_size=3, padding=1),
           nn.BatchNorm2d(128),
           nn.ReLU(inplace=True),
           nn.AdaptiveAvgPool2d(1)
       )
       
       self.projection = nn.Sequential(
           nn.Flatten(),
           nn.Linear(128, embedding_dim),
           nn.ReLU(inplace=True)
       )
       
       self.classifier = nn.Sequential(
           nn.Linear(embedding_dim * 2, 256),
           nn.ReLU(inplace=True),
           nn.Dropout(0.4),
           nn.Linear(256, 64),
           nn.ReLU(inplace=True),
           nn.Dropout(0.2),
           nn.Linear(64, 1)
       )
   
   def _encode(self, x):
       features = self.feature_extractor(x)
       return self.projection(features)
   
   def forward(self, img1, img2):
       emb1 = self._encode(img1)
       emb2 = self._encode(img2)
       features = torch.cat([torch.abs(emb1 - emb2), emb1 * emb2], dim=1)
       return self.classifier(features).squeeze(dim=1)


def train_epoch(model, dataloader, criterion, optimizer, device):
   model.train()
   running_loss = 0.0
   all_preds = []
   all_targets = []
   
   for imgs1, imgs2, targets in dataloader:
       imgs1, imgs2, targets = imgs1.to(device), imgs2.to(device), targets.to(device)
       
       optimizer.zero_grad(set_to_none=True)
       logits = model(imgs1, imgs2)
       loss = criterion(logits, targets)
       loss.backward()
       optimizer.step()
       
       running_loss += loss.item() * imgs1.size(0)
       probs = torch.sigmoid(logits)
       preds = (probs > 0.5).float()
       all_preds.extend(preds.cpu().numpy().astype(int).tolist())
       all_targets.extend(targets.cpu().numpy().astype(int).tolist())
   
   epoch_loss = running_loss / len(dataloader.dataset)
   epoch_acc = accuracy_score(all_targets, all_preds)
   return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
   model.eval()
   running_loss = 0.0
   all_preds = []
   all_targets = []
   all_probs = []
   
   with torch.no_grad():
       for imgs1, imgs2, targets in dataloader:
           imgs1, imgs2, targets = imgs1.to(device), imgs2.to(device), targets.to(device)
           
           logits = model(imgs1, imgs2)
           loss = criterion(logits, targets)
           running_loss += loss.item() * imgs1.size(0)
           
           probs = torch.sigmoid(logits)
           preds = (probs > 0.5).float()
           all_probs.extend(probs.cpu().numpy().tolist())
           all_preds.extend(preds.cpu().numpy().astype(int).tolist())
           all_targets.extend(targets.cpu().numpy().astype(int).tolist())
   
   epoch_loss = running_loss / len(dataloader.dataset)
   epoch_acc = accuracy_score(all_targets, all_preds)
   test_auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else float('nan')
   
   return epoch_loss, epoch_acc, test_auc


def main():
   device = torch.device(CONFIG['device'])
   print(f"Device: {device}")
   
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
   ])
   
   print("Loading MNIST...")
   data_root = './data'
   full_train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
   full_test = datasets.MNIST(data_root, train=False, download=True, transform=transform)
   
   train_size = max(int(len(full_train) * CONFIG['train_ratio']), 1)
   test_size = max(int(len(full_test) * CONFIG['test_ratio']), 1)
   
   np.random.seed(CONFIG['seed'])
   train_indices = np.random.choice(len(full_train), train_size, replace=False)
   test_indices = np.random.choice(len(full_test), test_size, replace=False)
   
   train_subset = torch.utils.data.Subset(full_train, train_indices)
   test_subset = torch.utils.data.Subset(full_test, test_indices)
   
   train_pairs = MNISTPairsDataset(train_subset, CONFIG['pairs_per_epoch'], CONFIG['seed'])
   test_pairs = MNISTPairsDataset(test_subset, CONFIG['pairs_per_epoch'] // 3, CONFIG['seed'] + 1)
   
   train_loader = DataLoader(train_pairs, batch_size=CONFIG['batch_size'], shuffle=True)
   test_loader = DataLoader(test_pairs, batch_size=CONFIG['batch_size'], shuffle=False)
   
   model = SiameseNet(CONFIG['embedding_dim']).to(device)
   criterion = nn.BCEWithLogitsLoss()
   optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
   
   print(f"\nTraining for {CONFIG['epochs']} epochs...")
   for epoch in range(CONFIG['epochs']):
       train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
       test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
       
       print(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
             f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
             f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f}")


if __name__ == "__main__":
   main()
