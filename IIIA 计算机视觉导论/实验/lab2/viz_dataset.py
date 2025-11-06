import os, json
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Constants
SEED = 42
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
TRAIN_RATIO, TEST_RATIO = 0.1, 0.1
DATA_ROOT = './data'
SAMPLES = 16  # number of pairs to visualize per split
ROWS = 4

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Utilities
def _format_ratio_tag(ratio: float) -> str:
    return f"{int(round(ratio * 10000)):05d}"

def load_or_create_indices(kind: str, dataset_length: int, ratio: float, seed: int, cache_root: Path, expected_size: int):
    cache_root.mkdir(parents=True, exist_ok=True)
    ratio_tag = _format_ratio_tag(ratio)
    cache_file = cache_root / f"{kind}_len{dataset_length}_ratio{ratio_tag}_seed{seed}.npy"
    if cache_file.exists():
        idx = np.load(cache_file)
        if idx.shape[0] == expected_size:
            return idx
        cache_file.unlink(missing_ok=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(dataset_length, expected_size, replace=False).astype(np.int64)
    np.save(cache_file, idx)
    return idx

def _get_subset_labels(subset) -> np.ndarray:
    if isinstance(subset, Subset):
        base = subset.dataset
        idx = np.array(list(subset.indices))
        if hasattr(base, 'targets'):
            targets = base.targets
        elif hasattr(base, 'labels'):
            targets = base.labels
        else:
            targets = [int(base[i][1]) for i in idx]
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        else:
            targets = np.array(targets)
        return targets[idx]
    return np.array([int(subset[i][1]) for i in range(len(subset))])

# Pair dataset (same logic as main, simplified)
class MNISTPairsDataset(Dataset):
    def __init__(self, mnist_subset, num_pairs: int, seed: int = 0):
        self.ds = mnist_subset
        self.n = num_pairs
        self.rng = np.random.default_rng(seed)
        labels = _get_subset_labels(mnist_subset)
        self.lbl2idx = {}
        for i, l in enumerate(labels.tolist()):
            self.lbl2idx.setdefault(int(l), []).append(i)
        self.lbls = list(self.lbl2idx.keys())
        self.same_lbls = [k for k, v in self.lbl2idx.items() if len(v) > 1]
        if len(self.lbls) < 2:
            raise ValueError('Need at least two classes to form pairs')
    def __len__(self):
        return self.n
    def __getitem__(self, _):
        if self.same_lbls and self.rng.random() < 0.5:
            l = int(self.rng.choice(self.same_lbls))
            i, j = self.rng.choice(self.lbl2idx[l], 2, replace=False)
            t = 1.0
        else:
            l1, l2 = self.rng.choice(self.lbls, 2, replace=False)
            i = self.rng.choice(self.lbl2idx[int(l1)])
            j = self.rng.choice(self.lbl2idx[int(l2)])
            t = 0.0
        x1, _ = self.ds[int(i)]
        x2, _ = self.ds[int(j)]
        return x1, x2, torch.tensor(t, dtype=torch.float32)

# Denormalize to show images
def denorm(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    arr = arr * MNIST_STD + MNIST_MEAN
    return np.clip(arr, 0.0, 1.0)

# Plot a grid of pairs
def plot_pairs(ds: Dataset, save_path: str, num_samples: int = 16, rows: int = 4):
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
    axes = np.atleast_1d(axes).flatten()
    k = min(num_samples, len(axes), len(ds))
    for i in range(k):
        x1, x2, t = ds[i]
        c1 = denorm(x1.squeeze(0))
        c2 = denorm(x2.squeeze(0))
        combo = np.hstack([c1, c2])
        axes[i].imshow(combo, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i].axis('off')
        axes[i].set_title('Same' if int(t.item()) == 1 else 'Different',
                          color='green' if int(t.item()) == 1 else 'royalblue')
    for j in range(k, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    # transforms (aligned with main.py)
    train_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(degrees=12, translate=(0.12, 0.12), scale=(0.9, 1.1), shear=8)
        ], p=0.7),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    # load base datasets
    train_full = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=train_transform)
    test_full = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=test_transform)

    # fixed 10% splits with cache
    train_size = max(int(len(train_full) * TRAIN_RATIO), 1)
    test_size = max(int(len(test_full) * TEST_RATIO), 1)
    split_root = Path(DATA_ROOT) / 'splits'
    train_idx = load_or_create_indices('train', len(train_full), TRAIN_RATIO, SEED, split_root, expected_size=train_size)
    test_idx = load_or_create_indices('test', len(test_full), TEST_RATIO, SEED, split_root, expected_size=test_size)

    train_subset = Subset(train_full, train_idx.tolist())
    test_subset = Subset(test_full, test_idx.tolist())

    # build pair datasets (smaller number just for visualization)
    vis_pairs_train = MNISTPairsDataset(train_subset, num_pairs=SAMPLES, seed=SEED)
    vis_pairs_test = MNISTPairsDataset(test_subset, num_pairs=SAMPLES, seed=SEED + 1)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('experiments') / f'dataset_viz_{ts}'
    train_img = out_dir / 'train_pairs.png'
    test_img = out_dir / 'test_pairs.png'

    plot_pairs(vis_pairs_train, str(train_img), num_samples=SAMPLES, rows=ROWS)
    plot_pairs(vis_pairs_test, str(test_img), num_samples=SAMPLES, rows=ROWS)

    summary = {
        'seed': SEED,
        'train_ratio': TRAIN_RATIO,
        'test_ratio': TEST_RATIO,
        'samples_per_split': SAMPLES,
        'output': {
            'train_pairs': str(train_img),
            'test_pairs': str(test_img)
        }
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('Saved dataset visualization to:')
    print(' -', train_img)
    print(' -', test_img)

if __name__ == '__main__':
    main()
