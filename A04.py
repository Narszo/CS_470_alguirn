import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2

# APPROACH REGISTRY
def get_approach_names():
    """Return a list of all approach names."""
    return ["SimpleCNN", "ResCNN"]


def get_approach_description(approach_name):
    """Return a one-sentence description of the given approach."""
    descriptions = {
        "SimpleCNN": (
            "A basic CNN with three conv-batchnorm-ReLU-maxpool blocks "
            "(32/64/128 filters) followed by two fully-connected layers; "
            "no data augmentation."
        ),
        "ResCNN": (
            "A deeper residual CNN with skip connections, ELU activations, "
            "larger filter counts (64/128/256), adaptive average pooling, "
            "and random-flip/random-crop data augmentation during training."
        ),
    }
    return descriptions[approach_name]


# DATA TRANSFORMS
def get_data_transform(approach_name, training):
    """Return the appropriate torchvision v2 transform pipeline."""
    base = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]

    if approach_name == "ResCNN" and training:
        # Data augmentation only for ResCNN training
        return v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32, padding=4),
        ] + base)

    # No augmentation for SimpleCNN, or for any test/eval data
    return v2.Compose(base)

# BATCH SIZE
def get_batch_size(approach_name):
    """Return the preferred batch size for the given approach."""
    if approach_name == "ResCNN":
        return 64
    return 128


# MODEL DEFINITIONS

# SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 32x32 -> 16x16

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 16x16 -> 8x8

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, class_cnt),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ResCNN
class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ELU()

        # Shortcut projection when spatial size or channel count changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + identity)
        return out

class ResCNN(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.stage1 = _ResBlock(64, 64, stride=1)     # 32x32
        self.stage2 = _ResBlock(64, 128, stride=2)     # 32x32 -> 16x16
        self.stage3 = _ResBlock(128, 256, stride=2)    # 16x16 -> 8x8

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, class_cnt)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

###############################################################################
# MODEL FACTORY
###############################################################################

def create_model(approach_name, class_cnt):
    """Build and return the appropriate model (not yet on GPU)."""
    if approach_name == "SimpleCNN":
        return SimpleCNN(class_cnt)
    if approach_name == "ResCNN":
        return ResCNN(class_cnt)
    raise ValueError(f"Unknown approach: {approach_name}")

###############################################################################
# TRAINING HELPERS
###############################################################################

def _train_one_epoch(model, device, dataloader, optimizer, loss_fn):
    """Run one epoch of training; return average loss."""
    model.train()
    total_loss = 0.0
    batches = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / batches

def _evaluate(model, device, dataloader):
    """Return accuracy on the given dataloader (for monitoring only)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

###############################################################################
# TRAIN MODEL
###############################################################################

def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    """
    Train the model and return it.  Test dataloader is used ONLY for
    printing progress (never for tuning hyper-parameters).
    """
    loss_fn = nn.CrossEntropyLoss()

    if approach_name == "SimpleCNN":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 20

        for epoch in range(epochs):
            avg_loss = _train_one_epoch(model, device, train_dataloader,
                                        optimizer, loss_fn)
            test_acc = _evaluate(model, device, test_dataloader)
            print(f"  Epoch {epoch + 1:>2}/{epochs}  "
                  f"loss={avg_loss:.4f}  test_acc={test_acc:.4f}")

    elif approach_name == "ResCNN":
        optimizer = optim.SGD(model.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        epochs = 30

        for epoch in range(epochs):
            avg_loss = _train_one_epoch(model, device, train_dataloader,
                                        optimizer, loss_fn)
            scheduler.step()
            test_acc = _evaluate(model, device, test_dataloader)
            print(f"  Epoch {epoch + 1:>2}/{epochs}  "
                  f"loss={avg_loss:.4f}  test_acc={test_acc:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")
    else:
        raise ValueError(f"Unknown approach: {approach_name}")

    return model
