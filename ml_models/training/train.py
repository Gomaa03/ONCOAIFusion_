"""
OncoAI Fusion - Multi-Cancer Classification Training Pipeline
ResNet50-based classifier for 22 cancer subtypes
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "backend"))

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATASET_ROOT = "/Users/lekhans/Desktop/new folder/dataset_new"
    CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
    LOG_DIR = Path(__file__).parent.parent / "logs"
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Model
    NUM_CLASSES = 22
    IMAGE_SIZE = 224
    PRETRAINED = True
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Device
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cancer class names
    CLASS_NAMES = [
        'brain_glioma', 'brain_menin', 'brain_tumor',
        'breast_benign', 'breast_malignant',
        'cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi',
        'colon_aca', 'colon_bnt',
        'kidney_normal', 'kidney_tumor',
        'lung_aca', 'lung_bnt', 'lung_scc',
        'lymph_cll', 'lymph_fl', 'lymph_mcl',
        'oral_normal', 'oral_scc'
    ]

config = Config()

# ============================================================================
# DATA TRANSFORMS
# ============================================================================

def get_transforms():
    """Get training and validation transforms with augmentation."""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class ResNet50MultiCancer(nn.Module):
    """ResNet50 with custom classification head for multi-cancer classification."""
    
    def __init__(self, num_classes=22, pretrained=True):
        super(ResNet50MultiCancer, self).__init__()
        
        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Replace final layer with custom head
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_embedding(self, x):
        """Extract features before classification layer."""
        modules = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        x = feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class MetricsLogger:
    """Log training metrics to file."""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(lr)
        self.history['epoch_time'].append(epoch_time)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_best_epoch(self):
        if not self.history['val_acc']:
            return 0
        return max(range(len(self.history['val_acc'])), key=lambda i: self.history['val_acc'][i])

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': {
            'num_classes': config.NUM_CLASSES,
            'image_size': config.IMAGE_SIZE,
            'class_names': config.CLASS_NAMES
        }
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        print(f"  💾 Saved best model with {val_acc:.2f}% accuracy")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(args):
    """Main training function."""
    
    print("=" * 60)
    print("🧬 OncoAI Fusion - Multi-Cancer Classification Training")
    print("=" * 60)
    print(f"\n📊 Configuration:")
    print(f"   Dataset: {config.DATASET_ROOT}")
    print(f"   Device: {config.DEVICE}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Learning Rate: {config.LEARNING_RATE}")
    print(f"   Epochs: {config.NUM_EPOCHS}")
    print(f"   Classes: {config.NUM_CLASSES}")
    
    # Create directories
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    print(f"\n📂 Loading datasets...")
    train_dataset = datasets.ImageFolder(
        os.path.join(config.DATASET_ROOT, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(config.DATASET_ROOT, 'val'),
        transform=val_transform
    )
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print(f"\n🔧 Initializing ResNet50 model...")
    model = ResNet50MultiCancer(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED
    )
    model = model.to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)
    
    # Metrics logger
    logger = MetricsLogger(config.LOG_DIR)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print("-" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        print(f"\n📈 Epoch [{epoch}/{config.NUM_EPOCHS}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)
        
        # Print epoch summary
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        print(f"   LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        save_checkpoint(model, optimizer, epoch, val_acc, config.CHECKPOINT_DIR, is_best)
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            break
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Best Epoch: {logger.get_best_epoch() + 1}")
    print(f"   Model saved to: {config.CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"   Logs saved to: {logger.log_file}")
    
    return model, logger.history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train OncoAI Fusion model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Update config with args
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    # Run training
    train(args)
