"""
OncoAI Fusion - Brain Cancer Classification Training
Multi-class classifier: Glioma, Meningioma, Pituitary
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
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths - Brain cancer data
    DATASET_ROOT = "/Users/lekhans/Desktop/Major-project/dataset_new"
    CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "brain_cancer"
    LOG_DIR = Path(__file__).parent.parent / "logs"
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Model - 3-class classification
    NUM_CLASSES = 3
    IMAGE_SIZE = 224
    PRETRAINED = True
    
    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 0.001
    
    # Device
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Brain cancer classes
    CLASS_NAMES = ['brain_glioma', 'brain_menin', 'brain_tumor']

config = Config()

# ============================================================================
# CUSTOM DATASET - Filter only brain cancer images
# ============================================================================

class BrainCancerDataset(torch.utils.data.Dataset):
    """Dataset that filters only brain cancer images."""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {'brain_glioma': 0, 'brain_menin': 1, 'brain_tumor': 2}
        
        split_dir = os.path.join(root_dir, split)
        
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"  Loaded {len(self.images)} brain cancer images from {split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# DATA TRANSFORMS
# ============================================================================

def get_transforms():
    """Get training and validation transforms."""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

class ResNet50BrainCancer(nn.Module):
    """ResNet50 for brain cancer classification (3 classes)."""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet50BrainCancer, self).__init__()
        
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Replace final layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
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


def train_epoch(model, train_loader, criterion, optimizer, device):
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
            'class_names': config.CLASS_NAMES,
            'model_type': 'brain_cancer_multiclass'
        }
    }
    
    torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
    
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        print(f"  💾 Saved best model with {val_acc:.2f}% accuracy")

# ============================================================================
# MAIN TRAINING
# ============================================================================

def train(args):
    print("=" * 60)
    print("🧠 OncoAI Fusion - Brain Cancer Classification")
    print("   Multi-class: Glioma, Meningioma, Pituitary")
    print("=" * 60)
    print(f"\n📊 Configuration:")
    print(f"   Dataset: {config.DATASET_ROOT}")
    print(f"   Device: {config.DEVICE}")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Learning Rate: {config.LEARNING_RATE}")
    print(f"   Epochs: {config.NUM_EPOCHS}")
    print(f"   Classes: {config.CLASS_NAMES}")
    
    # Create directories
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Load brain cancer datasets
    print(f"\n📂 Loading brain cancer datasets...")
    train_dataset = BrainCancerDataset(config.DATASET_ROOT, 'train', train_transform)
    val_dataset = BrainCancerDataset(config.DATASET_ROOT, 'val', val_transform)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    print(f"\n🔧 Initializing ResNet50 model...")
    model = ResNet50BrainCancer(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED)
    model = model.to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print("-" * 60)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n📈 Epoch [{epoch}/{config.NUM_EPOCHS}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        print(f"   Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        save_checkpoint(model, optimizer, epoch, val_acc, config.CHECKPOINT_DIR, is_best)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break
    
    # Save training history
    with open(config.LOG_DIR / f"brain_cancer_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ BRAIN CANCER TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {config.CHECKPOINT_DIR / 'best_model.pth'}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Brain Cancer classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    train(args)
