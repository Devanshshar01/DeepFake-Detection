import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from pathlib import Path
import numpy as np
import json

def mixup_data(x_frames, x_audio, y, alpha=0.2):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_frames.size()[0]
    index = torch.randperm(batch_size).to(x_frames.device)

    mixed_frames = lam * x_frames + (1 - lam) * x_frames[index, :]
    mixed_audio = lam * x_audio + (1 - lam) * x_audio[index, :]
    y_a, y_b = y, y[index]
    return mixed_frames, mixed_audio, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Trainer:
    """Enhanced training pipeline for deepfake detector with advanced strategies"""

    def __init__(self, model, train_loader, val_loader, config):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        label_smoothing = config['training'].get('label_smoothing', 0.0)
        self.use_focal_loss = config['training'].get('use_focal_loss', False)

        if self.use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )

        self.use_cosine_schedule = config['training'].get('use_cosine_schedule', True)
        self.warmup_epochs = config['training'].get('warmup_epochs', 5)
        self.min_lr = config['training'].get('min_learning_rate', 1e-6)

        if self.use_cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.min_lr
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5, min_lr=self.min_lr
            )

        self.mixup_alpha = config['training'].get('mixup_alpha', 0.0)
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.patience = config['training']['early_stopping_patience']

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

    def train_epoch(self, epoch) -> tuple:
        """Train for one epoch with mixup and warmup"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config['training']['learning_rate'] * warmup_factor

        pbar = tqdm(self.train_loader, desc=f"Training (Epoch {epoch+1})")
        for frames, audio, labels in pbar:
            frames = frames.to(self.device)
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            use_mixup = self.mixup_alpha > 0 and np.random.random() > 0.5

            if use_mixup:
                frames, audio, labels_a, labels_b, lam = mixup_data(
                    frames, audio, labels, self.mixup_alpha
                )

            self.optimizer.zero_grad()
            outputs = self.model(frames, audio)

            if use_mixup:
                loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            if use_mixup:
                correct += (lam * predicted.eq(labels_a).sum().item() +
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                correct += predicted.eq(labels).sum().item()

            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        return total_loss / len(self.train_loader), correct / total

    def validate(self) -> tuple:
        """Validate the model with detailed metrics"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for frames, audio, labels in tqdm(self.val_loader, desc="Validation"):
                frames = frames.to(self.device)
                audio = audio.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(frames, audio)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy

    def train(self, num_epochs: int):
        """Full training loop with enhanced strategies"""
        print(f"Training on device: {self.device}")
        print(f"Using mixup: {self.mixup_alpha > 0}")
        print(f"Using cosine schedule: {self.use_cosine_schedule}")
        print(f"Label smoothing: {self.config['training'].get('label_smoothing', 0.0)}")

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            if self.use_cosine_schedule and epoch >= self.warmup_epochs:
                self.scheduler.step()
            elif not self.use_cosine_schedule:
                self.scheduler.step(val_loss)

            is_best = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                self.patience_counter = 0
                print(f"  ✓ New best model! Accuracy: {val_acc*100:.2f}%")
                is_best = True
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.patience}")

            if not is_best and epoch % 5 == 0:
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation accuracy: {self.best_val_acc*100:.2f}%")
                break

        self.save_training_history()
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print(f"Best model saved to: models/checkpoints/best_model.pth")
        print(f"{'='*60}")

    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Save model checkpoint"""
        Path("models/checkpoints").mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config
        }

        if is_best:
            torch.save(checkpoint, 'models/checkpoints/best_model.pth')
        else:
            torch.save(checkpoint, f'models/checkpoints/checkpoint_epoch_{epoch+1}.pth')

    def save_training_history(self):
        """Save training history to JSON"""
        Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
        with open('models/checkpoints/training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
