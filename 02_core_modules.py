import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

CLASS_ORDER = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
MALIGNANT_INDICES = [1, 3, 4]  # mel(1), bcc(3), akiec(4)
BENIGN_INDICES = [0, 2, 5, 6]  # nv(0), bkl(2), vasc(5), df(6)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


class HAM10000Dataset(Dataset):
    _image_cache = {}

    def __init__(self, df, transform=None, preload_to_ram=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        if preload_to_ram:
            missing_paths = [p for p in self.df['image_path'].unique() if p not in HAM10000Dataset._image_cache]
            if missing_paths:
                print(f"[Cache] 正在将 {len(missing_paths)} 张图像预加载至内存以加速训练...")
                for p in tqdm(missing_paths, desc="Preloading images"):
                    with Image.open(p) as img:
                        HAM10000Dataset._image_cache[p] = img.convert("RGB").resize((256, 256), Image.BILINEAR)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        if path in HAM10000Dataset._image_cache:
            img = HAM10000Dataset._image_cache[path].copy()
        else:
            img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = int(row['label'])
        return img, label


class FocalLoss(nn.Module):
    """
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        modulating_factor = (1.0 - pt) ** self.gamma
        loss = -modulating_factor * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets.squeeze(1))
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_inverse_freq_weights(train_df, num_classes=7, power=0.5):
    """
    计算平滑逆类别频率权重 alpha_t (power=0.5 即平方根逆频率)，
    防止 58:1 的极端长尾导致模型崩溃向 DF/VASC，同时强化恶性类别 (MEL, BCC, AKIEC) 的学习。
    归一化使其总和等于类别数 K=7（严格符合报告 Section 3 描述）。
    """
    counts = train_df['label'].value_counts().sort_index()
    freqs = torch.tensor([counts.get(i, 1) for i in range(num_classes)], dtype=torch.float32)
    inv_freqs = (1.0 / freqs) ** power
    # 针对临床恶性组 (MEL=1, BCC=3, AKIEC=4) 赋予温和的临床优先级系数 (1.25x)，确保恶性召回率稳健提升
    clinical_boost = torch.tensor([1.0, 1.25, 1.0, 1.25, 1.25, 1.0, 1.0], dtype=torch.float32)
    raw_alpha = inv_freqs * clinical_boost
    alpha = raw_alpha / raw_alpha.sum() * num_classes
    return alpha


class SkinCancerResNet50(nn.Module):
    def __init__(self, num_classes=7, dropout_p=0.3, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        self.frozen_modules = []

    def set_trainable_stage(self, mode, epoch=1, head_lr=1e-3, layer4_lr=1e-4, layer3_lr=5e-5, weight_decay=1e-4):
        if mode == 'frozen':
            target_stage = 'stage1_head_only'
        elif mode == 'progressive':
            if epoch <= 3:
                target_stage = 'stage1_head_only'
            elif epoch <= 8:
                target_stage = 'stage2_unfreeze_layer4'
            else:
                target_stage = 'stage3_unfreeze_layer3_4'
        else:
            raise ValueError(f"Unknown backbone mode: {mode}")

        for param in self.backbone.parameters():
            param.requires_grad = False

        param_groups = []
        if target_stage == 'stage1_head_only':
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            param_groups.append({'params': self.backbone.fc.parameters(), 'lr': head_lr, 'weight_decay': weight_decay})
            self.frozen_modules = [
                self.backbone.conv1, self.backbone.bn1,
                self.backbone.layer1, self.backbone.layer2,
                self.backbone.layer3, self.backbone.layer4
            ]

        elif target_stage == 'stage2_unfreeze_layer4':
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            param_groups.append({'params': self.backbone.fc.parameters(), 'lr': head_lr, 'weight_decay': weight_decay})
            param_groups.append({'params': self.backbone.layer4.parameters(), 'lr': layer4_lr, 'weight_decay': weight_decay})
            self.frozen_modules = [
                self.backbone.conv1, self.backbone.bn1,
                self.backbone.layer1, self.backbone.layer2,
                self.backbone.layer3
            ]

        elif target_stage == 'stage3_unfreeze_layer3_4':
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            param_groups.append({'params': self.backbone.fc.parameters(), 'lr': head_lr, 'weight_decay': weight_decay})
            param_groups.append({'params': self.backbone.layer4.parameters(), 'lr': layer4_lr, 'weight_decay': weight_decay})
            param_groups.append({'params': self.backbone.layer3.parameters(), 'lr': layer3_lr, 'weight_decay': weight_decay})
            self.frozen_modules = [
                self.backbone.conv1, self.backbone.bn1,
                self.backbone.layer1, self.backbone.layer2
            ]

        return target_stage, param_groups

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for mod in self.frozen_modules:
                for m in mod.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.eval()
        return self

    def forward(self, x):
        return self.backbone(x)


def run_unit_tests():
    print("Running unit tests for FocalLoss...")
    torch.manual_seed(42)
    logits = torch.randn(32, 7, requires_grad=True)
    targets = torch.randint(0, 7, (32,))

    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    fl_zero = FocalLoss(alpha=None, gamma=0.0)(logits, targets)

    diff = torch.abs(ce_loss - fl_zero).item()
    assert diff < 1e-6, f"Unit Test Failed! Diff={diff}"
    print(f"[Unit Test Passed] FocalLoss(gamma=0, alpha=1) == CrossEntropyLoss (abs diff = {diff:.2e} < 1e-6)")


if __name__ == "__main__":
    run_unit_tests()