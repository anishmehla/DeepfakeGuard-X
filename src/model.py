# src/model.py
import torch
import torch.nn as nn
import timm


class SpatialAttentionBlock(nn.Module):
    """
    Computes a spatial attention mask across channel-wise mean and max projections
    to emphasize facial blending boundaries.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn


class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        use_spatial_attention: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        in_features = self.backbone.num_features
        self.spatial_attn = SpatialAttentionBlock() if use_spatial_attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        feat = self.spatial_attn(feat)
        pooled = self.pool(feat)
        logits = self.head(pooled).squeeze(-1)
        return logits