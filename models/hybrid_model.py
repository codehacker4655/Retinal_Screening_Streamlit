"""Exact H3 network definition extracted from notebook cell 161."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class HybridEfficientNetViT(nn.Module):

    def __init__(
        self,
        num_classes=5,
        embed_dim=256,
        num_heads=4,
        transformer_layers=2,
        dropout=0.1,
        pretrained=True
    ):
        super().__init__()

        # 1. CNN backbone
        weights = (
            EfficientNet_B2_Weights.DEFAULT
            if pretrained else None
        )

        backbone = efficientnet_b2(weights=weights)

        self.cnn = backbone.features

        cnn_channels = backbone.classifier[1].in_features

        # 2. Local CNN representation
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # 3. Convert CNN feature maps into tokens
        self.token_projection = nn.Conv2d(
            cnn_channels,
            embed_dim,
            kernel_size=1
        )

        # Fixed token grid for consistent training
        self.token_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Learned positional embeddings for 49 tokens
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 49, embed_dim)
        )
        nn.init.normal_(self.pos_embedding, std=0.02)

        # 4. Vision Transformer self-attention encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False
        )

        self.transformer_norm = nn.LayerNorm(embed_dim)

        # 5. Fuse local and global feature vectors
        fusion_dim = cnn_channels + embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        # CNN spatial feature extraction
        features = self.cnn(x)

        # Local representation
        cnn_vector = self.cnn_pool(features)
        cnn_vector = torch.flatten(cnn_vector, 1)

        # Spatial tokens for Transformer
        tokens = self.token_projection(features)
        tokens = self.token_pool(tokens)

        # B x C x 7 x 7 -> B x 49 x C
        tokens = tokens.flatten(2).transpose(1, 2)

        # Add spatial position information
        tokens = tokens + self.pos_embedding

        # Global self-attention
        tokens = self.transformer(tokens)
        tokens = self.transformer_norm(tokens)

        transformer_vector = tokens.mean(dim=1)

        # Local-global fusion
        fused = torch.cat(
            [cnn_vector, transformer_vector],
            dim=1
        )

        # Return logits for CrossEntropyLoss
        logits = self.classifier(fused)

        return logits
