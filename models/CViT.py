import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionwiseFeedforward(nn.Module):
    def __init__(self, dim_model, dim_feedforward, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, nheads, dropout):
        super().__init__()
        assert dim_model % nheads == 0, "dim_model must be divisible by nheads"
        self.dim_model = dim_model
        self.n_heads = nheads
        self.head_dim = dim_model // nheads

        self.W_Q = nn.Linear(dim_model, dim_model)
        self.W_K = nn.Linear(dim_model, dim_model)
        self.W_V = nn.Linear(dim_model, dim_model)
        self.W_O = nn.Linear(dim_model, dim_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(attention_weights), V)
        output = self.combine_heads(output)
        return self.W_O(output)

    def split_heads(self, x):
        return x.view(x.size(0), -1, self.n_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.dim_model)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_feedforward, nheads, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(dim_model, nheads, dropout)
        self.ff = PositionwiseFeedforward(dim_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.norm1(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim_model, dim_feedforward, nheads, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, dim_feedforward, nheads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CViT(nn.Module):
    def __init__(self, image_size=224, patch_size=4, num_classes=27, channels=512,
                 dim_model=1024, num_layers=3, nheads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size'

        self.cnn = nn.Sequential(
            CNNBlock(3, 32), CNNBlock(32, 32), CNNBlock(32, 32), nn.MaxPool2d(2),
            CNNBlock(32, 64), CNNBlock(64, 64), CNNBlock(64, 64), nn.MaxPool2d(2),
            CNNBlock(64, 128), CNNBlock(128, 128), CNNBlock(128, 128), nn.MaxPool2d(2),
            CNNBlock(128, 256), CNNBlock(256, 256), CNNBlock(256, 256), CNNBlock(256, 256), nn.MaxPool2d(2),
            CNNBlock(256, 512), CNNBlock(512, 512), CNNBlock(512, 512), CNNBlock(512, 512), nn.MaxPool2d(2)
        )

        self.patch_size = patch_size
        self.dim_model = dim_model

        self.patch_to_embedding = nn.Linear(channels * patch_size * patch_size, dim_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))
        self.pos_embedding = None  # lazy init after computing patch size

        self.transformer = TransformerEncoder(dim_model, dim_feedforward, nheads, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_classes)
        )

    def forward(self, img, mask=None):
        x = self.cnn(img)
        p = self.patch_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape
        if self.pos_embedding is None or self.pos_embedding.shape[1] != (n + 1):
            self.pos_embedding = nn.Parameter(torch.randn(1, n + 1, self.dim_model).to(x.device))

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :x.size(1)]

        x = self.transformer(x, mask)
        x = self.dropout(x[:, 0])
        return self.mlp_head(x)