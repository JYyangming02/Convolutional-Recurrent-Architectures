import torch
import torch.nn as nn
import torchvision.models as models


class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=512, num_layers=1, bidirectional=False, dropout=0.2):
        super(ResNetLSTM, self).__init__()

        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc
        self.feature_dim = 2048  # output channels of ResNet50 conv5 block

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.backbone(x)          # B x 2048 x H x W
        x = x.view(B, self.feature_dim, -1)  # B x 2048 x (H*W)
        x = x.permute(0, 2, 1)             # B x T x 2048

        lstm_out, _ = self.lstm(x)         # B x T x H
        out = lstm_out[:, -1, :]           # take last timestep
        return self.classifier(out)
