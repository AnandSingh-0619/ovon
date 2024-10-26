from typing import Any
from torch import nn
from gym import spaces

POSSIBLE_ENCODERS = {"cnn", "mlp", "resnet18"}

class CNNMaskEncoder(nn.Module):
    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # Input is (1, H, W)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))  # Reduce to (512, 1, 1)
        )
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

class MLPMaskEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class ResNet18MaskEncoder(nn.Module):
    def __init__(self, output_dim: int = 768):
        super().__init__()
        resnet18 = nn.resnet18(pretrained=True)
        resnet18.fc = nn.Identity()  # Remove the final classification layer
        self.resnet = resnet18
        self.fc = nn.Linear(512, output_dim)  # ResNet-18 outputs 512-dimensional features

    def forward(self, x):
        x = self.resnet(x)
        return self.fc(x)

def mask_encoder(observation_space: spaces.Dict, mask_encoding_method: str = "cnn") -> Any:

    # Choose mask encoding method based on the config parameter
    if mask_encoding_method == "cnn":
        return CNNMaskEncoder(output_dim=768)
    elif mask_encoding_method == "mlp":
        input_dim = observation_space["mask"].shape[0] * observation_space["mask"].shape[1]
        return MLPMaskEncoder(input_dim=input_dim, output_dim=768)
    elif mask_encoding_method == "resnet18":
        return ResNet18MaskEncoder(output_dim=768)
    else:
        raise ValueError(f"Unsupported mask encoding method: {mask_encoding_method}")

