import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Enable PyTorch memory optimizations
# ✅ Enable PyTorch memory optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))  # Skip connection

class GeneratorUSRGAN(nn.Module):
    def __init__(self, in_channels=3, num_residuals=16):
        super(GeneratorUSRGAN, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks (feature extraction & enhancement)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residuals)])

        # Middle convolution
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Final convolution (restore original channels)
        self.conv3 = nn.Conv2d(64, in_channels, kernel_size=9, padding=4)
        self.tanh = nn.Tanh()  # Output range [-1,1]

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.res_blocks(out1)
        out3 = self.conv2(out2) + out1  # Skip connection
        return self.tanh(self.conv3(out3))  # ✅ Output same resolution as input (240x320)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(Discriminator, self).__init__()

        self.use_sigmoid = use_sigmoid  # ✅ Store use_sigmoid argument

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        if self.use_sigmoid:
            x = self.sigmoid(x)  # Apply Sigmoid if enabled

        return x

