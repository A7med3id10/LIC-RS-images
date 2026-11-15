import torch
import torch.nn as nn
from compressai.models.google import MeanScaleHyperprior

class DepthwiseSeparableConv2d(nn.Module):
    """ Depthwise Separable Convolution for Edge Devices """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            groups=in_channels, bias=True
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableDeconv2d(nn.Module):
    """ Depthwise Separable Transposed Convolution for Decoder"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
        super().__init__()
        self.depthwise = nn.ConvTranspose2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            output_padding=output_padding, groups=in_channels, bias=True
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LightweightMBT2018Mean(MeanScaleHyperprior):
    """ mbt2018_mean Depthwise Separable Convolutions"""
    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        
        # Encoder (g_a)    
        from compressai.layers import GDN
        self.g_a = nn.Sequential(
            DepthwiseSeparableConv2d(3, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            DepthwiseSeparableConv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            DepthwiseSeparableConv2d(N, N, kernel_size=5, stride=2, padding=2),
            GDN(N),
            DepthwiseSeparableConv2d(N, M, kernel_size=5, stride=2, padding=2),
        )

        # Decoder (g_s)  
        self.g_s = nn.Sequential(
            DepthwiseSeparableDeconv2d(M, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            DepthwiseSeparableDeconv2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            DepthwiseSeparableDeconv2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            DepthwiseSeparableDeconv2d(N, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

# try Shapes
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightMBT2018Mean(N=100, M=150).to(device)
    
    # A test image with dimensions that are multiples of 64 (e.g., 256x256)
    x = torch.randn(1, 3, 256, 256).to(device)
    out = model(x)
    
    print("Forward pass successful!")
    print("Reconstructed image shape:", out["x_hat"].shape)
    print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    
# py -3.12 para.py
