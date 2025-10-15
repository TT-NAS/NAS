# Implementación de la arquitectura FCN-VGG para segmentación semántica
import torch.nn as nn
from utils import set_current_net_binary
import torch
def bilinear_kernel(in_ch, out_ch, k):
    import torch
    factor = (k + 1) // 2
    if k % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = torch.arange(k, dtype=torch.float32)
    filt = (1 - torch.abs(og - center) / factor)
    w2d = filt[:, None] * filt[None, :]
    w = torch.zeros((in_ch, out_ch, k, k), dtype=torch.float32)
    for i in range(min(in_ch, out_ch)):
        w[i, i] = w2d
    return w

def init_deconv_bilinear(deconv: nn.ConvTranspose2d):
    k = deconv.kernel_size[0]
    w = bilinear_kernel(deconv.in_channels, deconv.out_channels, k)
    w = w.to(device=deconv.weight.device, dtype=deconv.weight.dtype)
    with torch.no_grad():              
        deconv.weight.copy_(w)
        
class VGG_FCN(nn.Module):
    def __init__(self, version='32'):
        super(VGG_FCN, self).__init__()
        self.version = version
        self.conv_layers = nn.ModuleList()
        set_current_net_binary(f"fcn_vgg_{version}")
        
        # Capas convolucionales (VGG16)
        in_channels = 3
        depths = [64, 128, 256, 512, 512]
        n_convs = [2, 2, 3, 3, 3]
        
        for i in range(len(depths)):
            layers = []
            for j in range(n_convs[i]): # Convoluciones
                layers.append(nn.Conv2d(in_channels if j == 0 else depths[i], depths[i], kernel_size=3, padding=1, bias=True))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # Max Pooling
            self.conv_layers.append(nn.Sequential(*layers))
            in_channels = depths[i]
        
        # Capas para FCN
        self.score_conv = nn.Conv2d(512, 1, kernel_size=1, bias=True) # Capa de scoring (Pool 5)
        
        if version == '32':
            self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=64, stride=32, padding=16, bias=False) # Deconvolución x32
        elif version == '16':
            self.score_pool4 = nn.Conv2d(512, 1, kernel_size=1, bias=True) # Capa de scoring (Pool 4)
            self.upsample2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False) # Deconvolución x2
            self.upsample16 = nn.ConvTranspose2d(1, 1, kernel_size=32, stride=16, padding=8, bias=False) # Deconvolución x16

        elif version == '8':
            self.score_pool4 = nn.Conv2d(512, 1, kernel_size=1, bias=True) # Capa de scoring (Pool 4)
            self.score_pool3 = nn.Conv2d(256, 1, kernel_size=1, bias=True) # Capa de scoring (Pool 3)
            self.upsample2_1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False) # Deconvolución x2 (Pool 4)
            self.upsample2_2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False) # Deconvolución x2 (Pool 3)
            self.upsample8 = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=8, padding=4, bias=False) # Deconvolución x8

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init_deconv_bilinear(m) # Inicialización bilineal de las deconvoluciones
                
    def forward(self, x):       
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if i == 3:  # pool4
                pool4 = x
            elif i == 2:  # pool3
                pool3 = x
        
        # Score del pool5
        x = self.score_conv(x)
        
        if self.version == '32':
            x = self.upsample(x) # Deconvolución x32
            
        elif self.version == '16':
            x = self.upsample2(x) # Deconvolución x2 sobre el score del pool5
            pool4_scored = self.score_pool4(pool4) # Score del pool 4    
            x = x + pool4_scored # Combinación de los scores
            x = self.upsample16(x) # Deconvolución x16
            
        elif self.version == '8':
            x = self.upsample2_1(x) # Deconvolución x2 sobre el score del pool5
            pool4_scored = self.score_pool4(pool4) # Score del pool 4
            x = x + pool4_scored # Sumar scores
            x = self.upsample2_2(x) # Deconvolución x2 de la suma
            pool3_scored = self.score_pool3(pool3) # Score del pool 3
            x = x + pool3_scored # Sumar scores
            x = self.upsample8(x) # Deconvolución x8

        return x