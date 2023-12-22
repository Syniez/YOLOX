import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def create_random_mask(prob, size):
    """
    1-prob: Ratio of masking
    size: the size of masking tensor
    """
    mask = torch.rand(size)

    # masking region(70%) is True(=1)
    mask = mask > prob

    return mask


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # padding to align size
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # skip connection
        x = torch.cat([x2, x1], dim=1) 
        return self.conv(x)


class YOLOPATCH(nn.Module):
    def __init__(self, patch_masking, patch_size=32, use_updown=True, hidden=64):
        self.patch_masking = patch_masking
        super().__init__()
        self.patch_size = patch_size
        self.use_updown = use_updown
        
        if use_updown:
            self.inc = DoubleConv(3, 16)
            self.down1 = Down(16, 32)
            self.down2 = Down(32, hidden // 2)
            self.up1 = Up(hidden, 32 // 2)
            self.up2 = Up(32, 16)
            self.outc = nn.Conv2d(16, 3, kernel_size=1)
        else:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, 1, 1),
                nn.ReLU(),
            )


    def forward(self, images):
        
        if self.use_updown:
            # convert the patches to the batch images
            patches = rearrange(images, 'b c (n1 h) (n2 w) -> (b n1 n2) c h w', h=self.patch_size, w=self.patch_size)
            
            # the number of patches
            n1 = images.shape[2] // self.patch_size 
            n2 = images.shape[3] // self.patch_size

            # ramdom masking
            if self.patch_masking:
                batch_size = patches.shape[0]
                mask = create_random_mask(0.3, batch_size)
                patches[mask, :, :, :] = 0

            # light UNet structure
            x1 = self.inc(patches)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up1(x3, x2)
            x = self.up2(x, x1)
            x = self.outc(x) + patches

            # restore original shape
            output = rearrange(x, '(b n1 n2) c h w -> b c (n1 h) (n2 w)', n1=n1, n2=n2)
        else:
            patches = rearrange(images, 'b c (h p1) (w p2) -> (b p1 p2) c h w', p1=32, p2=32)
            if self.patch_masking:
                batch_size = patches.shape[0]
                mask = create_random_mask(0.3, batch_size)
                patches[mask, :, :, :] = 0
            patches = self.model(patches)
            output = rearrange(patches, '(b p1 p2) c h w -> b c (h p1) (w p2)', p1=32, p2=32)
            
        return output


if __name__ == "__main__":
    model = YOLOPATCH(patch_masking=True, patch_size=32, hidden=64).cuda()
    images = torch.randn(1, 3, 16, 16).cuda()
    output = model(images)
    
    import jhutil;jhutil.jhprint(1111, output)

