import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import timm

#Encoder
class Encoder(nn.Module):
    def __init__(self, backbone, pretrained=True, img_size = 640):
        super(Encoder, self).__init__()

        self.img_size = img_size
        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                          features_only=True, img_size=768)
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.encoder.children()):
            # x = layer(x)
            # if i in [2, 4, 5, 6, 7]:
            #     
            x = layer(x)
            features.append(x)

                
        return features

class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Decoder, self).__init__()
        
        self.num_classes = num_classes
        self.decoder_block = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels//(2**i), in_channels//(2**(i+1)), kernel_size=2, stride=2, dilation=1),
                nn.BatchNorm2d(in_channels//(2**(i+1))),
                nn.ReLU(inplace=True),
                #nn.Dropout2d(0.2),
                nn.Conv2d(in_channels//(2**(i+1)), in_channels//(2**(i+1)), kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(in_channels//(2**(i+1))),
                nn.ReLU(inplace=True),
            ) for i in range(2) 
        ])

        # #base
        # self.segmentationHad = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels//4, in_channels//4, kernel_size=2, stride=2, dilation=1),
        #     nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, dilation=1),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(192, num_classes, kernel_size=1)
        # )
        # large and swin
        
        self.segmentationHad = nn.Sequential(
            nn.ConvTranspose2d(in_channels//4, in_channels//4, kernel_size=2, stride=2, dilation=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        

    def forward(self, features):


        feat = features[-1] 
        x = feat.permute(0, 3, 1, 2) 

        # for vit
        # B, L, C = features[2].shape
        # H, W = torch.sqrt(torch.tensor(L)).int(), torch.sqrt(torch.tensor(L)).int() 
        #features = [f.view(B, H, W, C).permute(0, 3, 1, 2) for f in features]
        
        
        for i in range(2):
            x = self.decoder_block[i](x)
            
        x = self.segmentationHad(x) 
    
        x = self.up(x)
        x = self.up(x) #B3HW

        return  x


class Segmenter(nn.Module):
    def __init__(self, backbone, num_classes, img_size):
        super(Segmenter, self).__init__ ()
        self.encoder = Encoder(backbone, img_size = img_size)  
        if backbone == 'vit_base_patch32_384' or backbone == 'vit_base_r50_s16_384' or backbone == 'vit_base_patch16_siglip_384':
            self.decoder = Decoder(768 , num_classes)
        elif backbone == 'vit_large_patch32_384'or backbone == 'vit_large_r50_s32_384':
            self.decoder = Decoder(1024, num_classes)
        elif backbone == 'vit_small_patch32_384':
            self.decoder = Decoder(384, num_classes)
        elif 'swin' in backbone:
            self.decoder = Decoder(1024, num_classes)
        

    
    def forward(self, x):
        
        features = self.encoder(x)        
        x = self.decoder(features)
    
        return x


import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
class Segmenter_segformer(nn.Module):
    def __init__(self, encoder_name="mit_b5", encoder_weights=None, in_channels=3, classes=3,
                 attention_channels=32):  # Added attention_channels
        super(Segmenter_segformer, self).__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        # 1. Intensity Normalization Layer (Physics-Informed)
        self.intensity_norm = IntensityNormalization()  # Apply normalization

        # # 2. Attention Mechanism (Contextual Information)
        # # Adapt attention to the output of the *last* decoder stage
        # if hasattr(self.unet.decoder, 'blocks'):
        #     last_block = self.unet.decoder.blocks[-1]
        #     if hasattr(last_block, 'convs'):
        #         decoder_output_channels = last_block.convs[1][0].out_channels  # Access conv layer output channels
        #     elif hasattr(last_block, 'conv1'):
        #         decoder_output_channels = last_block.conv1[0].out_channels  # Access conv layer output channels
        #     else:
        #         raise ValueError("Decoder block structure not recognized.")
        # elif hasattr(self.unet.decoder, 'convs'):
        #     decoder_output_channels = self.unet.decoder.convs[-1][0].out_channels  # Access conv layer output channels
        # else:
        #     raise ValueError("Decoder structure not recognized.")
        # self.attention = nn.Conv2d(decoder_output_channels, attention_channels, kernel_size=1)  # simple convolution
        # #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Apply Intensity Normalization
        x = self.intensity_norm(x)
        # Pass through the base Unet
        x = self.unet(x)
        return x



class IntensityNormalization(nn.Module):
    """
    Intensity normalization layer for X-ray angiography images.  Normalizes
    the intensity values to a more consistent range, handling potential
    variations in contrast and brightness.
    """
    def __init__(self, clip_percentile=1.0):
        super(IntensityNormalization, self).__init__()
        self.clip_percentile = clip_percentile

    def forward(self, x):
        # Flatten the image to calculate percentiles across the batch and spatial dimensions
        x_flat = x.contiguous().view(x.size(0), -1)#x.view(x.size(0), -1)

        # Calculate the lower and upper percentile values.
        lower_bound = torch.quantile(x_flat, self.clip_percentile / 100.0, dim=1, keepdim=True)
        upper_bound = torch.quantile(x_flat, (100.0 - self.clip_percentile) / 100.0, dim=1, keepdim=True)  # Corrected line

        # Clip the intensities to the calculated bounds
        x_clipped = torch.max(torch.min(x_flat, upper_bound), lower_bound)

        # Reshape back to the original image shape
        x_clipped = x_clipped.view(x.size())
        # Normalize the clipped intensities to the range [0, 1]
        x_normalized = (x_clipped - lower_bound.view(x.size(0), 1, 1, 1)) / (upper_bound - lower_bound).view(x.size(0), 1, 1, 1)
        return x_normalized
