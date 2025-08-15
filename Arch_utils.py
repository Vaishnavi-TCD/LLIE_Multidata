# from grey_queryattention import my_model
# import torch.nn as nn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
# # class ButterworthFilter:
# #     def __init__(self, order=2):
# #         self.order = order
        
# #     def butterworth(self, input_tensor, cutoff_frequency, high_pass=True):
# #         # Assuming input_tensor is of shape (B, C, H, W)
# #         # Print shape for debugging
       
# #         # Ensure input_tensor has the right number of dimensions
# #         if input_tensor.dim() != 4:
# #             raise ValueError(f"Expected 4D tensor, got {input_tensor.dim()}D tensor instead.")

# #         # Extract dimensions
# #         device = input_tensor.device
# #         B, C, H, W = input_tensor.shape

# #         # Now reshape correctly
# #         input_tensor = input_tensor.view(B * C, H, W)  # Flatten B and C  
# #          # Create mesh grid for frequencies
# #         u = torch.arange(H).view(-1, 1)  # Vertical frequency
# #         v = torch.arange(W).view(1, -1)  # Horizontal frequency
# #         D = torch.sqrt(u**2 + v**2)  # Euclidean distance in frequency domain
        
# #         # Normalize the frequency
# #         D0 = cutoff_frequency * math.sqrt(H**2 + W**2) / 2
        
# #         # Create Butterworth filter
# #         if high_pass:
# #             Filter = 1 / (1 + (D0 / (D + 1e-6))**(2 * self.order))
# #         else:
# #             Filter = 1 / (1 + (D / (D0 + 1e-6))**(2 * self.order))
        
# #         Filter=Filter.to(device)
# #         # Apply filter in frequency domain
# #         transformed = torch.fft.fft2(input_tensor)  # Apply FFT
# #         filtered = transformed * Filter.unsqueeze(0)  # Element-wise multiplication
# #         output_tensor = torch.fft.ifft2(filtered).real  # Inverse FFT
        
# #         return output_tensor.view(B, C, H, W)  # Reshape back



# # def default_conv(in_channels, out_channels, kernel_size, bias=True):
# #     return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


# # class REMP(nn.Module):
# #     """Height and width need to be divided by 16"""

# #     def __init__(self):
# #         super(REMP, self).__init__()

# #         # Learnable frequency parameter
# #         self.learnable_cutoff = nn.Parameter(torch.tensor(0.1))  # Initialize with a value

# #         # Other initializations...
# #         in_channels = 6
# #         channel = 16
# #         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
# #         self.conv2 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # on low disparity
# #         self.conv_start = nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2)

# #         # Instantiate Butterworth filter
# #         self.butterworth_filter = ButterworthFilter(order=2)

# #         self.my_model = my_model()
 
 
# #         self.final_conv = nn.Conv2d(16, 3, 3, 1, 1)

# #     def forward(self, rgb, grey):

# #         out1=self.my_model(rgb, grey)  #pretrained model


# #         out = torch.cat((out1, rgb), dim=1)  # [B, 6, H, W]
# #         out = self.conv1(out)  # [B, 16, H, W]
# #         gry_fine = self.conv2(grey)  # [B, 16, H, W]
# #         x = torch.cat((out, gry_fine), dim=1)  # [B, 32, H, W]
# #         x = self.conv_start(x)  # [B, 32, H, W]

# #         # Apply the Butterworth filter using the learnable cutoff frequency

# #         low = self.butterworth_filter.butterworth(x, self.learnable_cutoff.item(), high_pass=False)
# #         motif = self.butterworth_filter.butterworth(x, self.learnable_cutoff.item(), high_pass=True)
       
# #         # Combine features
# #         x = torch.mul((1 - motif), low) + torch.mul(motif, x)

# #         x = self.final_conv(x)  # [B, 1, H, W]

# #         disp = F.relu(out1 + x, inplace=True)  # [B, 1, H, W]

# #         return -disp



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, prelu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.prelu = nn.PReLU() if prelu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.prelu is not None:
            x = self.prelu(x)
        return x

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.query_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, y):
        b, c, h, w = x.shape
        k, v = self.qkv_conv(self.qkv(x)).chunk(2, dim=1)
        q = self.query_conv(self.query(y))
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x, y):
        b, c, h, w = x.shape
        x1 = self.attn(
            self.norm1(x.reshape(b, c, -1).transpose(-2, -1)).transpose(-2, -1).reshape(b, c, h, w),
            self.norm1(y.reshape(b, c, -1).transpose(-2, -1)).transpose(-2, -1).reshape(b, c, h, w)
        )
        x_out = x + self.attn(x1, y)
        x_out = x_out + self.ffn(self.norm2(x_out.reshape(b, c, -1).transpose(-2, -1)).transpose(-2, -1).reshape(b, c, h, w))
        return x_out

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class PhaseTrans:
    def __call__(self, x, y):
        fftn1 = torch.fft.fftn(x)
        fftn2 = torch.fft.fftn(y)
        out = torch.fft.ifftn(torch.abs(fftn2) * torch.exp(1j * fftn1.angle()))
        return out.real

class my_model(nn.Module):
    def __init__(self, num_blocks=[1, 1, 1, 1], num_heads=[1, 2, 4, 8], channels=[16, 32, 64, 128], expansion_factor=2.66):
        super(my_model, self).__init__()
        self.PhaseTrans = PhaseTrans()
        self.embed_conv_rgb = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.conv_grey = nn.Conv2d(1, channels[0], kernel_size=3, padding=1, bias=False)
        self.downsample1 = DownSample(channels[0])
        self.downsample2 = DownSample(channels[1])
        self.downsample3 = DownSample(channels[2])
        self.upsample1 = UpSample(channels[3])
        self.upsample2 = UpSample(channels[2])
        self.upsample3 = UpSample(channels[1])

        # Encoder and decoder blocks
        self.encoder1_1 = TransformerBlock(channels[0], num_heads[0], expansion_factor)
        self.encoder2_1 = TransformerBlock(channels[1], num_heads[1], expansion_factor)
        self.encoder3_1 = TransformerBlock(channels[2], num_heads[2], expansion_factor)
        self.encoder4_1 = TransformerBlock(channels[3], num_heads[3], expansion_factor)
        self.decoder1_1 = TransformerBlock(channels[2], num_heads[2], expansion_factor)
        self.decoder2_1 = TransformerBlock(channels[1], num_heads[1], expansion_factor)
        self.decoder3_1 = TransformerBlock(channels[0], num_heads[0], expansion_factor)
        self.output = nn.Conv2d(channels[0], 3, kernel_size=3, padding=1, bias=False)

#     def forward(self, RGB_input, grey_input):
        fo_rgb = self.embed_conv_rgb(RGB_input)
        fo_grey = self.conv_grey(grey_input)
        enc_rgb1 = self.encoder1_1(fo_rgb, fo_grey)
        enc_rgb2 = self.encoder2_1(self.downsample1(enc_rgb1), self.downsample1(fo_grey))
        enc_rgb3 = self.encoder3_1(self.downsample2(enc_rgb2), self.downsample2(self.downsample1(fo_grey)))
        enc_rgb4 = self.encoder4_1(self.downsample3(enc_rgb3), self.downsample3(self.downsample2(self.downsample1(fo_grey))))

        out1 = self.PhaseTrans(self.upsample1(enc_rgb4), enc_rgb3)
        dec_input3 = self.upsample1(enc_rgb4)
        out_dec3_1 = self.decoder1_1(dec_input3, dec_input3)
        out2 = self.PhaseTrans(self.upsample2(out_dec3_1), enc_rgb2)
        dec_input2 = self.upsample2(out_dec3_1)
        out_dec2_1 = self.decoder2_1(dec_input2, dec_input2)
        out3 = self.PhaseTrans(self.upsample3(out_dec2_1), enc_rgb1)
        dec_input1 = self.upsample3(out_dec2_1)
        out_dec1 = self.decoder3_1(dec_input1, dec_input1)
        return self.output(out_dec1)



# class MultiHeadEnhancement(nn.Module):
#     def __init__(self, unet):
#         super(MultiHeadEnhancement, self).__init__()

#         # U-Net Backbone
#         self.unet = unet

#         # Multi-Head Outputs (Each with separate color and contrast corrections)
#         self.blue_head = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 3, kernel_size=3, padding=1)
#         )

#         self.green_head = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 3, kernel_size=3, padding=1)
#         )

#         self.muddy_head = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 3, kernel_size=3, padding=1)
#         )

#         self.blurry_head = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 3, kernel_size=3, padding=1)
#         )

#         self.hazy_head = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 3, kernel_size=3, padding=1))


#         self.other_head = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 3, kernel_size=3, padding=1))

#         self.apply(self.init_weights)

#         # Initialize weights
#     def init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)

#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#         # Forward pass

#     def forward(self, x_rgb,x_grey):
#         # Get features from U-Net
#         # input dimension : batch, channels, height, width  and output dimesion: batch, channels, height, width
#         features = self.unet(x_rgb,x_grey)

#         # Pass through each head for specific correction
#         out_blue = self.blue_head(features)
#         out_green = self.green_head(features)
#         out_muddy = self.muddy_head(features)
#         out_blurry = self.blurry_head(features)
#         out_hazy = self.hazy_head(features)
#         out_other = self.other_head(features)

#         # Return all heads' outputs
#         return out_blue, out_green, out_muddy, out_blurry, out_hazy, out_other



# class GatingLayer(nn.Module):
#     def __init__(self):
#         super(GatingLayer, self).__init__()
#         self.fc = nn.Linear(3, 6)  # 4 heads
#         self.prelu = nn.PReLU()
#         self.softmax = nn.Softmax(dim=1)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Replaces fixed pooling

#     def forward(self, features):
#         features = self.pool(features).view(features.size(0), -1)  # Adaptive average pooling
#         features = self.prelu(features)
#         # print(f"Features after pooling and reshaping: {features.shape}")
#         # print(f"Features: {features}")
#         weights = self.softmax(self.fc(features))
#         # print(f"Weights: {weights}")
#         return weights

# # Apply weights to outputs
# def weighted_combination(outputs, weights):
#     """
#     Combines the outputs with given weights using element-wise multiplication.
#     Assumes weights are broadcastable to the output shapes.
#     """
#     # Reshape weights to ensure correct broadcasting (1, 6, 1, 1)
#     weights = weights.view(weights.size(0), weights.size(1), 1, 1)
#     # Perform the weighted sum of outputs
#     return sum(w * o for w, o in zip(weights.split(1, dim=1), outputs))


# # # Initialize the main model components
# # model = my_model()
# # multi_head_model = MultiHeadEnhancement(model)
# # gating_layer = GatingLayer()

# # # Set model to evaluation mode (optional for testing)
# # multi_head_model.eval()
# # gating_layer.eval()

# # # Create dummy inputs: an RGB image and a grayscale image (batch size of 1, height and width of 256)
# # RGB_input = torch.randn(1, 3, 256, 256)  # RGB image with 3 channels
# # grey_input = torch.randn(1, 1, 256, 256) # Grayscale image with 1 channel

# # # Forward pass through `my_model`
# # with torch.no_grad():
# #     features = model(RGB_input, grey_input)
# #     print(f"Features from my_model: {features.shape}")

# #     # Forward pass through `MultiHeadEnhancement` for multi-head outputs
# #     out_blue, out_green, out_muddy, out_blurry, out_hazy, out_other = multi_head_model(RGB_input, grey_input)
# #     print(f"Output Blue: {out_blue.shape}")
# #     print(f"Output Green: {out_green.shape}")
# #     print(f"Output Muddy: {out_muddy.shape}")
# #     print(f"Output Blurry: {out_blurry.shape}")
# #     print(f"Output Hazy: {out_hazy.shape}")
# #     print(f"Output Other: {out_other.shape}")

# #     # Calculate head weights using `GatingLayer`
# #     weights = gating_layer(features)
# #     print(f"Head weights: {weights}")

# #     # Combine the outputs using `weighted_combination`
# #     final_output = weighted_combination([out_blue, out_green, out_muddy, out_blurry,out_hazy,out_other], weights[0])
# #     print(f"Final Output: {final_output.shape}")



import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadEnhancement(nn.Module):
    def __init__(self, unet):
        super(MultiHeadEnhancement, self).__init__()

        # U-Net Backbone (assuming `unet` is a pre-defined U-Net model instance)
        self.unet = unet

        # Task-specific heads
        self.color_correction_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        
        self.contrast_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        
        self.haze_removal_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        
        self.sharpness_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        
        self.illumination_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        self.init_weights()

    def init_weights(self):
        # Optional: Initialize weights for better convergence
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_rgb, x_grey=None):
        # Extract features from U-Net
        features = self.unet(x_rgb)  # Assuming output channels of U-Net are 64

        # Pass features through each enhancement head
        color_corrected = self.color_correction_head(features)
        contrast_enhanced = self.contrast_head(features)
        haze_removed = self.haze_removal_head(features)
        sharpness_enhanced = self.sharpness_head(features)
        illumination_adjusted = self.illumination_head(features)

        # Combine outputs if needed or return each head's output separately
        return {
            "color_corrected": color_corrected,
            "contrast_enhanced": contrast_enhanced,
            "haze_removed": haze_removed,
            "sharpness_enhanced": sharpness_enhanced,
            "illumination_adjusted": illumination_adjusted
        }
        
class CombineOutputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterDict({
            "color": nn.Parameter(torch.tensor(1.0)),
            "contrast": nn.Parameter(torch.tensor(1.0)),
            "haze": nn.Parameter(torch.tensor(1.0)),
            "sharpness": nn.Parameter(torch.tensor(1.0)),
            "illumination": nn.Parameter(torch.tensor(1.0)),
        })

    def forward(self, outputs):
        combined_output = (self.weights["color"] * outputs["color_corrected"] +
                           self.weights["contrast"] * outputs["contrast_enhanced"] +
                           self.weights["haze"] * outputs["haze_removed"] +
                           self.weights["sharpness"] * outputs["sharpness_enhanced"] +
                           self.weights["illumination"] * outputs["illumination_adjusted"])
        return combined_output
