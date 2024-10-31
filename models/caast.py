import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from copy import deepcopy
from timm.models.layers import to_2tuple,trunc_normal_
from timm.models.vision_transformer import Block
import torch.nn.functional as F
from .functions import ReverseLayerF

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        


    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True, mix_beta=None, domain_label_dim=527,device_label_dim=527,dropout_rate = 0.1):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        # self.v.blocks = nn.ModuleList([CustomTransformerBlock(...) for _ in range(num_blocks)])

        self.final_feat_dim = 768
        self.mix_beta = mix_beta
        self.dropout_rate = dropout_rate
        
        # original_embedding_dim 설정
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.original_embedding_dim = 192
            elif model_size == 'small224':
                self.original_embedding_dim = 384
            else:  # base224 or base384
                self.original_embedding_dim = 768
        else:
            self.original_embedding_dim = 768

        # f_dim 계산
        self.f_dim, _ = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        
        self.freq_attn_weights = nn.Parameter(torch.randn(self.original_embedding_dim, self.f_dim))


        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Dropout(self.dropout_rate),nn.Linear(self.original_embedding_dim, label_dim))
            self.domain_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),nn.Dropout(self.dropout_rate), nn.Linear(self.original_embedding_dim, domain_label_dim)) # added for domain adapation
            self.device_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, device_label_dim))
            
            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            out_dir = './pretrained_models/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            
            if os.path.exists(os.path.join(out_dir, 'audioset_10_10_0.4593.pth')) == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out=os.path.join(out_dir, 'audioset_10_10_0.4593.pth'))
            
            sd = torch.load(os.path.join(out_dir, 'audioset_10_10_0.4593.pth'), map_location=device)
            # sd = torch.load(os.path.join('./save/icbhi_ast_ce_jmir_ast_stethoscope_fold0/best.pth'), map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False, domain_label_dim=527,device_label_dim=527)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]  #1024
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            self.domain_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, domain_label_dim))
            self.device_mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, device_label_dim))# added for domain adapation
            
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))   # test input (1,1,128,400) original embedding 1214 stride()
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def square_patch(self, patch, hw_num_patch):
        h, w = hw_num_patch
        B, _, dim = patch.size()
        square = patch.reshape(B, h, w, dim)
        return square

    def flatten_patch(self, square):
        B, h, w, dim = square.shape
        patch = square.reshape(B, h * w, dim)
        return patch

        
        
        


    @autocast()
    def forward(self, x, y=None, y2=None, da_index=None, patch_mix=False, time_domain=False, args=None, alpha=None,training=False,return_attention_maps=False,mc_dropout=False):
        """
        :param x: the input spectrogram, expected shape: (batch_size, 1, time_frame_num, frequency_bins), e.g., (12, 1, 1024, 128)
        :return: prediction
        """
        

        x = x.transpose(2, 3) # B, 1, F, T

        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        

        
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
 
        x = self.v.pos_drop(x)
 
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        # x = self.mlp_head(x)
    
            

        return x




# class SimpleCNN(nn.Module):
#     """Simple CNN encoder for generating secondary embeddings"""
#     def __init__(self, in_channels=1, embed_dim=768):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(128, embed_dim),
#             nn.LayerNorm(embed_dim)
#         )
    
#     def forward(self, x):
#         return self.encoder(x)
    
    
# class SimpleCNN(nn.Module):
#    def __init__(self, in_channels=1, embed_dim=768):
#        super().__init__()
#        self.encoder = nn.Sequential(
#            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
#            nn.ReLU(),
#            nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1), 
#            nn.ReLU(),
#            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),
#            nn.ReLU(),
#            nn.AdaptiveAvgPool2d((1, 1)),
#            nn.Flatten(),
#        )
       
#        self.projection = nn.Sequential(
#            nn.Linear(512, 1024),
#            nn.ReLU(),
#            nn.Linear(1024, 1024),
#            nn.ReLU(), 
#            nn.Linear(1024, embed_dim),
#            nn.LayerNorm(embed_dim)
#        )
       
#    def forward(self, x):
#        x = self.encoder(x)
#        x = self.projection(x)
#        return x
   


class AudioCNN(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        
        # Simple CNN structure
        self.encoder = nn.Sequential(
            # Initial conv layer
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Final pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Simple projection head
        self.projection = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x):
        # Ensure input is in shape (B, 1, T)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Feature extraction
        x = self.encoder(x)
        x = x.squeeze(-1)
        
        # Projection
        x = self.projection(x)
        
        return x

class DualCrossAttention(nn.Module):
    """Implements dual cross attention mechanism between AST and CNN features"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for AST features
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v1 = nn.Linear(dim, dim, bias=qkv_bias)

        # Projections for CNN features
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, ast_features, cnn_features):
        B = ast_features.shape[0]
        N_ast = ast_features.shape[1]
        N_cnn = cnn_features.shape[1]
        
        # First cross attention: AST query, CNN key/value
        q1 = self.q1(ast_features).reshape(B, N_ast, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k1 = self.k1(cnn_features).reshape(B, N_cnn, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v1 = self.v1(cnn_features).reshape(B, N_cnn, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        cross_out1 = (attn1 @ v1).transpose(1, 2).reshape(B, N_ast, -1)
        
        # Second cross attention: CNN query, AST key/value
        q2 = self.q2(cnn_features).reshape(B, N_cnn, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k2 = self.k2(ast_features).reshape(B, N_ast, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v2 = self.v2(ast_features).reshape(B, N_ast, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        cross_out2 = (attn2 @ v2).transpose(1, 2).reshape(B, N_cnn, -1)
        
        return cross_out1, cross_out2

class GateFusion(nn.Module):
    """Gate-based fusion mechanism for combining different feature representations"""
    def __init__(self, dim):
        super().__init__()
        self.gate1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, ast_feat, cross_feat1, cross_feat2):
        gate1 = self.gate1(ast_feat)
        gate2 = self.gate2(cross_feat1)
        
        fused = ast_feat + gate1 * cross_feat1 + gate2 * cross_feat2
        return self.norm(fused)

class DualAttentionAST(ASTModel):
    """
    Enhanced AST model with dual cross attention mechanism
    Inherits from base ASTModel and adds parallel CNN processing and cross attention
    """
    def __init__(self, *args, fusion_type='concat', **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize SimpleCNN for parallel processing
        # self.cnn = SimpleCNN(in_channels=1, embed_dim=self.original_embedding_dim)
        
        self.cnn = AudioCNN(embed_dim=self.original_embedding_dim)
        
        # Dual cross attention mechanism
        self.cross_attention = DualCrossAttention(
            dim=self.original_embedding_dim,
            num_heads=8,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1
        )
        
        # Feature fusion mechanism
        self.fusion_type = fusion_type
        if fusion_type == 'gate':
            self.fusion = GateFusion(self.original_embedding_dim)

    @autocast()
    def forward(self, x,raw_signal):
        # Get original AST features
        ast_features = super().forward(x)  # Use parent class's forward method
        
        # CNN path for secondary features
        # x_cnn = x.transpose(2, 3)  # Match AST input format
        cnn_features = self.cnn(raw_signal)
        cnn_features = cnn_features.unsqueeze(1)  # Add sequence dimension
        
        # Expand AST features to match sequence dimension
        ast_features = ast_features.unsqueeze(1)
        
        # Apply dual cross attention
        cross_feat1, cross_feat2 = self.cross_attention(ast_features, cnn_features)
        
        # Feature fusion based on specified type
        if self.fusion_type == 'concat':
            # Squeeze sequence dimensions
            ast_features = ast_features.squeeze(1)
            cross_feat1 = cross_feat1.squeeze(1)
            cross_feat2 = cross_feat2.squeeze(1)
            # Concatenate all features
            final_features = torch.cat([ast_features, cross_feat1, cross_feat2], dim=1)
        else:  # gate fusion
            final_features = self.fusion(
                ast_features.squeeze(1),
                cross_feat1.squeeze(1),
                cross_feat2.squeeze(1)
            )
            
        return final_features