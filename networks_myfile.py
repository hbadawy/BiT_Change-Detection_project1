
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

# import functools
from einops import rearrange

import torchvision.models as models
from help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d


###############################################################################
# main Functions
###############################################################################

class ResNet(nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=4, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True, device=None):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)#,
                                          #replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc, device=device)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1, device=device)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
            # print ("x.shape after upsamplex2:", x.shape)   # torch.Size([1, 32, 32, 32])
        x = self.upsamplex4(x)
        # print ("x.shape after upsamplex4:", x.shape)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        # print ("x_8.shape:", x_8.shape)   # torch.Size([1, 512, 16, 16])
        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
            # print ("x.shape after upsamplex2:", x.shape)   # torch.Size([1, 512, 16, 16])
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x

class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """
    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=4,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True, device=None):
        super(BASE_Transformer, self).__init__(input_nc, output_nc,backbone=backbone,
                                             resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x, device=device
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False, device=device)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2*dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, 32))
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0, device=device)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                            heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                                      softmax=decoder_softmax, device=device)
    
    ############### _forward_semantic_tokens ####################
    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        # print ("spatial_attention.shape:", spatial_attention.shape)   # torch.Size([1, 4, 16, 16])
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        # print ("spatial_attention.shape after view:     ", spatial_attention.shape)   # torch.Size([1, 4, 256])
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        # print ("spatial_attention.shape after softmax:           ", spatial_attention.shape)   # torch.Size([1, 4, 256])
        x = x.view([b, c, -1]).contiguous()
        # print ("x.shape after view:                        ", x.shape)          # torch.Size([1, 32, 256])
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        # print ("tokens.shape after einsum:                  ", tokens.shape)   # torch.Size([1, 4, 32])

        return tokens

    ############### _forward_reshape_tokens ####################
    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    ############### _forward_transformer ####################
    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    ############### _forward_transformer_decoder ####################
    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        # print ("x.shape after rearrange in decoder  :      ", x.shape)         # torch.Size([1, 256, 32])
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    ############### _forward_simple_decoder ####################
    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h,w,b,l,c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    ############### forward ####################
    def forward(self, x1, x2):
        # forward backbone resnet
        # print ("x1.shape:", x1.shape)   # torch.Size([1, 3, 256, 256])
        # print ("x2.shape:", x2.shape)   # torch.Size([1, 3, 256, 256])

        x1 = self.forward_single(x1)                              ####### resnet18
        x2 = self.forward_single(x2)                              ####### resnet18
        # print ("x1.shape:", x1.shape)   # torch.Size([1, 32, 16, 16])
        # print ("x2.shape:", x2.shape)   # torch.Size([1, 32, 16, 16])

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
            # print ("token1.shape:", token1.shape)   # torch.Size([1, 4, 32])
            # print ("token2.shape:", token2.shape)   # torch.Size([1, 4, 32])
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            # print ("self.tokens_.shape:", self.tokens_.shape)   # torch.Size([1, 8, 32])
            self.tokens = self._forward_transformer(self.tokens_)
            # print ("self.tokens.shape after transformer  :    ", self.tokens.shape)   # torch.Size([1, 8, 32])  
            token1, token2 = self.tokens.chunk(2, dim=1)
            # print ("token1.shape after chunk:", token1.shape)   # torch.Size([1, 4, 32])
            # print ("token2.shape after chunk:", token2.shape)   # torch.Size([1, 4, 32])
        
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            # print ("x1.shape after transformer decoder:             ", x1.shape)   # torch.Size([1, 32, 16, 16])
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
            # print ("upsampledddddddddddddddddddddddddddddddddddddddddddd")
        x = self.upsamplex4(x)
        # x = self.upsamplex4(x) 
        x = self.upsamplex2(x)       

        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        x = self.sigmoid(x)
        # print ("x.shape after sigmoid:                          ", x.shape)   # torch.Size([1, 1, 256, 256])
        return x
    


if __name__ == '__main__':
    
    model = BASE_Transformer(input_nc=3, output_nc=1, with_pos='learned')
    x = torch.randn(8, 3, 256, 256)
    y = torch.randn(8, 3, 256, 256)
    out = model(x, y)
    print ("out.shape:", out.shape)   # torch.Size([1, 1, 64, 64]) !!!!!!

    # print ("Done!")
    import torchinfo
    torchinfo.summary(model, input_size=[(8,3,256,256),(8,3,256,256)])