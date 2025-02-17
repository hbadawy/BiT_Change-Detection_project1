
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, device=None):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False, device=device),
                         nn.BatchNorm2d(in_channels, device=device),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, device=device),
                         )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn , device=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim, device=device)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn, device=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim, device=device)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, device=device),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, device=device),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


##################### Cross Attention #####################
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False, device=device)
        self.to_k = nn.Linear(dim, inner_dim, bias=False, device=device)
        self.to_v = nn.Linear(dim, inner_dim, bias=False, device=device)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, device=device),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):     # x: input to decoder,      m: coming from encoder
        b, n, _, h = *x.shape, self.heads
        # print ("x.shape", x.shape)                 # [1, 8, 64],  [batch, seq_len, dim]
        # print ("m.shape", m.shape)                 # [1, 8, 64],  [batch, seq_len, dim]

        q = self.to_q(x)      # input to decoder
        k = self.to_k(m)      # coming from encoder
        v = self.to_v(m)      # coming from encoder
        # print ("q.shape", q.shape)                 # [1, 8, 512] , [batch, seq_len, inner_dim]
        # print ("k.shape", k.shape)                 # [1, 8, 512] , [batch, seq_len, inner_dim]
        # print ("v.shape", v.shape)                 # [1, 8, 512] , [batch, seq_len, inner_dim]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])
        # print ("q.shape", q.shape)                 # [1, 8, 8, 64], [batch, heads, seq_len, dim_head]
        # print ("k.shape", k.shape)                 # [1, 8, 8, 64], [batch, heads, seq_len, dim_head]
        # print ("v.shape", v.shape)                 # [1, 8, 8, 64], [batch, heads, seq_len, dim

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # print ("dots.shape", dots.shape)           # [1, 8, 8, 8], [batch, heads, seq_len, seq_len]

        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v) 
        # print ("out.shape", out.shape)             # [1, 8, 8, 64], [batch, heads, seq_len, dim_head]

        out = rearrange(out, 'b h n d -> b n (h d)')
        # print ("out.shape", out.shape)             # [1, 8, 512], [batch, seq_len, inner_dim]

        out = self.to_out(out)
        # print ("to_out.shape", out.shape)             # [1, 8, 64], [batch, seq_len, dim]
        # vis_tmp2(out)

        return out

##################### Attention #####################
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., device=None):
        super().__init__()
        inner_dim = dim_head *  heads             # 64 * 8 = 512
        self.heads = heads                        # 8
        self.scale = dim ** -0.5                  # 64 ** -0.5 = 0.125

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False, device=device)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, device=device),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        # print ("x.shape", x.shape)                 # [1, 8, 64],  [batch, seq_len, dim]
        qkv = self.to_qkv(x)
        # print ("qkv.shape", qkv.shape)             # [1, 8, 512*3], [batch, seq_len, inner_dim * 3]

        qkv = qkv.chunk(3, dim = -1)
        # print ("qkv[0].shape", qkv[0].shape)       # [1, 8, 512] , [batch, seq_len, inner_dim]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # print ("q.shape", q.shape)                 # [1, 8, 8, 64], [batch, heads, seq_len, dim_head]
        # print ("k.shape", k.shape)                 # [1, 8, 8, 64], [batch, heads, seq_len, dim_head]
        # print ("v.shape", v.shape)                 # [1, 8, 8, 64], [batch, heads, seq_len, dim_head]

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # print ("dots.shape", dots.shape)           # [1, 8, 8, 8], [batch, heads, seq_len, seq_len]
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        # print ("attn.shape", attn.shape)           # [1, 8, 8, 8], [batch, heads, seq_len, seq_len]


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # print ("out.shape", out.shape)             # [1, 8, 8, 64], [batch, heads, seq_len, dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print ("out.shape", out.shape)             # [1, 8, 512], [batch, seq_len, inner_dim]
        
        out = self.to_out(out)
        # print ("to_out.shape", out.shape)             # [1, 8, 64], [batch, seq_len, dim]
        return out

##################### Transformer #####################
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, device=None):  #depth is the depth of transformer layers
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, device=device))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, device=device)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

##################### Transformer Decoder #####################
class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True, device=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax, device=device))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, device=device)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x
    


if __name__ == '__main__':
    # model = TransformerDecoder(64, 3, 8, 64, 64, 0.1)
    # x = torch.randn(1, 10, 64)
    # m = torch.randn(1, 10, 64)
    # out = model(x, m)
    # print(out.shape)  # torch.Size([1, 10, 64])

    # model = TwoLayerConv2d(64, 64)
    # x = torch.randn(1, 64, 224, 224)
    # out = model(x)
    # print(out.shape)  # torch.Size([1, 64, 224, 224])

    # model = FeedForward(64, 64)
    # x = torch.randn(1, 64)
    # out = model(x)
    # print(out.shape)  # torch.Size([1, 64])

    # model = Attention(64, 8, 64)
    # x = torch.randn(1, 8, 64)
    # out = model(x)
    # print(out.shape)  # torch.Size([1, 10, 64])

    # model = FeedForward(64, 128)
    # x = torch.randn(1, 8, 64)
    # out = model(x)
    # print(out.shape)  # torch.Size([1, 64])

    # model = Transformer(64, 3, 8, 64, 64, 0.1)  #dim, depth, heads, dim_head, mlp_dim, dropout
    # x = torch.randn(1, 8, 64)
    # out = model(x)
    # print(out.shape)  # torch.Size([1, 8, 64])

    # model = Cross_Attention(64, 8, 64)
    # x = torch.randn(1, 32, 64)       # x has different size than m
    # m = torch.randn(1, 8, 64)
    # out = model(x, m)
    # print(out.shape)  # torch.Size([1, 32, 64])

    model = TransformerDecoder(64, 3, 8, 64, 64, 0.1)   # dim, depth, heads, dim_head, mlp_dim, dropout
    x = torch.randn(1, 32, 64)
    m = torch.randn(1, 8, 64)
    out = model(x, m)
    print(out.shape)  # torch.Size([1, 32, 64])