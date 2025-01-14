# this is atm for input_size of 1024*1024
# '''
from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor

class WindowAttention(nn.Module):
    """窗口注意力机制"""

    def __init__(self, dim, window_size=256):
        super().__init__()
        self.window_size = window_size
        self.scale = dim ** -0.5

    def forward(self, q, k, v):
        B, N, C = q.shape  # 假设q,k,v具有相同的batch_size和seq_len

        # 分割成窗口
        num_windows = N // self.window_size
        q = q.view(B, num_windows, self.window_size, C)
        k = k.view(B, num_windows, self.window_size, C)
        v = v.view(B, num_windows, self.window_size, C)

        # 在每个窗口内计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # 恢复原始形状
        out = out.view(B, N, C)
        return out

@MODELS.register_module()
class atms(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.num_heads = num_heads
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        # new
        self.mlp_projection = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims // 2, self.token_length)
        )
        # self.cross_attention = nn.MultiheadAttention(
        #     embed_dim=self.token_length,
        #     num_heads=self.num_heads,
        #     batch_first=True
        # )
        self.window_attn = WindowAttention(self.token_length)
        self.linear_proj = nn.Linear(99,100)
        self.linear_proj1 = nn.Linear(1024,4096)
        # self.linear_proj2 = nn.Linear(100, 1024)
        # self.cross_attention2 = nn.MultiheadAttention(
        #     embed_dim=self.token_length,
        #     num_heads=self.num_heads,
        #     batch_first=True
        # )


        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        # feats  4096,4,1024   tokens 100,1024
        # output  n b m ; 1024,4,100
        seq_len, batch_size, _ = feats.size()
        # feats = feats.permute(1, 0, 2)
        feats_proj = self.mlp_projection(feats)
        q = feats_proj.permute(1, 0, 2)  # (2, 4096, 100)
        k = feats_proj.permute(1, 0, 2)  # (2, 4096, 100)
        v = self.linear_proj1(tokens).unsqueeze(0).expand(batch_size, -1, -1)
        attn = self.window_attn(q, k, v).permute(1, 0, 2)# (4096, 4, 100)
        del q, k, v
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)

        delta_f = torch.einsum(
                    "nbm,mc->nbc",
                      attn[:,:,1:],
                    self.mlp_token2feat(tokens[1:,:]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f

@MODELS.register_module()
class LoRAatms(atms):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
# '''


'''
# this is atm for input_size of 512*512 no window attn
from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor


@MODELS.register_module()
class atms(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.num_heads = num_heads
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        # new
        self.mlp_projection = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims // 2, self.token_length)
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.token_length,
            num_heads=self.num_heads,
            batch_first=True
        )
        self.linear_proj = nn.Linear(99,100)
        self.linear_proj2 = nn.Linear(100, 1024)
        self.cross_attention2 = nn.MultiheadAttention(
            embed_dim=self.token_length,
            num_heads=self.num_heads,
            batch_first=True
        )


        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        # feats  1024,4,1024   tokens 100,1024

        seq_len, batch_size, _ = feats.size()
        feats_proj = self.mlp_projection(feats)
        q = feats_proj.permute(1, 0, 2)  # (4, 1024, 100)
        k = feats_proj.permute(1, 0, 2)  # (4, 1024, 100)
        v = tokens.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1)  # (4, 1024, 100)
        attn = self.cross_attention(q, k, v)[0].permute(1, 0, 2)# (1024, 4, 100)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        # attn 1024, 2, 100
        tokens = self.linear_proj(self.mlp_token2feat(tokens[1:, :]).permute(1,0)).permute(1,0)
        # 99,1024
        attn = self.linear_proj(attn[:, :, 1:])
           # 准备交叉注意力的输入
        q = attn.permute(1, 0, 2)  # (4, 1024, 100)
        k = attn.permute(1, 0, 2)  # (4, 1024, 100)
        v = tokens.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1)  # (4, 1024, 100)
        delta_f = self.cross_attention2(q, k, v)[0].permute(1, 0, 2)  # (1024, 4, 100)
        delta_f = self.linear_proj2(delta_f)
        delta_f = self.mlp_delta_f(delta_f + feats)
        # 1024,4,1024
        return delta_f

@MODELS.register_module()
class LoRAatms(atms):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
'''


'''

#atm window_attn 512*512

from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor

class WindowAttention(nn.Module):
    """窗口注意力机制"""

    def __init__(self, dim, window_size=256):
        super().__init__()
        self.window_size = window_size
        self.scale = dim ** -0.5

    def forward(self, q, k, v):
        B, N, C = q.shape  # 假设q,k,v具有相同的batch_size和seq_len

        # 分割成窗口
        num_windows = N // self.window_size
        q = q.view(B, num_windows, self.window_size, C)
        k = k.view(B, num_windows, self.window_size, C)
        v = v.view(B, num_windows, self.window_size, C)

        # 在每个窗口内计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # 恢复原始形状
        out = out.view(B, N, C)
        return out

@MODELS.register_module()
class atms(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.num_heads = num_heads
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        # new
        self.mlp_projection = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims // 2, self.token_length)
        )
        self.window_attn1 = WindowAttention(self.token_length)
        self.window_attn2 = WindowAttention(self.token_length)
        self.linear_proj = nn.Linear(99,100)
        self.linear_proj2 = nn.Linear(100, 1024)
     
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        # feats  1024,4,1024   tokens 100,1024

        seq_len, batch_size, _ = feats.size()
        feats_proj = self.mlp_projection(feats)
        q = feats_proj.permute(1, 0, 2)  # (4, 1024, 100)
        k = feats_proj.permute(1, 0, 2)  # (4, 1024, 100)
        v = tokens.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1)  # (4, 1024, 100)
        # v = self.linear_proj1(tokens).unsqueeze(0).expand(batch_size, -1, -1)

        attn = self.window_attn1(q, k, v).permute(1, 0, 2)# (4096, 4, 100)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        # attn 1024, 2, 100
        tokens = self.linear_proj(self.mlp_token2feat(tokens[1:, :]).permute(1,0)).permute(1,0)
        # 99,1024
        attn = self.linear_proj(attn[:, :, 1:])
           # 准备交叉注意力的输入
        q = attn.permute(1, 0, 2)  # (4, 1024, 100)
        k = attn.permute(1, 0, 2)  # (4, 1024, 100)
        v = tokens.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1)  # (4, 1024, 100)
        delta_f = self.window_attn2(q, k, v).permute(1, 0, 2)# (4096, 4, 100)
        delta_f = self.linear_proj2(delta_f)
        delta_f = self.mlp_delta_f(delta_f + feats)
        # 1024,4,1024
        return delta_f

@MODELS.register_module()
class LoRAatms(atms):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
'''