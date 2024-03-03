import math
import numpy as np
import torch
import torch.nn as nn


def drop_path(x,drop_prob:float=0.0,training:bool=False):
    if drop_prob==0.0 or not training:
        return x
    keep_prob=1-drop_prob
    shape=(x.shape[0],)+(1,)*(x.ndim -1)
    random_tensor=keep_prob=torch.rand(shape,dtype=x.dtype,device=x.device)
    random_tensor.floor_()
    output=x.div(keep_prob)*random_tensor
    return output

class DropPath(nn.Module):
    
    def __init__(self,drop_prob=None):
        super(DropPath,self).__init__()
        self.drop_prob=drop_prob

    def forward(self,x):
        return drop_path(x,self.drop_prob,self.training)
    
class Mlp(nn.Module):
    def __init__(self,in_features:torch.Tensor,hidden_features:torch.Tensor=None,out_features:torch.Tensor=None,act_layer=nn.GELU,drop:float=0.0,):
        super().__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self,dim:int,num_heads:int=8,qkv_bias:bool=False,qk_scale=None,attn_drop:float=0.0,proj_drop:float=0.0):
        super().__init__()
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim**-0.5
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)

    def forward(self,x):
        B,N,C=x.shape
        qkv=(self.qkv(x)
             .reshape(B,N,3,self.num_heads,C//self.num_heads)
             .permute(2,0,3,1,4)
             )
        q,k,v=qkv[0],qkv[1],qkv[2]
        attn=(q@k.transpose(-2,-1))*self.scale
        attn=attn.softmax(dim=-1)
        attn=self.attn_drop(attn)
        x=(attn@v).transpose(1,2).reshape(B,N,C)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x,attn
    
class Block(nn.Module):
    def __init__(self,dim:int,num_heads:int,mlp_ratio:float=4.0,qkv_bias:bool=False,qk_scale=None,drop:float=0.0,attn_drop:float=0.0,drop_path:float=0.0,act_layer=nn.GELU,norm_layer=nn.LayerNorm,):
        super().__init__()
        self.norm1=norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()