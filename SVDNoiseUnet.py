import torch
import torch.nn as nn
import einops

from torch.nn import functional as F
from torch.jit import Final
from timm.layers import use_fused_attn
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, get_act_layer
from abc import abstractmethod
from NoiseTransformer import NoiseTransformer
from einops import rearrange
__all__ = ['SVDNoiseUnet', 'SVDNoiseUnet_Concise']

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SVDNoiseUnet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, resolution=(128,96)): # resolution = size // 8
        super(SVDNoiseUnet, self).__init__()

        _in_1 = int(resolution[0] * in_channels // 2)
        _out_1 = int(resolution[0] * out_channels // 2)
        
        _in_2 = int(resolution[1] * in_channels // 2)
        _out_2 = int(resolution[1] * out_channels // 2)
        self.mlp1 = nn.Sequential(
            nn.Linear(_in_1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out_1),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(_in_2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out_2),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(_in_2, _out_2),
        )

        self.attention = Attention(_out_2)

        self.bn = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(192)

        self.mlp4 =  nn.Sequential(
            nn.Linear(_out_2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, _out_2),
        )
        self.ffn = nn.Sequential(
            nn.Linear(256, 384),  # Expand
            nn.ReLU(inplace=True),
            nn.Linear(384, 192)   # Reduce to target size
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(256, 384),  # Expand
            nn.ReLU(inplace=True),
            nn.Linear(384, 192)   # Reduce to target size
        )
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((256, 192))

    def forward(self, x, residual=False):
        b, c, h, w = x.shape
        x = einops.rearrange(x, "b (a c)h w ->b (a h)(c w)", a=2,c=2) # x -> [1, 256, 256]
        U, s, V = torch.linalg.svd(x) # U->[b 256 256], s-> [b 256], V->[b 256 256]
        U_T = U.permute(0, 2, 1)
        U_out = self.ffn(self.mlp1(U_T))
        U_out = self.bn(U_out)
        U_out = U_out.transpose(1, 2)
        U_out = self.ffn2(U_out)  # [b, 256, 256] -> [b, 256, 192]
        U_out = self.bn2(U_out)
        U_out = U_out.transpose(1, 2)
        # U_out = self.bn(U_out)
        V_out = self.mlp2(V)
        s_out = self.mlp3(s).unsqueeze(1)  # s -> [b, 1, 256]  => [b, 256, 256]
        out = U_out + V_out + s_out
        # print(out.size())
        out = out.squeeze(1)
        out = self.attention(out).mean(1)
        out = self.mlp4(out) + s
        diagonal_out = torch.diag_embed(out)
        padded_diag = F.pad(diagonal_out, (0, 0, 0, 64), mode='constant', value=0)  # Shape: [b, 1, 256, 192]
        pred = U @ padded_diag @ V
        return einops.rearrange(pred, "b (a h)(c w) -> b (a c) h w", a=2,c=2)
    
class SVDNoiseUnet64(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, resolution=64): # resolution = size // 8
        super(SVDNoiseUnet64, self).__init__()

        _in = int(resolution * in_channels // 2)
        _out = int(resolution * out_channels // 2)
        self.mlp1 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(_in, _out),
        )

        self.attention = Attention(_out)

        self.bn = nn.BatchNorm2d(_out)

        self.mlp4 =  nn.Sequential(
            nn.Linear(_out, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, _out),
        )

    def forward(self, x, residual=False):
        b, c, h, w = x.shape
        x = einops.rearrange(x, "b (a c)h w ->b (a h)(c w)", a=2,c=2) # x -> [1, 256, 256]
        U, s, V = torch.linalg.svd(x) # U->[b 256 256], s-> [b 256], V->[b 256 256]
        U_T = U.permute(0, 2, 1)
        out = self.mlp1(U_T) + self.mlp2(V) + self.mlp3(s).unsqueeze(1) # s -> [b, 1, 256]  => [b, 256, 256]
        out = self.attention(out).mean(1)
        out = self.mlp4(out) + s
        pred = U @ torch.diag_embed(out) @ V
        return einops.rearrange(pred, "b (a h)(c w) -> b (a c) h w", a=2,c=2)
    


class SVDNoiseUnet128(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, resolution=128): # resolution = size // 8
        super(SVDNoiseUnet128, self).__init__()

        _in = int(resolution * in_channels // 2)
        _out = int(resolution * out_channels // 2)
        self.mlp1 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(_in, _out),
        )

        self.attention = Attention(_out)

        self.bn = nn.BatchNorm2d(_out)

        self.mlp4 =  nn.Sequential(
            nn.Linear(_out, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, _out),
        )

    def forward(self, x, residual=False):
        b, c, h, w = x.shape
        x = einops.rearrange(x, "b (a c)h w ->b (a h)(c w)", a=2,c=2) # x -> [1, 256, 256]
        U, s, V = torch.linalg.svd(x) # U->[b 256 256], s-> [b 256], V->[b 256 256]
        U_T = U.permute(0, 2, 1)
        out = self.mlp1(U_T) + self.mlp2(V) + self.mlp3(s).unsqueeze(1) # s -> [b, 1, 256]  => [b, 256, 256]
        out = self.attention(out).mean(1)
        out = self.mlp4(out) + s
        pred = U @ torch.diag_embed(out) @ V
        return einops.rearrange(pred, "b (a h)(c w) -> b (a c) h w", a=2,c=2)
    


class SVDNoiseUnet_Concise(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, resolution=64):
        super(SVDNoiseUnet_Concise, self).__init__()
        

from diffusers.models.normalization import AdaGroupNorm

class NPNet(nn.Module):
      def __init__(self, model_id, pretrained_path=' ', device='cuda') -> None:
            super(NPNet, self).__init__()

            assert model_id in ['SD1.5', 'DreamShaper', 'DiT']

            self.model_id = model_id
            self.device = device
            self.pretrained_path = pretrained_path

            (
                  self.unet_svd, 
                  self.unet_embedding, 
                  self.text_embedding, 
                  self._alpha, 
                  self._beta
             ) = self.get_model()
      def save_model(self, save_path: str):
            """
            Save this NPNet so that get_model() can later reload it.
            """
            torch.save({
                  "unet_svd":        self.unet_svd.state_dict(),
                  "unet_embedding":  self.unet_embedding.state_dict(),
                  "embeeding":       self.text_embedding.state_dict(),  # matches get_model’s key
                  "alpha":           self._alpha,
                  "beta":            self._beta,
            }, save_path)
            print(f"NPNet saved to {save_path}")
      def get_model(self):

            unet_embedding = NoiseTransformer(resolution=(128,96)).to(self.device).to(torch.float32)
            unet_svd = SVDNoiseUnet(resolution=(128,96)).to(self.device).to(torch.float32)

            if self.model_id == 'DiT':
                  text_embedding = AdaGroupNorm(768 * 77, 4, 1, eps=1e-6).to(self.device).to(torch.float32)
            else:
                  text_embedding = AdaGroupNorm(768 * 77, 4, 1, eps=1e-6).to(self.device).to(torch.float32) 

            # initialize random _alpha and _beta when no checkpoint is provided
            _alpha = torch.randn(1, device=self.device)
            _beta = torch.randn(1, device=self.device)

            if '.pth' in self.pretrained_path:
                  gloden_unet = torch.load(self.pretrained_path)
                  unet_svd.load_state_dict(gloden_unet["unet_svd"],strict=True)
                  unet_embedding.load_state_dict(gloden_unet["unet_embedding"],strict=True)
                  text_embedding.load_state_dict(gloden_unet["embeeding"],strict=True)
                  _alpha = gloden_unet["alpha"]
                  _beta = gloden_unet["beta"]

                  print("Load Successfully!")

                  return unet_svd, unet_embedding, text_embedding, _alpha, _beta
            
            else:
                  return unet_svd, unet_embedding, text_embedding, _alpha, _beta
            

      def forward(self, initial_noise, prompt_embeds):

            prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
            text_emb = self.text_embedding(initial_noise.float(), prompt_embeds)

            encoder_hidden_states_svd = initial_noise
            encoder_hidden_states_embedding = initial_noise + text_emb

            golden_embedding = self.unet_embedding(encoder_hidden_states_embedding.float())

            golden_noise = self.unet_svd(encoder_hidden_states_svd.float()) + (
                        2 * torch.sigmoid(self._alpha) - 1) * text_emb + self._beta * golden_embedding

            return golden_noise

class NPNet64(nn.Module):
      def __init__(self, model_id, pretrained_path=' ', device='cuda') -> None:
            super(NPNet64, self).__init__()
            self.model_id = model_id
            self.device = device
            self.pretrained_path = pretrained_path

            (
                  self.unet_svd, 
                  self.unet_embedding, 
                  self.text_embedding, 
                  self._alpha, 
                  self._beta
             ) = self.get_model()

      def save_model(self, save_path: str):
            """
            Save this NPNet so that get_model() can later reload it.
            """
            torch.save({
                  "unet_svd":        self.unet_svd.state_dict(),
                  "unet_embedding":  self.unet_embedding.state_dict(),
                  "embeeding":       self.text_embedding.state_dict(),  # matches get_model’s key
                  "alpha":           self._alpha,
                  "beta":            self._beta,
            }, save_path)
            print(f"NPNet saved to {save_path}")
        
      def get_model(self):

            unet_embedding = NoiseTransformer(resolution=(64,64)).to(self.device).to(torch.float32)
            unet_svd = SVDNoiseUnet64(resolution=64).to(self.device).to(torch.float32)
            _alpha = torch.randn(1, device=self.device)
            _beta = torch.randn(1, device=self.device)

            text_embedding = AdaGroupNorm(768 * 77, 4, 1, eps=1e-6).to(self.device).to(torch.float32) 

            
            if '.pth' in self.pretrained_path:
                  gloden_unet = torch.load(self.pretrained_path)
                  unet_svd.load_state_dict(gloden_unet["unet_svd"])
                  unet_embedding.load_state_dict(gloden_unet["unet_embedding"])
                  text_embedding.load_state_dict(gloden_unet["embeeding"])
                  _alpha = gloden_unet["alpha"]
                  _beta = gloden_unet["beta"]

                  print("Load Successfully!")

            return unet_svd, unet_embedding, text_embedding, _alpha, _beta
            

      def forward(self, initial_noise, prompt_embeds):

            prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
            text_emb = self.text_embedding(initial_noise.float(), prompt_embeds)

            encoder_hidden_states_svd = initial_noise
            encoder_hidden_states_embedding = initial_noise + text_emb

            golden_embedding = self.unet_embedding(encoder_hidden_states_embedding.float())

            golden_noise = self.unet_svd(encoder_hidden_states_svd.float()) + (
                        2 * torch.sigmoid(self._alpha) - 1) * text_emb + self._beta * golden_embedding

            return golden_noise

class NPNet128(nn.Module):
      def __init__(self, model_id, pretrained_path=True, device='cuda') -> None:
            super(NPNet128, self).__init__()

            assert model_id in ['SDXL', 'DreamShaper', 'DiT']

            self.model_id = model_id
            self.device = device
            self.pretrained_path = pretrained_path

            (
                  self.unet_svd, 
                  self.unet_embedding, 
                  self.text_embedding, 
                  self._alpha, 
                  self._beta
             ) = self.get_model()

      def get_model(self):

            unet_embedding = NoiseTransformer(resolution=(128,128)).to(self.device).to(torch.float32)
            unet_svd = SVDNoiseUnet128(resolution=128).to(self.device).to(torch.float32)

            if self.model_id == 'DiT':
                  text_embedding = AdaGroupNorm(1024 * 77, 4, 1, eps=1e-6).to(self.device).to(torch.float32)
            else:
                  text_embedding = AdaGroupNorm(2048 * 77, 4, 1, eps=1e-6).to(self.device).to(torch.float32) 

            
            if '.pth' in self.pretrained_path:
                  gloden_unet = torch.load(self.pretrained_path)
                  unet_svd.load_state_dict(gloden_unet["unet_svd"])
                  unet_embedding.load_state_dict(gloden_unet["unet_embedding"])
                  text_embedding.load_state_dict(gloden_unet["embeeding"])
                  _alpha = gloden_unet["alpha"]
                  _beta = gloden_unet["beta"]

                  print("Load Successfully!")

                  return unet_svd, unet_embedding, text_embedding, _alpha, _beta
            
            else:
                  assert ("No Pretrained Weights Found!")
            

      def forward(self, initial_noise, prompt_embeds):

            prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
            text_emb = self.text_embedding(initial_noise.float(), prompt_embeds)

            encoder_hidden_states_svd = initial_noise
            encoder_hidden_states_embedding = initial_noise + text_emb

            golden_embedding = self.unet_embedding(encoder_hidden_states_embedding.float())

            golden_noise = self.unet_svd(encoder_hidden_states_svd.float()) + (
                        2 * torch.sigmoid(self._alpha) - 1) * text_emb + self._beta * golden_embedding

            return golden_noise
      