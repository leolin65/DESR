import torch
from torch import nn


import torch
from torch import nn

# 
# ================================================================
# FE-CLIP high-pass version of Adapter
#   👉 高通分支在「投影後空間」做，避免維度不合問題
#   👉 hp_alpha 改成 learnable parameter（類似 0.3 但可學）
# ================================================================


class MultiScaleFusion(nn.Module):
    def __init__(self, num_scales: int):
        super().__init__()
        # 可學的 multi-scale 權重
        self.weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, H, W)
        return: (B, 1, H, W)
        """
        assert x.dim() == 4 and x.shape[1] == self.weights.numel(), \
            f"MultiScaleFusion expects (B,{self.weights.numel()},H,W), got {x.shape}"

        w = torch.softmax(self.weights, dim=0).view(1, -1, 1, 1)  # (1,S,1,1)
        return (x * w).sum(dim=1, keepdim=True)


# ================================================================
# FE-CLIP high-pass version of Adapter
#   👉 高通分支在「投影後空間」做，避免維度不合問題
#   👉 hp_alpha 改成 learnable parameter（類似 0.3 但可學）
# ================================================================

class SimpleAdapter(nn.Module):
    """
    FE-CLIP style SimpleAdapter:
      - Linear → LeakyReLU 做投影
      - 在投影後特徵上做 1D high-pass：y_hp = y - mean(y)
      - 回傳 y + hp_alpha * y_hp，其中 hp_alpha 可學

    輸入形狀：
      - 主要情境：x: (L, B, C)  來自 AdaptedCLIP 的 transformer blocks
      - 也支援 (B, N, C) 或 (N, C)
    """
    def __init__(self, c_in, c_out=768, init_hp_alpha: float = 0.2):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_out, bias=False),
            nn.LeakyReLU()
        )
        # ⭐ 可學 gate，初始化為原本的 0.2/0.3 類似大小
        self.hp_alpha = nn.Parameter(torch.tensor(init_hp_alpha, dtype=torch.float32))
        print("[SimpleAdapter] init hp_alpha =", float(self.hp_alpha.data))

    def _high_pass_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        1D high-pass: x_i - mean(x)

        - 如果 x: (B, N, C) → 在 N 維度做 high-pass（token 維）
        - 如果 x: (N, C)    → 在 N 維度做 high-pass
        """
        if x.dim() == 2:
            # (N, C)
            mean = x.mean(dim=0, keepdim=True)
            return x - mean
        elif x.dim() == 3:
            # (B, N, C)
            mean = x.mean(dim=1, keepdim=True)
            return x - mean
        else:
            return torch.zeros_like(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        主要情況：x 來自 CLIP transformer block:
          - x: (L, B, C)

        也允許其他形狀：
          - (B, N, C)
          - (N, C)
        """
        if x.dim() == 3 and x.shape[0] != x.shape[1]:
            # 假設是 (L, B, C) → 轉成 (B, L, C) 當作 (B, N, C) 處理
            # 這是 AdaptedCLIP / encode_text 的主要情況
            x_bn = x.permute(1, 0, 2)          # (B, L, C)
            y_bn = self.fc(x_bn)               # (B, L, c_out)
            hp_bn = self._high_pass_1d(y_bn)   # (B, L, c_out)
            out_bn = y_bn + self.hp_alpha * hp_bn
            out = out_bn.permute(1, 0, 2)      # (L, B, c_out)
            return out

        # 其他情況直接投影 + high-pass
        y = self.fc(x)
        hp = self._high_pass_1d(y)
        return y + self.hp_alpha * hp


# ---------------------------------------------------------------

class SimpleProj(nn.Module):
    """
    Projection-only 版本：
      - 維持原本 SimpleProj 介面 (Linear [+ReLU])
      - 一樣在投影後特徵上做 high-pass
      - hp_alpha 改成可學 gate
    """
    def __init__(self, c_in, c_out=768, relu=True, init_hp_alpha: float = 0.15):
        super(SimpleProj, self).__init__()
        if relu:
            self.core = nn.Sequential(
                nn.Linear(c_in, c_out, bias=False),
                nn.LeakyReLU()
            )
        else:
            self.core = nn.Linear(c_in, c_out, bias=False)

        self.hp_alpha = nn.Parameter(torch.tensor(init_hp_alpha, dtype=torch.float32))
        print("[SimpleProj ] init hp_alpha =", float(self.hp_alpha.data))

    def _high_pass_1d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # (N, C)
            mean = x.mean(dim=0, keepdim=True)
            return x - mean
        elif x.dim() == 3:
            # (B, N, C)
            mean = x.mean(dim=1, keepdim=True)
            return x - mean
        else:
            return torch.zeros_like(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        支援：
          - (B, N, C) 或 (N, C)
        """
        y = self.core(x)              # (..., c_out)
        hp = self._high_pass_1d(y)    # same shape as y
        return y + self.hp_alpha * hp

# class MultiScaleFusion(nn.Module):
#     def __init__(self, num_scales: int):
#         super().__init__()
#         self.weights = nn.Parameter(torch.ones(num_scales))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, S, H, W)
#         return: (B, 1, H, W)
#         """
#         assert x.dim() == 4 and x.shape[1] == self.weights.numel(), \
#             f"MultiScaleFusion expects (B,{self.weights.numel()},H,W), got {x.shape}"

#         w = torch.softmax(self.weights, dim=0).view(1, -1, 1, 1)
#         return (x * w).sum(dim=1, keepdim=True)


# class MultiScaleFusion(nn.Module):
#     """
#     Learnable multi-scale fusion for AA-CLIP patch predictions.

#     x: (B, S, H, W), S = num_scales (例如 4 個 block)
#     回傳: (B, H, W)
#     """
#     def __init__(self, num_scales: int):
#         super().__init__()
#         # 初始化成全 1，等價於 sum；用 softmax 讓權重穩定、可解釋
#         self.weights = nn.Parameter(torch.ones(num_scales, dtype=torch.float32))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, S, H, W)
#         B, S, H, W = x.shape
#         assert S == self.weights.numel(), \
#             f"MultiScaleFusion: expect {self.weights.numel()} scales, got {S}"

#         # softmax → non-negative, sum=1
#         w = torch.softmax(self.weights, dim=0).view(1, S, 1, 1)  # (1,S,1,1)
#         y = (x * w).sum(dim=1)  # (B,H,W)
#         return y




# # ================================================================
# # FE-CLIP high-pass version of Adapter
# #   👉 高通分支改在「投影後空間」做，避免 1024 vs 768 維度不合
# # ================================================================

# class SimpleAdapter(nn.Module):
#     """
#     FE-CLIP style SimpleAdapter:
#       - 保留原本 Linear → LeakyReLU
#       - 在投影後的特徵上做 1D high-pass：y_hp = y - mean(y)
#       - 回傳 y + hp_alpha * y_hp
#     """
#     def __init__(self, c_in, c_out=768, hp_alpha=0.2):
#         super(SimpleAdapter, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(c_in, c_out, bias=False),
#             nn.LeakyReLU()
#         )
#         self.hp_alpha = hp_alpha
#         print("SimpleAdapter self.hp_alpha = ", self.hp_alpha)

#     def _high_pass_1d(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         1D high-pass: x_i - mean(x)
#         支援 (N,C) or (B,N,C)
#         """
#         if x.dim() == 2:
#             mean = x.mean(dim=0, keepdim=True)
#             return x - mean
#         elif x.dim() == 3:
#             mean = x.mean(dim=1, keepdim=True)
#             return x - mean
#         else:
#             return torch.zeros_like(x)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 先做投影：c_in → c_out
#         y = self.fc(x)           # (..., c_out)

#         # 在「投影後空間」做高通
#         hp = self._high_pass_1d(y)   # same shape as y

#         # FE 合成：不再有 768 vs 1024 mismatch
#         return y + self.hp_alpha * hp


# # ---------------------------------------------------------------

# class SimpleProj(nn.Module):
#     """
#     Projection-only 版本：
#       - 維持原本 SimpleProj 介面
#       - 一樣在投影後特徵上做 high-pass
#     """
#     def __init__(self, c_in, c_out=768, relu=True, hp_alpha=0.15):
#         super(SimpleProj, self).__init__()
#         if relu:
#             self.core = nn.Sequential(
#                 nn.Linear(c_in, c_out, bias=False),
#                 nn.LeakyReLU()
#             )
#         else:
#             self.core = nn.Linear(c_in, c_out, bias=False)

#         self.hp_alpha = hp_alpha
#         print("SimpleProj  self.hp_alpha = ",  self.hp_alpha)

#     def _high_pass_1d(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() == 2:
#             mean = x.mean(dim=0, keepdim=True)
#             return x - mean
#         elif x.dim() == 3:
#             mean = x.mean(dim=1, keepdim=True)
#             return x - mean
#         else:
#             return torch.zeros_like(x)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 先把 1024 → 768
#         y = self.core(x)          # (..., c_out)

#         # 對「投影後」特徵做高通
#         hp = self._high_pass_1d(y)    # same shape as y

#         return y + self.hp_alpha * hp


# from torch import nn
# import ipdb
# import math


# class SimpleAdapter(nn.Module):
#     def __init__(self, c_in, c_out=768):
#         super(SimpleAdapter, self).__init__()
#         self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())

#     def forward(self, x):
#         x = self.fc(x)
#         return x


# class SimpleProj(nn.Module):
#     def __init__(self, c_in, c_out=768, relu=True):
#         super(SimpleProj, self).__init__()
#         if relu:
#             self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())
#         else:
#             self.fc = nn.Linear(c_in, c_out, bias=False)

#     def forward(self, x):
#         x = self.fc(x)
#         return x