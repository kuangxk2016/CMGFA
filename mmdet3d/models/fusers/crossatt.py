from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["CrossAttention"]

@FUSERS.register_module()
class CrossAttention(nn.Module):
    def __init__(self,in_dim):
        super(CrossAttention, self).__init__()
        self.channel_in = in_dim
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1,x2) -> torch.Tensor:
        """
        Args:
            x1: input1 B,C1,H,W
            x2: input2 B,C2,H,W
            C1 should equal C2
        Returns:
            output: B,C,H,W

        """
        B, C, H, W = x1.size()
        _, C2, _, _ = x1.size()
        assert C == C2, "inputs channels(C1、C2) should be the same"
        # b,c,h,w->b,c,hw
        CA_q = x2.view(B,C,-1)
        # b,c,h,w->b,c,hw->b,hw,c
        CA_k = x1.view(B,C,-1).permute(0,2,1)
        #(b, c, hw)*(b, hw, c)->(b, c, c)
        energy = torch.bmm(CA_q, CA_k)
        #(b, c, c)
        energy_temp = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        #(b, c, c)
        attention_score = self.softmax(energy_temp)
        #b,c,h,w->b,c,hw
        CA_v = x1.view(B,C,-1)
        # (b,c,c)*(b,c,hw)->(b,c,hw)
        output = torch.bmm(attention_score, CA_v)
        # b,c,hw->b,c,h,w
        output = output.view(B,C,H,W)
        output = self.beta * output

        return output