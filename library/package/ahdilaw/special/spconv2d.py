import torch
import torch.nn as nn
import torch.nn.functional as F

class SPConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 threshold_init=0.1,
                 scale_init=10.0):
        super().__init__()
        assert kernel_size == 3, "Kernel size must be 3x3"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Core 1×1 kernel
        self.core = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        
        # Learnable threshold & scale
        self.threshold = nn.Parameter(torch.tensor([threshold_init]))
        self.scale = nn.Parameter(torch.tensor([scale_init]))
        
        # Shared periphery weights (8 neighbors)
        self.periphery = nn.Parameter(torch.Tensor(8))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.core)
        nn.init.uniform_(self.periphery, -0.1, 0.1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1) extract 3×3 patches with given stride & padding
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )                                      # (B, C*9, L)
        _, _, L = patches.shape
        patches = patches.view(B, C, 9, L)     # (B, C, 9, L)

        # compute output spatial dims
        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        # 2) divergence on center vs periphery
        center = patches[:, :, 4, :].unsqueeze(2)    # (B, C, 1, L)
        diffs = (patches - center) ** 2
        div = diffs.sum(dim=(1,2)).view(B, H_out, W_out)

        # 3) STE mask
        m_soft = torch.sigmoid((div - self.threshold) * self.scale).unsqueeze(1)
        m_hard = (m_soft > 0.5).float()
        mask = m_hard.detach() - m_soft.detach() + m_soft

        # 4) allocate output
        out = torch.zeros(B, self.out_channels, H_out, W_out, device=x.device)

        # 5) core-only where mask==0
        zeros = torch.where(m_hard.view(B, -1) == 0)
        if zeros[0].numel():
            b_idx, pos = zeros
            cen = center.squeeze(2)                   # (B, C, L)
            vals = cen[b_idx, :, pos]                # (N0, C)
            w = self.core.view(self.out_channels, self.in_channels)  # (OC, C)
            out.view(B, self.out_channels, -1)[b_idx, :, pos] = vals @ w.t()

        # 6) detailed where mask==1
        ones = torch.where(m_hard.view(B, -1) == 1)
        if ones[0].numel():
            b_idx, pos = ones
            flat = patches.permute(0,3,1,2).reshape(-1, C, 9)  # (B*L, C, 9)
            sel = flat[b_idx * L + pos]                       # (Nd, C, 9)
            peri = sel[:, :, [0,1,2,3,5,6,7,8]]               # (Nd, C, 8)
            w = peri * self.periphery.view(1,1,8)            # (Nd, C, 8)
            agg = w.sum(dim=2)                               # (Nd, C)
            w1 = self.core.view(self.out_channels, self.in_channels)
            out.view(B, self.out_channels, -1)[b_idx, :, pos] = agg @ w1.t()

        return out