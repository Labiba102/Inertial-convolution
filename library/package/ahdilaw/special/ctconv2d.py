import torch
from torch import nn

class CTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, threshold_init=0.1, scale_init=10.0):
        super().__init__()

        #GPU acceleration check
        if not torch.cuda.is_available():
            raise AssertionError(
                "CTConv2d requires GPU acceleration. "
                "Use SPConv2d for CPU-only systems or check CUDA installation."
            )
        
        # Lazy import to prevent circular dependencies
        try:
            from ..cuda import inertial_conv_ext_v2
        except ImportError as e:
            raise ImportError(
                "CUDA extensions not found. Reinstall with GPU support "
                "or use SPConv2d instead."
            ) from e
        
        assert kernel_size == 3, "CTConv2d only supports 3x3 kernels"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        self.core = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        self.threshold = nn.Parameter(torch.full((out_channels,), threshold_init))
        self.scale = nn.Parameter(torch.tensor([scale_init]))
        self.periphery = nn.Parameter(torch.Tensor(8))
        
        nn.init.kaiming_uniform_(self.core)
        nn.init.uniform_(self.periphery, -0.1, 0.1)

    def forward(self, x):
        #GPU acceleration check
        if not x.is_cuda:
            raise RuntimeError(
                "CTConv2d requires GPU tensors. "
                "Input tensor is on CPU [Solution: move to GPU first.]"
            )
        
        return inertial_conv_ext_v2.forward(
            x, self.core, self.periphery, 
            self.threshold, self.scale, self.stride, self.padding
        )