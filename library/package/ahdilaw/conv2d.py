import torch
from torch import nn

class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, D=3, K=1, stride=1, padding=None, threshold_init=0.1, scale_init=10.0):
        super().__init__()

        #GPU acceleration check
        if not torch.cuda.is_available():
            raise AssertionError(
                "Generic Conv2d requires GPU acceleration. "
                "Use SPConv2d for CPU-only systems."
            )
        
        #Lazy import to prevent circular dependencies
        try:
            from .cuda import inertial_conv_ext_generic
        except ImportError as e:
            raise ImportError(
                "CUDA extensions not found. Reinstall with GPU support "
                "or use SPConv2d instead."
            ) from e
        
        assert D % 2 == 1 and K % 2 == 1 and K <= D
        self.stride = stride
        self.padding = padding if padding is not None else (D-1)//2
        self.D = D
        self.K = K
        
        self.core = nn.Parameter(torch.Tensor(out_ch, in_ch, K, K))
        self.periphery = nn.Parameter(torch.Tensor(D*D - K*K))
        self.thresh = nn.Parameter(torch.ones(out_ch) * threshold_init)
        self.scale = nn.Parameter(torch.tensor([scale_init]))
        
        nn.init.kaiming_uniform_(self.core)
        nn.init.uniform_(self.periphery, -0.1, 0.1)

    def forward(self, x):
        #GPU acceleration check
        if not x.is_cuda:
            raise RuntimeError(
                "Generic Conv2d requires GPU tensors. "
                "Input tensor is on CPU [Solution: move to GPU first.]"
            )
        
        return inertial_conv_ext_generic.forward(
            x, self.core, self.periphery, self.thresh, 
            self.scale, self.D, self.K, self.stride, self.padding
        )