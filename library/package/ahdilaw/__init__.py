from .special.spconv2d import SPConv2d

class _SpecialLayers:
    def __init__(self):
        from .special.spconv2d import SPConv2d
        self.SPConv2d = SPConv2d
        
        # CTConv2d will only be available if CUDA is present
        try:
            from .special.ctconv2d import CTConv2d
            self.CTConv2d = CTConv2d
        except (ImportError, RuntimeError):
            self.CTConv2d = None

class InertialFilters:
    def __init__(self):
        self.special = _SpecialLayers()
        
        # Generic Conv2d will only be available if CUDA is present
        try:
            from .conv2d import Conv2d
            self.Conv2d = Conv2d
        except (ImportError, RuntimeError):
            self.Conv2d = None

inertial = InertialFilters()