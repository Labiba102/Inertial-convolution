from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch

# Check if CUDA is available
ext_modules = []
if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension(
            name='ahdilaw.inertial_conv_ext_v2',
            sources=[
                'ahdilaw/cuda/inertial_conv.cpp',
                'ahdilaw/cuda/inertial_conv_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['--use_fast_math']}
        )
    )
    ext_modules.append(
        CUDAExtension(
            name='ahdilaw.inertial_conv_ext_generic',
            sources=[
                'ahdilaw/cuda/inertial_conv_generic.cpp',
                'ahdilaw/cuda/inertial_conv_generic_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['--use_fast_math']}
        )
    )

setup(
    name="ahdilaw",
    version="0.0.1.1",
    author="@ahdilaw",
    description="Plugin Library for Inertial Filters for Deep Convolutional Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["torch","pybind11"],
    python_requires=">=3.7",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
