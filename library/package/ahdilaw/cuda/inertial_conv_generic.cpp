#include <torch/extension.h>

torch::Tensor inertial_conv_generic_forward(
    torch::Tensor input,
    torch::Tensor core,
    torch::Tensor perip,
    torch::Tensor thresh,
    torch::Tensor scale,
    int D,
    int K,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &inertial_conv_generic_forward, "Generic InertialConv (CUDA)");
}