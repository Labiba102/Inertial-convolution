#include <torch/extension.h>

torch::Tensor inertial_conv_forward(
    torch::Tensor input,
    torch::Tensor core,
    torch::Tensor perip,
    torch::Tensor thresh,
    torch::Tensor scale,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &inertial_conv_forward, "InertialConv forward (CUDA)");
}