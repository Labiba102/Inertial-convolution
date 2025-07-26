#include <torch/extension.h>

// kernel: for each (b, oc, y, x) compute inertial conv
template <typename scalar_t>
__global__ void inertial_conv_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ core,
    const scalar_t* __restrict__ perip,
    const scalar_t* __restrict__ thresh,
    scalar_t scale,
    scalar_t* __restrict__ output,
    int B, int C, int H, int W,
    int OC, int stride, int pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Hout = (H + 2*pad - 3) / stride + 1;
    int Wout = (W + 2*pad - 3) / stride + 1;
    int total = B * OC * Hout * Wout;
    if (idx >= total) return;

    // decode idx -> b, oc, oy, ox
    int tmp = idx;
    int ox = tmp % Wout; tmp /= Wout;
    int oy = tmp % Hout; tmp /= Hout;
    int oc = tmp % OC;   tmp /= OC;
    int b  = tmp;

    int in_row = oy*stride - pad;
    int in_col = ox*stride - pad;
    const scalar_t* inp_base = input + b*C*H*W;
    scalar_t sum_div = 0, core_out = 0, detp_out = 0;

    // loop over channels
    for (int c = 0; c < C; ++c) {
        // center pixel
        int cy = in_row + 1, cx = in_col + 1;
        scalar_t center = 0;
        if (cy >= 0 && cy < H && cx >= 0 && cx < W)
            center = inp_base[c*H*W + cy*W + cx];

        // core conv (1×1)
        scalar_t w_core = core[oc*C + c];
        core_out += center * w_core;

        // 8 neighbors
        static const int offs[8][2] = {
            {-1,-1},{-1,0},{-1,1},
            { 0,-1},       { 0,1},
            { 1,-1},{ 1,0},{ 1,1}
        };
        for (int i = 0; i < 8; ++i) {
            int ny = cy + offs[i][0];
            int nx = cx + offs[i][1];
            scalar_t pix = 0;
            if (ny >= 0 && ny < H && nx >= 0 && nx < W)
                pix = inp_base[c*H*W + ny*W + nx];
            scalar_t diff = pix - center;
            sum_div += diff*diff;
            detp_out += pix * (perip[i] * w_core);
        }
    }

    // mask (STE)
    scalar_t d = sum_div;
    scalar_t m = 1.0f / (1.0f + expf(-(d - thresh[oc]) * scale));
    scalar_t mh = m > 0.5f ? 1.0f : 0.0f;
    scalar_t mask = mh - m + m;

    // write
    output[idx] = core_out * (1 - mask) + detp_out * mask;
}

torch::Tensor inertial_conv_forward(
    torch::Tensor input,
    torch::Tensor core,
    torch::Tensor perip,
    torch::Tensor thresh,
    torch::Tensor scale,
    int stride,
    int padding
) {
    auto B = input.size(0), C = input.size(1),
         H = input.size(2), W = input.size(3);
    auto OC = core.size(0);
    int Hout = (H + 2*padding - 3) / stride + 1;
    int Wout = (W + 2*padding - 3) / stride + 1;
    auto out = torch::zeros({B, OC, Hout, Wout}, input.options());

    int total = B * OC * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "inertial_conv_cuda", ([&](){
        inertial_conv_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            core.data_ptr<scalar_t>(),
            perip.data_ptr<scalar_t>(),
            thresh.data_ptr<scalar_t>(),
            scale.item<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, C, H, W, OC, stride, padding
        );
    }));
    cudaDeviceSynchronize();
    return out;
}