#include <torch/extension.h>

template <typename scalar_t>
__global__ void inertial_conv_kernel_generic(
    const scalar_t* __restrict__ input,  // B*C*H*W
    const scalar_t* __restrict__ core,   // OC*C*K*K
    const scalar_t* __restrict__ perip,  // D*D - K*K
    const scalar_t* __restrict__ thresh, // OC
    scalar_t scale,
    scalar_t* __restrict__ output,       // B*OC*Hout*Wout
    int B, int C, int H, int W,
    int OC, int D, int K, int stride, int pad
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int Hout = (H + 2*pad - D)/stride + 1;
    int Wout = (W + 2*pad - D)/stride + 1;
    int total = B*OC*Hout*Wout;
    if (idx >= total) return;

    // decode (b,oc,oy,ox)
    int tmp = idx;
    int ox  = tmp % Wout;  tmp /= Wout;
    int oy  = tmp % Hout;  tmp /= Hout;
    int oc  = tmp % OC;    tmp /= OC;
    int b   = tmp;

    const scalar_t* inp    = input + b*C*H*W;
    const scalar_t* core_w = core  + oc*C*K*K;
    const scalar_t* th     = thresh + oc;
    int cen_i = oy*stride - pad + D/2;
    int cen_j = ox*stride - pad + D/2;

    scalar_t sum_div = 0, core_out = 0, detp_out = 0;

    // compute central‐patch convolution (KxK core)
    for (int c=0; c<C; ++c) {
      for (int ki=0; ki<K; ++ki) {
        for (int kj=0; kj<K; ++kj) {
          int yi = cen_i - (K/2) + ki;
          int xj = cen_j - (K/2) + kj;
          scalar_t pix = 0;
          if (yi>=0 && yi<H && xj>=0 && xj<W)
            pix = inp[c*H*W + yi*W + xj];
          core_out += pix * core_w[c*K*K + ki*K + kj];
        }
      }
    }

    // gather periphery divergences & weighted sum
    int per_idx = 0;
    for (int di = 0; di < D; ++di) {
      for (int dj = 0; dj < D; ++dj) {
        // skip core region
        if (di >= (D-K)/2 && di < (D+K)/2 &&
            dj >= (D-K)/2 && dj < (D+K)/2) {
          continue;
        }
        scalar_t perip_val = perip[per_idx];
        per_idx++;

        // Loop over input channels
        for (int c = 0; c < C; ++c) {
          int yi = oy*stride - pad + di;
          int xj = ox*stride - pad + dj;
          scalar_t pix = 0;
          if (yi>=0 && yi<H && xj>=0 && xj<W)
            pix = inp[c*H*W + yi*W + xj];
          scalar_t center = inp[c*H*W + cen_i*W + cen_j];
          scalar_t diff = pix - center;
          sum_div += diff * diff;
          
          // Get central weight for this channel
          scalar_t w0 = core_w[c*K*K + (K/2)*K + (K/2)];
          detp_out += pix * (perip_val * w0);
        }
      }
    }

    // STE mask
    scalar_t d = sum_div;
    scalar_t m = 1.0f / (1.0f + expf(-(d - *th) * scale));
    scalar_t mh = m > 0.5f ? 1.0f : 0.0f;
    scalar_t mask = mh - m + m;

    // write
    output[idx] = core_out * (1-mask) + detp_out * mask;
}

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
) {
    auto B = input.size(0), C = input.size(1),
         H = input.size(2), W = input.size(3);
    auto OC = core.size(0);
    int Hout = (H + 2*padding - D)/stride + 1;
    int Wout = (W + 2*padding - D)/stride + 1;
    auto out = torch::zeros({B,OC,Hout,Wout}, input.options());

    int total = B*OC*Hout*Wout;
    int threads = 256;
    int blocks  = (total + threads - 1)/threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "inertial_conv_generic", ([&](){
        inertial_conv_kernel_generic<scalar_t><<<blocks,threads>>>(
            input.data_ptr<scalar_t>(),
            core.data_ptr<scalar_t>(),
            perip.data_ptr<scalar_t>(),
            thresh.data_ptr<scalar_t>(),
            scale.item<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, C, H, W, OC, D, K, stride, padding
        );
    }));
    cudaDeviceSynchronize();
    return out;
}