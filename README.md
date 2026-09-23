# Inertial Convolution

**Official repository for *Inertial Filters for Deep Convolutional Networks*** — submitted by Labiba Shahab and Ahmed Wali for the Deep Learning coursework at the Lahore University of Management Sciences (LUMS), Spring 2025.

## Abstract

This project introduces **Inertial Convolution**, a novel convolutional layer architecture that improves computational and parameter efficiency in CNNs. Instead of applying fixed-size kernels across all spatial regions, Inertial Convolution uses a dynamic gating mechanism that decides where full computation is necessary and skips redundant operations in simpler regions.

The concept is inspired by physical inertia — just as objects in motion continue unless acted upon, the network "coasts" through low-divergence regions with minimal computation. Tested on the MNIST dataset, the approach achieves substantial reductions in FLOPs and parameter count while maintaining competitive accuracy.

## Motivation

Convolutional Neural Networks (CNNs) are powerful but often computationally expensive. Deploying them on resource-constrained devices such as mobile or embedded systems requires reducing both parameter count and memory footprint.

Inertial Convolution addresses this challenge by introducing a friction-aware, divergence-sensitive mechanism that adapts computation to the visual complexity of input regions — performing more work where it matters most, and less where it doesn't.

## Repository Structure

```
.
├── baselines/          # Standard CNN architectures used for comparison
├── discussions/        # Notes and write-ups accompanying the experiments
├── library/package/    # The packaged InertialConv2D layer, installable as a Python module
├── models/              # Model definitions (LeNet, FashionNet, ResNet-18/VGG variants, InertialConv2D)
├── results/             # Benchmark outputs and evaluation artifacts
├── LICENSE               # MIT License
└── README.md
```

## Methodology

### 1. Baseline Models

To evaluate the performance of the proposed Inertial Convolution layer, we first trained standard CNN architectures on three benchmark datasets — MNIST, CIFAR-10, and Fashion-MNIST — to establish reference accuracies for comparison.

**MNIST — LeNet.** Two LeNet variants were trained on MNIST for 15 epochs, following the standard architecture from PyTorch's official examples. The 3×3-kernel version achieved **99.17%** accuracy, while the simplified 1×1-kernel variant reached **95.74%**, confirming that reducing kernel size slightly lowers performance but significantly reduces model complexity.

**CIFAR-10 — ResNet-18 / VGG-style CNN.** Both architectures are widely used for small-scale RGB datasets. The resulting accuracy was **92.48%**, in line with the expected 92–96% range for these architectures.

**Fashion-MNIST — FashionNet.** A custom CNN architecture, FashionNet, was tested with three kernel sizes: 5×5 achieved **83.38%**, 3×3 achieved **83.17%**, and 1×1 achieved **81.69%** — showing that smaller kernels can maintain competitive accuracy while offering computational benefits.

### 2. The InertialConv2D Layer

A custom PyTorch layer, `InertialConv2D`, was designed to make convolution computation dynamic and adaptive.

**Core components:**
- **Divergence Estimation** — a learned kernel measures local feature variance.
- **Gating Mechanism** — a soft or hard gate produces a spatial mask determining whether to use a 3×3 or a cheaper 1×1 convolution.
- **Conditional Convolution** — combines the outputs of both paths based on the gating values.

**Training strategy:**
- Trained using Absolute Difference Loss and Jensen-Shannon Divergence (JSD) to guide the gating function.
- Best results were achieved with **soft sigmoid gating**:
  - Accuracy: **96.48%**
  - FLOPs reduction: **≈42%**
  - Parameters reduced by **16K**

### 3. Architectural Optimizations

Several iterative modifications (Mod 1–9) were developed on top of the base `InertialConv2D` layer to improve the accuracy/efficiency trade-off. The two standout variants were:
- **Mod 8 & Mod 9** — highest efficiency, using custom CUDA kernels
- **Mod 6** — best balance between accuracy and speed

*(See the `discussions/` and `results/` folders for the full breakdown of each modification.)*

## Results

Benchmarked on MNIST against the LeNet (3×3) baseline:

| Metric | Result |
|---|---|
| FLOPs reduction | ≈42% |
| Model size | 6.14 MB → 4.52 MB |
| Accuracy drop vs. baseline | < 1% |

**Suitability:** Inertial Convolution is well-suited to edge and mobile deployment, balancing computation, memory, and accuracy.

## Tools & Technologies

- **PyTorch** — model design and training
- **CUDA** — custom kernel implementation
- **NumPy & Matplotlib** — analysis and visualization
- **GitHub** — open-source distribution

## Installation

```bash
git clone https://github.com/Labiba102/Inertial-convolution.git
cd Inertial-convolution/library/package
pip install .
```

## Authors

Developed by **Labiba Shahab** and **Ahmed Wali** for the Deep Learning coursework at LUMS (Spring 2025).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
