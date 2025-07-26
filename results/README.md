In this paper, we evaluated the perfomed over 15 epochs on the MNIST dataset, keeping the architecture similar to the LeNet adaptation PyTorch's official repository would reference. 

[Link to the actual LeNet implementation.](https://github.com/pytorch/examples/blob/main/mnist/main.py)

Model Comparison:

Model           |   Accuracy |          Params |           FLOPs
-----------------------------------------------------------------
LeNet 3x3       |     0.9914 |       1,199,882 |      23,984,896
LeNet 1x1       |     0.9647 |       1,609,226 |       6,475,264
LeNet SP        |     0.9738 |       1,183,166 |       4,764,416


For the other baselines, we anticipate testing in the future to check with the effectiveness of the implementation in regards to varying degree of divergence.

<h1>How to test?</h1>

```bash
pip install ahdilaw
```

```python
from ahdilaw import inertial
... = inertial.special.Conv2d(...)
```

See the package [documentation](https://pypi.org/project/ahdilaw/0.0.1.1/) for more details.