# Get Started


## Installation

Our package is available on Pypi:

`pip install spectral-metric`

## How to use

This library works with two arrays, the features and the labels. The features are ideally normalized and have
low-dimensionality. In the paper, we use t-SNE to reduce the dimensionality.

```python
from spectral_metric.estimator import CumulativeGradientEstimator
from spectral_metric.visualize import make_graph

X, y = ...  # Your dataset with shape [N, ?], [N]
estimator = CumulativeGradientEstimator(M_sample=250, k_nearest=5)
estimator.fit(data=X, target=y)
csg = estimator.csg  # The actual complexity values.
estimator.evals, estimator.evecs  # The eigenvalues and vectors.

# You can plot the dataset with:
make_graph(estimator.difference, title="Your dataset", classes=["A", "B", "C"])
```

Below we can see the results on MNIST, CIFAR10 and MIO-TCD

![](./images/example.png)