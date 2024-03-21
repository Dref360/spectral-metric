<p align="center">
    <br>
    <h1 align="center">
      Spectral Metric
    </h1>
    <br>
<p align="center">
  <a href="https://dref360.github.io/spectral-metric">
    Documentation
  </a>
</p>

This library provides an implementation of CSG, from CVPR 2019 paper: [Spectral Metric for Dataset Complexity Assessment](https://arxiv.org/abs/1905.07299).

> [!NOTE]  
> CSG is a measure that estimates the complexity of a dataset by combining probability product kernel (Jebara et al.) and Graph Theory. By doing so, one can estimate the complexity of their dataset without training a model.

For the experiment part of the repo, please see [./experiments/README.md](./experiments/README.md)

**Spectral metric in action**:

1. [🤗 HuggingFace Space](https://huggingface.co/spaces/Dref360/spectral-metric)
2. [In-depth analysis of CLINC-150](https://github.com/Dref360/spectral-metric/blob/master/notebooks/clinc_oos.ipynb)

**Installation**

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

<p align="center">
<img src="./images/example.png" width="50%">
</p>

# Results

We can compare multiple datasets without training any classifier.
For example, we can plot the eigenvalues of the datasets, the
higher the values are, the harder the dataset is.

![](./images/evals.png)

**Note:** The actual CSG is based on the gradient of the eigenvalues,
this is done to overcome issues where the first classes are easy to separate, but not the last ones.

Please refer to the paper for more details.

## Support

For support, please submit an issue!


# Contributing

We are open to contributions, please submit an issue or a pull request.

To get yourself a running environment you will need [Poetry](https://python-poetry.org/), our package manager.

```bash
# Install the package and the development dependencies
poetry install 

# Format the code
make format

# Test with flake8, mypy and pytest
make test
```
