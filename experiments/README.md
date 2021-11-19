# Cumulative Spectral Gradient (CSG) metric

**WARNING** I recently moved the repo around, the instructions might not be up-to-date.

This is the code to reproduce the CVPR 2019 paper : Spectral Metric for Dataset Complexity Assessment. 

Paper: https://arxiv.org/pdf/1905.07299.pdf

CSG is a measure which estimates the complexity of a dataset by combining probability product kernel (Jebara et al.) and Graph Theory.

## Module
``` bash
$COMPLEXITY/spectral_metric
├── visualize.py
|   - Visualize the results
├── lib.py
|   - Common function
├── handle_datasets.py
|   - Handle all the datasets loading
├── estimator.py
|   - Sklearn-like version of the CSG
├── embedding
|   - Compute multiple types of embeddings
│   ├── imagenet.py
│   ├── ...
│   ├── cnn_autoencoder.py
│   └── autoencoder.py
├── config.py
|   - Configuration and common functions
└── accuracy
    - Compute the accuracy for all datasets.
    ├── model_definition.py
    └── cnn_train.py
```

## Data and embedding
Download the data
```bash
wget "https://onedrive.live.com/download?cid=AB307638A9FB0EF9&resid=AB307638A9FB0EF9%21368&authkey=AC9YsfqB8u8f-nA" -O dataset.zip
unzip dataset.zip
export DATASET_ROOT=$PWD
```
In `config.py`, modify `TMP_DIR` to your desire.

**NOTE**: CompCars requires a licence so it's not included in the archive.
**NOTE**: MIO-TCD is not included as well due to its size.
          `ln -s $MIO_TCD_CLASSIFICATION $DATASET_ROOT/Datasets/mio_tcd_classification`


**OPTIONAL** If you want to pretrained the embeddings
Run `embedding/script.sh` to get everything (It will take a long time) otherwise, it will be computed as needed.
**NOTE**: If you don't want to train on all datasets, modify `handle_datasets.paper_datasets`.

---

## Requirements

`pip install -r requirements.txt`

* Keras
* Scikit-learn
* joblib
* Seaborn
* pandas
* tqdm
* OpenCV
* MulticoreTSNE (on Github)

`pip install keras tensorflow-gpu scikit-learn joblib pandas tqdm seaborn`

__Install MulticoreTSNE__
```bash
pip install git+https://github.com/DmitryUlyanov/Multicore-TSNE.git
```

## Usage
```bash
usage: main.py [-h] [--embd {embd,cnn_embd,vgg,xception}] [--tsne]
               [--shuffled_class SHUFFLED_CLASS] [--small SMALL]
               [--make_graph] [--k_nearest M_NEAREST] [--M M]
               datasets [datasets ...]

positional arguments:
  datasets              List of datasets

optional arguments:
  -h, --help            show this help message and exit
  --embd                One of {embd,cnn_embd,vgg,xception}
                        Default : None, will use the raw pixels
  --tsne                Whether to use t-sne or not
  --shuffled_class SHUFFLED_CLASS
                        Number of class to shuffle
  --small SMALL         Reduce the number of sample per class
  --make_graph          Show the dependency graph of the first dataset
  --k_nearest K_NEAREST
                        k-nn hyperparameter
  --M M                 M sample per class
```

*Example*
Here's how to run the measure on mnist and cifar10 using CNN t-SNE. This will also show the graph for the first dataset (mnist)
`python $COMPLEXITY/main.py mnist cifar10 --embd cnn_embd --tsne --make_graph`


### Advanced Usage

```python
# Using estimator.CumulativeGradientEstimator
from spectral_metric.estimator import CumulativeGradientEstimator
from spectral_metric import visualize
from experiments import config

dataset_name = "my_dataset"
data, target = ...  # X_train, y_train
# WARNINGS: data may need to be normalized and/or rescaled
k_nearest = 3  # neigborhood size
M_sample = 100  # Number of estimation per class
estimator = CumulativeGradientEstimator(M_sample, k_nearest)
estimator.fit(data, target)

""" You can now access your estimator fields"""
estimator.difference  # Similarity matrix
estimator.L_mat  # The computed Laplacian matrix
estimator.evals, estimator.evecs  # The sorted eigenvalues and eigenvectors

# To plot the graph
visualize.make_graph(estimator.difference, dataset_name)

# Using a test loop
# Create a config run, see doc for more information.
config = config.make_config(dataset_name, embd=None, tsne=False, small=None, shuffled_class=None)

config, estimators = visualize.process_many(args, k_nearest=10, M_sample=200, loop=5)
visualize.plot_with_err((config, estimators))

## With many configurations
configs = ...
visualize.plot_with_err(*[visualize.process_many(args, k_nearest=10, M_sample=200, loop=5) for args in configs])

```

## Background
We used DCoL for the background computation. You just need to convert the datasets to ARFF and feed it.
We also provide the result for each dataset in a json file.
```bash
cd $TMP_DIR
wget "https://onedrive.live.com/download?cid=AB307638A9FB0EF9&resid=AB307638A9FB0EF9%21369&authkey=ACAfmAfVcSnxuoM" -O dcol.json
```
