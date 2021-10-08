# NeurIPS 2021 Challenge

Notebooks and analysis for the Open Problems in Single-Cell Analysis NeurIPS 2021 comptition. More information at [here](https://openproblems.bio/neurips).

## Download the Data

The current forms of the data are public available on S3. To download the data, first install the AWS CLI on your computer: https://aws.amazon.com/cli/

You can download the data to your local computer with the following command (note the dataset size is roughly 1.2 GiB):

```bash
aws s3 sync s3://openproblems-bio/public/explore  /tmp/public/ --no-sign-request
```
You’ll find the following files:
```
explore
├── LICENSE.txt
├── README.txt
├── cite/cite_adt_processed_training.h5ad
├── cite/cite_gex_processed_training.h5ad
├── multiome/multiome_atac_processed_training.h5ad
└── multiome/multiome_gex_processed_training.h5ad
```
These are all [AnnData](https://anndata.readthedocs.io/en/latest/) h5ad files, as described in the following section.

## Data file format

The training data is accessible in an [AnnData](https://anndata.readthedocs.io/en/latest/) h5ad file. More information can be found on AnnData objects [here](https://openproblems.bio/neurips_docs/submission/quickstart/). You can load these files is to use the ```AnnData.read_h5ad()``` function. The easiest way to get started is to [spin up a free Jupyter Server on Saturn Cloud](https://openproblems.bio/neurips_docs/about/explore/).

```python
!pip install anndata
import anndata as ad

adata_gex = ad.read_h5ad("cite/cite_gex_processed_training.h5ad")
adata_adt = ad.read_h5ad("cite/cite_adt_processed_training.h5ad")
```
You can find code examples for exploring the data in our data [exploration notebooks](notebooks/).

## Overview

### [Task 1: Modality Prediction](https://openproblems.bio/neurips_docs/about_tasks/task1_modality_prediction/) | [Leaderboard](https://eval.ai/web/challenges/challenge-page/1111/leaderboard/2856)

<p align="center"><img width="80%" src="https://github.com/asarigun/NeurIPS2021-Challenge/blob/main/docs/wgan.png"></p>

* ```wgan.py```: Model Architecture for Modality Prediction task. This model adapted from WGAN approaches to convert one type of datastructure to another one by **Adversarial Training**. In the figure, as an example ```GEX``` to ```ATAC``` have been illusturated but the other type of conversions can be seen below:

* <table><thead><tr><th><code>mod1</code></th><th><code>mod2</code></th></tr></thead><tbody><tr><td><code>"GEX"</code></td><td><code>"ATAC"</code></td></tr><tr><td><code>"ATAC"</code></td><td><code>"GEX"</code></td></tr><tr><td><code>"GEX"</code></td><td><code>"ADT"</code></td></tr><tr><td><code>"ADT"</code></td><td><code>"GEX"</code></td></tr></tbody></table>



## License

[MIT](LICENSE)

## Acknowledgement

Above explanations from the official site which  can be looked detail [here](https://openproblems.bio/neurips_docs/data/dataset/)!




