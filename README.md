# PointNet
PyTorch implementation of the PointNet [1], a deep neural network that can directly process point-clouds, without using intermediate representations such as voxels or multi-view images, and is suitable for a variety of point-based 3D recognition tasks such as object classification and part segmentation.

## Introduction

PointNet is a neural network with that can arbitrarily aproximate any uniformly continuous, permutation-invariant (symmetric) function `f` on finite sets of points (point clouds) by decomposing it into:
```
f({x_1, x_n}) ≈ (g ∘ POOL)({h(x_1), ..., h(x_n)})
```
where `x_i` is the i-th point-vector of size `C_in`, `X_in = {x_1, ..., x_n}` is a point-cloud of cardinality `n`, `h: R^{C_in} -> R^{C_out}` and `g: R^{C_out} -> R^L` are continuous functions, `POOL` is a symmetric pooling operation such as max-pooling or avg-pooling that aggregates information from the points and enforces permutation-invariance on the whole function.

The functions `g` and `h` can be approximated by Multilayer-perceptrons (MLPs) which are a combination of fully-connected layers (FCs). Since the MLP for the function `h` acts on all the points of a point-cloud identically and independently, we name it as PointMLP:
```
PointMLP({x_1, ..., x_n}) = {MLP(x_1), ..., MLP(x_n)}.
```
<p align="center">
  <img align="center" src="https://user-images.githubusercontent.com/15230238/169560375-f784ecba-a2d7-4bb9-a70f-6182254b8cc5.svg">
</p>

We note that in order for a PointNet to arbitrarily approximate any continuous set function, it is required to have a PointMLP with sufficiently large number of neurons (C_out) [1].


The vanilla PointNet thus can be represented as:
```
f(X_in) ≈ MLP(MAX(PointMLP(X_in))).
```

<p align="center">
  <img align="center" src="https://user-images.githubusercontent.com/15230238/169559785-45b89b86-e74c-4d0b-85b4-c81d5bbccc33.svg">
</p>

In addition to the permutation-invariance, it is desirable to have invariance to certain geometric transformations of the point-clouds. TNets...

<p align="center">
  <img align="center" src="https://user-images.githubusercontent.com/15230238/169561305-f6f60359-42f8-4edf-92e7-f0fcb7e5b076.svg">
</p>


## Architectures

### PointNet for classification

<p align="center">
  <img align="center" src="https://user-images.githubusercontent.com/15230238/169564557-7b7c9778-071c-49bd-a1dd-2ce9bd43a65e.svg">
</p>

### PointNet for part segmentation

<p align="center">
  <img align="center" src="https://user-images.githubusercontent.com/15230238/169564573-099b8476-be74-4ece-afc0-fa84aa2d2709.svg">
</p>


## Dependencies
- Python 3 (3.8.3)
- Pytorch (1.10.1)
- Numpy (1.18.5)
- Matplotlib (3.4.1)
- Trimesh (3.9.32)
- h5py (3.1.0)

In parentheses are the versions that were used for creating and testing the code.


## Usage
For training the PointNet for 3D object classification on ModelNet dataset:
```
cd applications/classification
python train_clf.py --hdf5_path "dataset.hdf5" --device "cuda" --batch_size_train 32 --num_epochs 100 ...
```
Similarly for evaluation and inference:
```
python eval_clf.py ...
python infere_clf.py ...
```
See [arg_parser.py](applications/classification/arg_parser.py) for all the possible command-line arguments and [run_clf.ipynb](applications/classification/run_clf.ipynb) for an example.


## References
[1] [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, C. Qi et al., 2016](https://arxiv.org/abs/1612.00593)

[2] [Original TensorFlow implementation of the PointNet](https://github.com/charlesq34/pointnet)
