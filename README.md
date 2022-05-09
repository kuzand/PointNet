# PointNet
PyTorch implementation of the PointNet [1], a deep neural network that can directly handle unstructured raw point-clouds, without dealing with intermediate representations such as voxels or multi-view images, and is suitable for a variety of point-based 3D recognition tasks such as object classification and part segmentation.


## Dependencies
- Python 3 (3.8.3)
- Pytorch (1.10.1)
- Numpy (1.18.5)
- Matplotlib (3.4.1)
- Trimesh (3.9.32)
- h5py (3.1.0)

In parentheses are the versions that were used for creating and testing the code.


## Usage
For training the PointNet for 3D object classification on ModelNet dataset (run within the `applications/classification`):
```
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
