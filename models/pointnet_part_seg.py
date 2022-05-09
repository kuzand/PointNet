from pathlib import Path
import sys

ROOT_DIR = Path.cwd().parent
module_path = Path(ROOT_DIR, "components")
if module_path not in map(Path, sys.path):
    sys.path.append(str(module_path))
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP2d
from tnet import TNet
from block import Block



class PointNetPartSeg(nn.Module):
    """
    Original PointNet for 3D object part segmentation.
    """
    def __init__(self, in_features=3, num_parts=50):
        
        if not isinstance(num_parts, int) or num_parts < 1:
            raise TypeError("The num_parts must be a positive integer.")
        
        super().__init__()
        self.in_features = in_features
        self.num_parts = num_parts
        
        # Classification part
        self.block0 = Block(TNet(in_features=self.in_features,
                                 mat_size=3,
                                 agg_fn="max"),
                            MLP2d(layer_sizes=[self.in_features, 64, 64],
                                  add_bias=False,
                                  apply_bn=True,
                                  activation_fn="relu",
                                  dropout_probs=None))
        
        self.block1 = Block(TNet(in_features=64,
                                 mat_size=64,
                                 agg_fn="max"),
                            MLP2d(layer_sizes=[64, 64, 128, 1024],
                                  add_bias=False,
                                  apply_bn=True,
                                  activation_fn="relu",
                                  dropout_probs=None))
        
        self.agg_fn = nn.AdaptiveMaxPool1d(output_size=1)
        
        # Segmentation part
        self.block2 = Block(None,
                            MLP2d(layer_sizes=[1088, 512, 256, 128],
                                  add_bias=False,
                                  apply_bn=True,
                                  activation_fn="relu",
                                  dropout_probs=None))

        self.block3 = Block(None,
                            MLP2d(layer_sizes=[128, 128, num_parts],
                                  add_bias=False,
                                  apply_bn=True,
                                  activation_fn="relu",
                                  dropout_probs=None))
        
        
    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, num_points).

        Returns
        -------
        y : torch.Tensor
            Output tensor of shape (batch_size, num_parts, num_points).
            Note that these are log-probabilities for each part per point.
            
        transform_mat_0 : torch.Tensor
            Transformation matrix of shape (batch_size, 3, 3) from the TNet of the block0.
        
        transform_mat_1 : torch.Tensor
            Transformation matrix of shape (batch_size, 64, 64) from the TNet of the block1.
        """
        
        num_points = x.shape[2]
        
        y_0, x_transformed_0, transform_mat_0 = self.block0(x)  # (batch_size, 64, num_points), (batch_size, 3, num_points), (batch_size, 3, 3)
        y_1, x_transformed_1, transform_mat_1 = self.block1(y_0)  # (batch_size, 1024, num_points), (batch_size, 64, num_points), (batch_size, 64, 64)
        
        global_feat = self.agg_fn(y_1)  # (batch_size, 1024, 1)
        
        x_concat = torch.concat([x_transformed_1, global_feat.expand(-1, -1, num_points)], dim=1)  # (batch_size, 1088, num_points)
        
        y_2, _, _ = self.block2(x_concat)  # (batch_size, 128, num_points), None, None
        y_3, _, _ = self.block3(y_2)  # (batch_size, num_parts, num_points), None, None
        
        y = F.log_softmax(y_3, dim=1)  # (batch_size, num_parts, num_points)
        
        return y, transform_mat_0, transform_mat_1
       

    
if __name__ == "__main__": 
    
    # PointNetPartSeg test
    x = torch.rand(2, 3, 15)
    pointnet_part_seg = PointNetPartSeg(in_features=3, num_parts=40)
    # pointnet_part_seg.eval()
    y, t0, t1 = pointnet_part_seg(x)
    print(y.shape)
    print(t0.shape)
    print(t1.shape)