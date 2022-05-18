from pathlib import Path
import sys

ROOT_DIR = Path.cwd().parent
module_path = Path(ROOT_DIR, "components")
if module_path not in map(Path, sys.path):
    sys.path.append(str(module_path))
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import PointMLP
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
                                 t_size=3,
                                 agg_fn="max"),
                            PointMLP(layer_sizes=[self.in_features, 64, 128, 128],
                                     add_bias=False,
                                     apply_bn=True,
                                     activation_fn="relu",
                                     dropout_probs=None))
        
        self.block1 = Block(TNet(in_features=128,
                                 t_size=128,
                                 agg_fn="max"),
                            PointMLP(layer_sizes=[128, 512, 2048],
                                     add_bias=False,
                                     apply_bn=True,
                                     activation_fn="relu",
                                     dropout_probs=None))
        
        self.agg_fn = nn.AdaptiveMaxPool1d(output_size=1)
        
        # Segmentation part
        self.point_mlp = PointMLP(layer_sizes=[3008, 256, 256, 128, self.num_parts],
                                  add_bias=[False, False, False, True],
                                  apply_bn=[True, True, True, False],
                                  activation_fn=["relu", "relu", "relu", ""],
                                  dropout_probs=None)

        
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
            
        t_0 : torch.Tensor
            Transformation matrix of shape (batch_size, 3, 3) from the TNet of the block0.
        
        t_1 : torch.Tensor
            Transformation matrix of shape (batch_size, 128, 128) from the TNet of the block1.
        """
        
        num_points = x.shape[2]
        
        # Block 0
        t_0, xt_0, (y_0_1, y_0_2, y_0_out) = self.block0(x)
        
        # Block 1
        t_1, xt_1, (y_1_1, y_1_out) = self.block1(y_0_out)
        
        # Global feature vector
        global_feat = self.agg_fn(y_1_out)  # (batch_size, 2048, 1)
        global_feat = global_feat.expand(-1, -1, num_points)  # (batch_size, 2048, num_points)

        # Concatenate local and global information
        x_concat = torch.concat([y_0_1, y_0_2, y_0_out, xt_1, y_1_1, global_feat], dim=1)  # (batch_size, 3008, num_points)        
        
        # Point MLP output
        y_out = self.point_mlp(x_concat)[-1]  # (batch_size, num_parts, num_points)
        
        # Compute log-probabilities
        y = F.log_softmax(y_out, dim=1)  # (batch_size, num_parts, num_points)
        
        return y, t_0, t_1
       

    
if __name__ == "__main__": 
    
    # PointNetPartSeg test
    x = torch.rand(2, 3, 15)
    pointnet_part_seg = PointNetPartSeg(in_features=3, num_parts=50)
    # pointnet_part_seg.eval()
    y, t0, t1 = pointnet_part_seg(x)
    print(y.shape)
    print(t0.shape)
    print(t1.shape)