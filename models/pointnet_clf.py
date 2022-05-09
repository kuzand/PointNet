from pathlib import Path
import sys

ROOT_DIR = Path.cwd().parent
module_path = Path(ROOT_DIR, "components")
if module_path not in map(Path, sys.path):
    sys.path.append(str(module_path))
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP1d, MLP2d
from tnet import TNet
from block import Block



class PointNetClf(nn.Module):
    """
    Original PointNet for 3D object classification.
    """
    def __init__(self, in_features=3, num_classes=40):
        
        if not isinstance(num_classes, int) or num_classes < 1:
            raise TypeError("The num_classes must be a positive integer.")
        
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        
        # Backbone
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
        
        # Classification mlp
        self.mlp = MLP1d(layer_sizes=[1024, 512, 256, num_classes],
                         add_bias=[False, False, True],
                         apply_bn=[True, True, False],
                         activation_fn=["relu", "relu", ""],
                         dropout_probs=[0.3, 0.3, 0.0])
        
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, 3, num_points).
        
        Returns
        -------
        y : torch.Tensor
            Output tensor of shape (batch_size, num_classes).
            Note that these are log-probabilities for each class.
            
        transform_mat_0 : torch.Tensor
            Transformation matrix of shape (batch_size, 3, 3) from the TNet of the block0.
        
        transform_mat_1 : torch.Tensor
            Transformation matrix of shape (batch_size, 64, 64) from the TNet of the block1.
        """
        
        y_0, _, transform_mat_0 = self.block0(x)    # (batch_size, 64, num_points), (batch_size, 3, num_points), (batch_size, 3, 3)
        y_1, _, transform_mat_1 = self.block1(y_0)  # (batch_size, 1024, num_points), (batch_size, 64, num_points), (batch_size, 64, 64)
        
        global_feat = self.agg_fn(y_1).squeeze(dim=-1)  # (batch_size, 1024)
        
        y = self.mlp(global_feat)  # (batch_size, num_classes)
        y = F.log_softmax(y, dim=-1)  # (batch_size, num_classes)
        
        return y, transform_mat_0, transform_mat_1
       

    
if __name__ == "__main__": 
    
    # PointNetClf test
    x = torch.rand(2, 3, 10)
    pointnet_clf = PointNetClf(in_features=3, num_classes=40)
    pointnet_clf.eval()
    y, t0, t1 = pointnet_clf(x)
    print(y.shape)
    print(t0.shape)
    print(t1.shape)
