import torch
import torch.nn as nn
from mlp import MLP2d
from tnet import _TNetBase, TNet


class Block(nn.Module):
    """
    Basic PointNet block: TNet + pointMLP
    """
    
    def __init__(self, tnet, point_mlp):
        
        # Checks
        if tnet is not None and not isinstance(tnet, _TNetBase):
            raise TypeError("The tnet can be None or an instance of the _TNetBase class.")
        
        if not isinstance(point_mlp, MLP2d):
            raise TypeError("The point_mlp should be an instance of the MLP2d class.")
        
        # Initialization
        super().__init__()
        self.tnet = tnet
        self.point_mlp = point_mlp
        
        
    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor 
            Input tensor of shape (batch_size, in_features, num_points).

        Returns
        -------
        y : torch.Tensor
            Output tensor of shape (batch_size, point_mlp.out_features, num_points).
            
        x_transformed : torch.Tensor or None
            Transformed input x of the same shape as the input's shape, (batch_size, in_features, num_points), or None.
            
        transform_mat : torch.Tensor or None
            Transformation matrix of (batch_size, tnet.mat_size, tnet.mat_size) from the TNet, or None.
        """
        
        x_transformed = None
        transform_mat = None
        if self.tnet:
            x_transformed, transform_mat = self.tnet(x)  # ()
            y = self.point_mlp(x_transformed)  # (batch_size, point_mlp.out_features, num_points)
        else:
            y = self.point_mlp(x)  # (batch_size, point_mlp.out_features, num_points)
        
        return y, x_transformed, transform_mat



if __name__ == "__main__":
    
    x = torch.rand(2, 6, 10)
    
    b = Block(TNet(in_features=6,
                   mat_size=3,
                   agg_fn="max"),
              MLP2d(layer_sizes=[6, 64, 128],
                    add_bias=False,
                    apply_bn=True,
                    activation_fn="relu",
                    dropout_probs=None))
    y, x_transformed, transform_mat = b(x)
    print(y.shape)
    print(x_transformed.shape)
    print(transform_mat.shape)
    

    b = Block(None,
              MLP2d(layer_sizes=[6, 64, 128],
                    add_bias=False,
                    apply_bn=True,
                    activation_fn="relu",
                    dropout_probs=None))
    y, x_transformed, transform_mat = b(x)
    print(y.shape)
    print(x_transformed)
    print(transform_mat)