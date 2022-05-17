import torch
import torch.nn as nn
from mlp import PointMLP
from tnet import _TNetBase, TNet


class Block(nn.Module):
    """
    Basic PointNet block: TNet + pointMLP
    """
    
    def __init__(self, tnet, point_mlp):
        
        # Checks
        if tnet is not None and not isinstance(tnet, _TNetBase):
            raise TypeError("The tnet can be None or an instance of the _TNetBase class.")
        
        if not isinstance(point_mlp, PointMLP):
            raise TypeError("The point_mlp should be an instance of the PointMLP class.")
        
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
        outputs : list of:
        
            t : torch.Tensor or None
                Transformation matrix of (batch_size, tnet.mat_size, tnet.mat_size) from the TNet, or None.
        
            x_t : torch.Tensor or None
                Transformed input x of the same shape as the input's shape, (batch_size, in_features, num_points), or None.
                
            y : list of torch.Tensor
                List of outputs from all the layers (hidden and output) of the PointMLP, [y_1, y_2, ..., y_out].
        """

        t = None        
        x_t = None
        
        if self.tnet:
            t, x_t = self.tnet(x)
            y = self.point_mlp(x_t)  # (batch_size, point_mlp.out_features, num_points)
        else:
            y = self.point_mlp(x)  # (batch_size, point_mlp.out_features, num_points)
            
        outputs = [t, x_t] + [y]
        
        return outputs



if __name__ == "__main__":
    
    x = torch.rand(2, 6, 10)
    
    b = Block(TNet(in_features=6,
                   t_size=3,
                   agg_fn="max"),
              PointMLP(layer_sizes=[6, 64, 128],
                       add_bias=False,
                       apply_bn=True,
                       activation_fn="relu",
                       dropout_probs=None))
    
    t, x_t, y = b(x)
    y_1, y_out = y
    print(t.shape)
    print(x_t.shape)
    print(y_1.shape)
    print(y_out.shape)
    

    b = Block(None,
              PointMLP(layer_sizes=[6, 64, 128],
                       add_bias=False,
                       apply_bn=True,
                       activation_fn="relu",
                       dropout_probs=None))
    t, x_t, y = b(x)
    y_1, y_out = y
    print(t)
    print(x_t)
    print(y_1.shape)
    print(y_out.shape)