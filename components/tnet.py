import torch
import torch.nn as nn
from mlp import MLP, PointMLP


class _TNetBase(nn.Module):
    """
    Base class for the Transformation Network (TNet).
    
    Each point-cloud of a batch is transformed by its own transformation
    matrix of dimension t_size x t_size.
    
    Note that only the first t_size features of each point are transformed.
    E.g. if the points have 6 features (x, y, z, r, g, b) and t_size = 3,
    then only the (x, y, z) will be transformed.
        
    Parameters
    ----------       
    t_size : int
        Size of the (square) transformation matrix.
          
    point_mlp : PointMLP
        Point-MLP that is applied on each point, before the aggregation function.

    agg_fn : str 
        Aggregation function from {'max', 'avg'}.
        
    mlp : MLP
        MLP applied on the global feature vector, after the aggregation function.
    """
    
    # Aggregation is done over the points
    AGG = {"max": nn.AdaptiveMaxPool1d(output_size=1),
           "avg": nn.AdaptiveAvgPool1d(output_size=1)}
    
    def __init__(self, t_size, point_mlp, agg_fn, mlp):
        
        # Checks
        if not isinstance(t_size, int) or t_size < 1:
            raise TypeError("The t_size must be a positive integer > 0.")
                    
        if not isinstance(mlp, MLP):
            raise TypeError("The mlp must be an instance of the MLP class.")
        
        if mlp.out_features != t_size**2:
            raise ValueError(
                f"The output size of the mlp must be equal to t_size^2 = {t_size**2}.")
            
        if not isinstance(point_mlp, PointMLP):
            raise TypeError("The point_mlp must be an instance of the PointMLP class.")
            
        if point_mlp.out_features != mlp.in_features:
            raise ValueError("point_mlp.out_features should be equal to mlp.in_features.")
            
        if point_mlp.in_features < t_size:
            raise ValueError("point_mlp.in_features should be greater or equal to t_size.")
            
        if not isinstance(agg_fn, str) or agg_fn not in self.AGG.keys():
            raise TypeError(f"The agg_fn must be a string from {set(self.AGG.keys())}.")
                 
        # Initialization
        super().__init__()
        self.t_size = t_size
        self.point_mlp = point_mlp
        self.agg_fn = self.AGG[agg_fn]
        self.mlp = mlp
        
        # The transformation matrix is initialized as identity transformation.
        nn.init.constant_(self.mlp.net[-1][0].weight, 0.0)
        nn.init.eye_(self.mlp.net[-1][0].bias.view(t_size, t_size))      
                   

    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features, num_points) with in_features >= t_size.

        Returns
        -------            
        t : torch.Tensor
            Transformation matrix of shape (batch_size, t_size, t_size).
            
        x_t : torch.Tensor
            Transformed input x, same shape as input.
        """
        
        batch_size = x.shape[0]
 
        t = self.point_mlp.net(x)  # (batch_size, point_mlp.out_features, num_points)
        t = self.agg_fn(t).squeeze(dim=-1)  # (batch_size, point_mlp.out_features)
        t = self.mlp.net(t)  # (batch_size, t_size**2)
        t = t.view(batch_size, self.t_size, self.t_size)  # (batch_size, t_size, t_size)
        
        if x.shape[1] == self.t_size:
            x_t = torch.bmm(t, x)
        else:
            # In this case x.shape[1] > self.t_size, so we transform only the first t_size features
            x_t = torch.cat([torch.bmm(t, x[:, :self.t_size, :]), x[:, self.t_size:, :]], dim=1)
            
        return t, x_t



class TNet(_TNetBase):
    """
    The original TNet from the PointNet paper.
    """
    
    def __init__(self, in_features, t_size, agg_fn="max"):
        
        point_mlp = PointMLP(layer_sizes=[in_features, 64, 128, 1024],
                             add_bias=False,
                             apply_bn=True,
                             activation_fn="relu",
                             dropout_probs=None)
        
        mlp = MLP(layer_sizes=[1024, 512, 256, t_size**2],
                  add_bias=[False, False, True],
                  apply_bn=[True, True, False],
                  activation_fn=["relu", "relu", ""],
                  dropout_probs=None)
        
        super().__init__(t_size, point_mlp, agg_fn, mlp)
        
    
 
if __name__ == "__main__":
    
    x = torch.rand(5, 3, 100)
    
    tnet_3d = TNet(in_features=3, t_size=2, agg_fn="max")
    tnet_3d.eval()
    
    t, x_t = tnet_3d(x)
    print(f"t shape: {tuple(t.shape)}")
    print(f"x_t shape: {tuple(x_t.shape)}")
